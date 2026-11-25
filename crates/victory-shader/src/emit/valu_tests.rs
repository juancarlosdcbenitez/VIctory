//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, and turn-range reduction for `v_sin_*`/`v_cos_*`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        probe: None,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        probe: lowered.probe,
    };

    if let Some(mut v) = validator::Validator::for_tests() {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Whether a GPU test whose WARP half cannot run (no WARP device, or WARP
/// without a capability the test needs) fails instead of skipping that
/// half: Windows CI sets `VICTORY_WARP_REQUIRED=1`, since its runners always
/// have WARP, so a test that passed there ran on it.
pub(super) fn warp_required() -> bool {
    std::env::var("VICTORY_WARP_REQUIRED").is_ok_and(|v| v == "1")
}

/// The device for one half of a GPU test (`warp`: WARP, else the first
/// hardware adapter), or `None` to skip that half; see [`warp_required`].
pub(super) fn test_host(label: &str, warp: bool) -> Option<Host> {
    match Host::new(warp) {
        Ok(host) => Some(host),
        Err(e) if warp && warp_required() => {
            panic!("{label}: VICTORY_WARP_REQUIRED=1, but there is no WARP device: {e}")
        }
        Err(e) => {
            eprintln!("{label}: skipped: {e}");
            None
        }
    }
}

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        let kind = if host.adapter.1 {
            "software"
        } else {
            "hardware"
        };
        eprintln!("{label}: on {} ({kind})", host.adapter.0);
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// 64-bit image atomics (SH-1's lowering, behind its test switch) on an
/// R32G32_UINT texture UAV, the format D3D12 gives them, on each SM 6.6
/// device that supports them (WARP on CI): the texel each one changes and
/// the old value it returns match the evaluator's, which the evaluator
/// tests check against the ISA. Devices without the capability bits are
/// skipped, and say so.
#[test]
fn wide_image_atomics_on_the_host() {
    use super::synth::{eval_image, image_program, lower_native, mimg};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let op = |name: &str| {
        crate::gfx10::opcode(crate::gfx10::Encoding::Mimg, name).unwrap_or_else(|| panic!("{name}"))
    };
    // (op, x, y, value, compare): the texture starts zeroed.
    let cases: [(&str, u32, u32, u64, u64); 6] = [
        ("image_atomic_umax", 2, 3, 0xFEDC_BA98_7654_3210, 0),
        ("image_atomic_add", 1, 0, 0x1_0000_0001, 0),
        ("image_atomic_xor", 3, 3, 0x8000_0000_0000_0000, 0),
        ("image_atomic_smin", 0, 2, 0x8000_0000_0000_0005, 0),
        ("image_atomic_cmpswap", 2, 1, 0x1234_5678_9ABC_DEF0, 0),
        ("image_atomic_cmpswap", 0, 0, 0x1234_5678_9ABC_DEF0, 1),
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        let kind = if host.adapter.1 {
            "software"
        } else {
            "hardware"
        };
        if !host.typed_atomics_64() {
            assert!(
                !(warp && warp_required()),
                "{label}: VICTORY_WARP_REQUIRED=1, but {} has no 64-bit typed atomics",
                host.adapter.0
            );
            eprintln!(
                "{label}: skipped: {} ({kind}) has no 64-bit typed atomics",
                host.adapter.0
            );
            continue;
        }
        eprintln!("{label}: on {} ({kind})", host.adapter.0);
        for &(name, x, y, value, compare) in &cases {
            let cmpswap = name.ends_with("cmpswap");
            let mut body = vec![
                vop1(0x001, 5, 128 + x as u16),
                vop1(0x001, 6, 128 + y as u16),
            ];
            body.extend(mimg(
                op(name),
                1,
                5,
                12,
                if cmpswap { 0xF } else { 0x3 },
                1,
                true,
            ));
            body.push(vop1(0x001, 0, 256 + 1));
            body.push(vop1(0x001, 5, 256 + 2));
            super::TEST_IMAGE_ATOMICS_64.set(true);
            let lowered = lower_native(&image_program(&body));
            super::TEST_IMAGE_ATOMICS_64.set(false);
            let l = lowered.unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            let k = [
                value as u32,
                (value >> 32) as u32,
                compare as u32,
                (compare >> 32) as u32,
            ];
            let (want_out, want_texels) = eval_image(&l, k, [4, 4, 1], vec![0; 16 * 8])
                .unwrap_or_else(|e| panic!("{name}: evaluator: {e}"));
            let shader = HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil.clone(),
                user_data: l.user_data.clone(),
                samplers: l.samplers.clone(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs.clone(),
                root_srvs: l.root_srvs.clone(),
                probe: l.probe,
            };
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => k[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label} {name}: compute_pipeline failed: {e}"));
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(4), View::Uint2TextureUav(4, 4)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} {name}: dispatch failed: {e}"));
            let texels: Vec<u8> = out[1].iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(
                texels, want_texels,
                "{label} {name} at ({x}, {y}): the texture"
            );
            assert_eq!(
                out[0][..2],
                want_out[..2],
                "{label} {name}: the returned value"
            );
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device with 64-bit typed atomics; nothing ran");
    }
}

/// `eval_tests::contended_wide_image_atomics_keep_every_update` on each
/// device with 64-bit typed atomics (WARP on CI): 32 threads do one 64-bit
/// atomic each on the same texel, in whatever order the device runs them,
/// and what they leave must pass the same order-independent checks (no
/// update or carry lost, one cmpswap winner, each old value one the texel
/// held). Repeated, since an interleaving that loses an update need not
/// come every time.
#[test]
fn contended_wide_image_atomics_on_the_host() {
    use super::synth::{check_contended, contended_image_atomic, image_wave_program, user_data};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let op = |name: &str| {
        crate::gfx10::opcode(crate::gfx10::Encoding::Mimg, name).unwrap_or_else(|| panic!("{name}"))
    };
    const LANES: u32 = 32;
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        if !host.typed_atomics_64() {
            assert!(
                !(warp && warp_required()),
                "{label}: VICTORY_WARP_REQUIRED=1, but {} has no 64-bit typed atomics",
                host.adapter.0
            );
            eprintln!(
                "{label}: skipped: {} has no 64-bit typed atomics",
                host.adapter.0
            );
            continue;
        }
        eprintln!("{label}: on {}", host.adapter.0);
        for name in [
            "image_atomic_add",
            "image_atomic_cmpswap",
            "image_atomic_umax",
        ] {
            let body =
                contended_image_atomic(op(name), name.ends_with("cmpswap"), name.ends_with("add"));
            super::TEST_IMAGE_ATOMICS_64.set(true);
            let lowered = super::synth::lower_native(&image_wave_program(&body, LANES));
            super::TEST_IMAGE_ATOMICS_64.set(false);
            let l = lowered.unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            let shader = HostShader {
                stage: HwStage::Cs,
                dxil: l.dxil.clone(),
                user_data: l.user_data.clone(),
                samplers: l.samplers.clone(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs.clone(),
                root_srvs: l.root_srvs.clone(),
                probe: l.probe,
            };
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label} {name}: compute_pipeline failed: {e}"));
            for round in 0..8 {
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(user_data(&l, [0; 4])))],
                            views: &[View::RawUav(192), View::Uint2TextureUav(4, 4)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label} {name}: dispatch failed: {e}"));
                check_contended(name, LANES, &out[0], &out[1])
                    .unwrap_or_else(|e| panic!("{label}, round {round}: {e}"));
            }
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device with 64-bit typed atomics; nothing ran");
    }
}

/// Runs a lowered `synth::valu_program` on `host` for each input set: the
/// four stored dwords (`v0, v5, v6, v7`) per set.
fn run_lowered(host: &Host, l: &super::lower::Lowered, cases: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil.clone(),
        user_data: l.user_data.clone(),
        samplers: l.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs.clone(),
        root_srvs: l.root_srvs.clone(),
        probe: l.probe,
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
    cases
        .iter()
        .map(|&inputs| {
            let ud = super::synth::user_data(l, inputs);
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(4)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch");
            [out[0][0], out[0][1], out[0][2], out[0][3]]
        })
        .collect()
}

/// The first session's integer lowerings, checked until now on the
/// evaluator only (SH-7), on each SM 6.6 device (WARP on CI): every stored
/// dword matches the evaluator's for a spread of inputs. Integer
/// operations only, so no choice D3D leaves to drivers (NaN payloads,
/// signed zeros, float-to-integer saturation) can tell them apart.
#[test]
fn integer_lowerings_agree_with_the_evaluator_on_the_host() {
    use super::synth::{eval_valu, lower_valu, vgpr, vop3, vop3b};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const NULL: u8 = 0x7D;
    let v = |op| vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec();
    let programs: Vec<(&str, Vec<u32>)> = vec![
        ("v_mul_hi_u32_u24", v(0x10c)),
        ("v_mul_hi_i32_i24", v(0x10a)),
        ("v_add_nc_i32", v(0x37f)),
        ("v_sub_nc_i32", v(0x376)),
        ("v_mul_lo_i32", v(0x16b)),
        ("v_mad_u32_u16", v(0x373)),
        ("v_mad_i32_i16", v(0x375)),
        ("v_cvt_pk_u16_u32", v(0x36a)),
        ("v_cvt_pk_i16_i32", v(0x36b)),
        ("v_sad_u8", v(0x15a)),
        ("v_sad_hi_u8", v(0x15b)),
        ("v_sad_u16", v(0x15c)),
        ("v_sad_u32", v(0x15d)),
        ("v_lerp_u8", v(0x14d)),
        // v[6:7] = v1 * v2 + v[3:4], no carry out.
        (
            "v_mad_u64_u32",
            vop3b(0x176, 6, NULL, vgpr(1), vgpr(2), vgpr(3)).to_vec(),
        ),
        (
            "v_mad_i64_i32",
            vop3b(0x177, 6, NULL, vgpr(1), vgpr(2), vgpr(3)).to_vec(),
        ),
    ];
    let dwords = [
        0u32,
        1,
        2,
        0x7F,
        0x80,
        0xFF,
        0x7FFF,
        0x8000,
        0xFFFF,
        0x0001_0000,
        0x00FF_FFFF,
        0x0100_0000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xDEAD_BEEF,
        0xFFFF_FFFF,
    ];
    let cases: Vec<[u32; 4]> = (0..dwords.len())
        .map(|i| {
            let d = |k: usize| dwords[(i * 7 + k * 5) % dwords.len()];
            [d(0), d(1), d(2), d(3)]
        })
        .collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        let kind = if host.adapter.1 {
            "software"
        } else {
            "hardware"
        };
        eprintln!("{label}: on {} ({kind})", host.adapter.0);
        for (name, body) in &programs {
            let l = lower_valu(body).unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            let got = run_lowered(&host, &l, &cases);
            for (inputs, got) in cases.iter().zip(got) {
                let want = eval_valu(&l, *inputs).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(got, want, "{label} {name}({inputs:08x?})");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Where a float lowering's result lives, for comparing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Domain {
    /// An f32.
    F32,
    /// Integer bits: conversions to integers, packed integers.
    Int,
    /// An f16 in the low half; the high half compared as bits.
    F16Lo,
    /// Two f16 halves.
    F16x2,
}

/// How a device's result may differ from the evaluator's for the same
/// DXIL, each only where Microsoft's specifications leave it to the device
/// (the D3D11.3 functional specification, v1.17, and DXIL.rst).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Leeway {
    /// Both NaN, with other bits: "the exact bit pattern of the NaN is not
    /// required to stay the same" (D3D11.3 3.1.3.2); halves too (3.2.2:
    /// NaN converts to "NaN representation").
    NanPayload,
    /// A float operand is a denormal, or the evaluator's f32 result is one
    /// and the device gives the zero of its sign: "Denorms MUST be flushed
    /// to sign-preserved zero on input and output of any floating point
    /// mathematical operation" (3.1.3.2), under the default fp32 denormal
    /// mode the translated shaders keep; the evaluator preserves them.
    /// Halves keep denormals (3.1.5), so never for f16 results.
    DenormalFlush,
    /// Zeros of different signs from min/max-like operations: "it is
    /// permissible for the result of comparing zeros to be dependent on the
    /// order of parameters" (3.1.3.2).
    ZeroSign,
    /// `v_fma_f32`'s `FMad`, fused or not: DXIL.rst says it "is not fused
    /// for 'precise' operations", and the lowering's is not marked precise.
    /// The device's result must be one of the two roundings.
    FusedOrNot,
    /// An approximate operation within its D3D11.3 bound (chapter 22), plus
    /// the evaluator's own error (2^-23 relative): rcp, rsq and exp 2^-21
    /// relative; sqrt 1 ULP; log 2^-21 absolute on [0.5, 2], else
    /// relative; sin and cos 0.0008 absolute within [-100 pi, 100 pi]
    /// radians, and only "[-1 to +1]" outside it.
    Approximate,
    /// None of the above: a real disagreement.
    Other,
}

/// Whether `got` is within D3D11.3's bound for approximate operation `name`
/// of the evaluator's `want` (`x`: the input); `None` for other operations.
fn approximate(name: &str, x: f32, want: f32, got: f32) -> Option<bool> {
    let (w, g) = (f64::from(want), f64::from(got));
    let finite = want.is_finite() && got.is_finite();
    let ours = 2f64.powi(-23);
    let relative = |bound: f64| finite && (w - g).abs() <= w.abs() * (bound + ours);
    Some(match name {
        "v_rcp_f32" | "v_rsq_f32" | "v_exp_f32" => relative(2f64.powi(-21)),
        // The evaluator's sqrt is correctly rounded: one step apart at most.
        "v_sqrt_f32" => {
            finite
                && want.is_sign_negative() == got.is_sign_negative()
                && want.to_bits().abs_diff(got.to_bits()) <= 1
        }
        "v_log_f32" if (0.5..=2.0).contains(&x) => {
            finite && (w - g).abs() <= 2f64.powi(-21) + ours * w.abs().max(1.0)
        }
        "v_log_f32" => relative(2f64.powi(-21)),
        // The ISA's input is in revolutions.
        "v_sin_f32" | "v_cos_f32" => {
            let radians = f64::from(x) * std::f64::consts::TAU;
            if radians.abs() <= 100.0 * std::f64::consts::PI {
                finite && (w - g).abs() <= 0.0008 + ours
            } else {
                got.is_finite() && (-1.0..=1.0).contains(&got)
            }
        }
        _ => return None,
    })
}

/// Classifies a differing f32 result of float lowering `name` (`floats`:
/// which of its three operands are floats).
fn leeway_f32(name: &str, floats: u8, inputs: [u32; 4], want: u32, got: u32) -> Leeway {
    let (w, g) = (f32::from_bits(want), f32::from_bits(got));
    let denormal = |b: u32| b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0;
    if w.is_nan() && g.is_nan() {
        return Leeway::NanPayload;
    }
    let input = (0..3).any(|k| floats >> k & 1 != 0 && denormal(inputs[k]));
    let output = denormal(want) && got == want & 0x8000_0000;
    if input || output {
        return Leeway::DenormalFlush;
    }
    let min_max = matches!(
        name,
        "v_min_f32" | "v_max_f32" | "v_min3_f32" | "v_max3_f32" | "v_med3_f32"
    );
    if min_max && w == 0.0 && g == 0.0 {
        return Leeway::ZeroSign;
    }
    if name == "v_fma_f32" {
        let [a, b, c] = [0, 1, 2].map(|k| f32::from_bits(inputs[k]));
        if [a.mul_add(b, c), a * b + c]
            .iter()
            .any(|r| r.to_bits() == got)
        {
            return Leeway::FusedOrNot;
        }
    }
    match approximate(name, f32::from_bits(inputs[0]), w, g) {
        Some(true) => Leeway::Approximate,
        _ => Leeway::Other,
    }
}

/// Classifies a differing f16 result: only NaN bits may differ.
fn leeway_f16(want: u16, got: u16) -> Leeway {
    let nan = |h: u16| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
    if nan(want) && nan(got) {
        Leeway::NanPayload
    } else {
        Leeway::Other
    }
}

/// Classifies a differing result dword of a lowering whose result is in
/// `domain` (`floats`: which operands are floats).
fn leeway(name: &str, domain: Domain, floats: u8, inputs: [u32; 4], want: u32, got: u32) -> Leeway {
    let denormal = |b: u32| b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0;
    let halves = |x: u32| [x as u16, (x >> 16) as u16];
    match domain {
        Domain::F32 => leeway_f32(name, floats, inputs, want, got),
        // A conversion of a flushed operand.
        Domain::Int if (0..3).any(|k| floats >> k & 1 != 0 && denormal(inputs[k])) => {
            Leeway::DenormalFlush
        }
        Domain::Int => Leeway::Other,
        Domain::F16Lo if want >> 16 != got >> 16 => Leeway::Other,
        Domain::F16Lo => leeway_f16(want as u16, got as u16),
        Domain::F16x2 => {
            let (w, g) = (halves(want), halves(got));
            let kinds = [0, 1].map(|k| (w[k] != g[k]).then(|| leeway_f16(w[k], g[k])));
            kinds.into_iter().flatten().max().unwrap_or(Leeway::Other)
        }
    }
}

/// Runs each of `ops` (name, float operands, result domain) on each SM 6.6
/// device for `cases` and compares every stored dword with the evaluator's:
/// prints every difference with its [`Leeway`], and fails on any `Other`.
fn compare_float_lowerings(ops: &[(&str, u8, Domain)], cases: &[[u32; 4]]) {
    use super::synth::{vgpr, vop3};
    let bodies: Vec<(&str, Vec<u32>, u8, Domain)> = ops
        .iter()
        .map(|&(name, floats, domain)| {
            let op = crate::gfx10::opcode(crate::gfx10::Encoding::Vop3, name)
                .unwrap_or_else(|| panic!("{name}: no opcode"));
            (
                name,
                vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec(),
                floats,
                domain,
            )
        })
        .collect();
    compare_lowered_bodies(&bodies, cases);
}

/// [`compare_float_lowerings`] for programs encoded by the caller: (name,
/// body writing `v0` from `v1..v3`, float operands, result domain).
fn compare_lowered_bodies(ops: &[(&str, Vec<u32>, u8, Domain)], cases: &[[u32; 4]]) {
    use super::synth::{eval_valu, lower_valu};
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        let kind = if host.adapter.1 {
            "software"
        } else {
            "hardware"
        };
        eprintln!("{label}: on {} ({kind})", host.adapter.0);
        let mut other = Vec::new();
        let mut counts: std::collections::BTreeMap<(&str, Leeway), usize> =
            std::collections::BTreeMap::new();
        for (name, body, floats, domain) in ops {
            let (name, floats, domain) = (*name, *floats, *domain);
            let l = lower_valu(body).unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            let got = run_lowered(&host, &l, cases);
            for (inputs, got) in cases.iter().zip(got) {
                let want = eval_valu(&l, *inputs).unwrap_or_else(|e| panic!("{name}: {e}"));
                for k in 0..4 {
                    if got[k] == want[k] {
                        continue;
                    }
                    let kind = if k == 0 {
                        leeway(name, domain, floats, *inputs, want[k], got[k])
                    } else {
                        Leeway::Other
                    };
                    let line = format!(
                        "{name}({inputs:08x?}) dword {k}: evaluator {:#010x}, {label} {:#010x}: {kind:?}",
                        want[k], got[k]
                    );
                    eprintln!("{label}: {line}");
                    *counts.entry((name, kind)).or_default() += 1;
                    if kind == Leeway::Other {
                        other.push(line);
                    }
                }
            }
        }
        for ((name, kind), count) in &counts {
            eprintln!("{label}: {name}: {count} x {kind:?}");
        }
        assert!(
            other.is_empty(),
            "{label}: float lowerings disagree with the evaluator:\n{}",
            other.join("\n")
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// The float lowerings (SH-7) on each SM 6.6 device (WARP on CI) against
/// the evaluator, 43 of them for 48 input sets of special and ordinary
/// values (zeros, denormals, infinities, NaNs, ties, f16 rounding and range
/// edges, integer range edges): every stored dword the same but where
/// Microsoft's specifications leave the result to the device ([`Leeway`],
/// each with its citation), f16 results compared as halves. Runs 16 and 17
/// found three conversions disagreeing; their lowerings were wrong (see
/// `f16_nearest_even` and `int_to_f32` in the lowering).
#[test]
fn float_lowerings_agree_with_the_evaluator_on_the_host() {
    use Domain::{F16Lo, F16x2, F32, Int};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (name, which of the three operands are floats, result)
    let ops: [(&str, u8, Domain); 43] = [
        ("v_add_f32", 0b011, F32),
        ("v_sub_f32", 0b011, F32),
        ("v_mul_f32", 0b011, F32),
        ("v_mul_legacy_f32", 0b011, F32),
        ("v_min_f32", 0b011, F32),
        ("v_max_f32", 0b011, F32),
        ("v_fma_f32", 0b111, F32),
        ("v_mad_f32", 0b111, F32),
        ("v_min3_f32", 0b111, F32),
        ("v_max3_f32", 0b111, F32),
        ("v_med3_f32", 0b111, F32),
        ("v_ldexp_f32", 0b001, F32),
        ("v_frexp_mant_f32", 0b001, F32),
        ("v_frexp_exp_i32_f32", 0b001, Int),
        ("v_fract_f32", 0b001, F32),
        ("v_floor_f32", 0b001, F32),
        ("v_ceil_f32", 0b001, F32),
        ("v_trunc_f32", 0b001, F32),
        ("v_rndne_f32", 0b001, F32),
        ("v_cvt_f32_i32", 0, F32),
        ("v_cvt_f32_u32", 0, F32),
        ("v_cvt_i32_f32", 0b001, Int),
        ("v_cvt_u32_f32", 0b001, Int),
        ("v_cvt_flr_i32_f32", 0b001, Int),
        ("v_cvt_rpi_i32_f32", 0b001, Int),
        ("v_cvt_f16_f32", 0b001, F16Lo),
        ("v_cvt_f32_f16", 0, F32),
        ("v_cvt_pkrtz_f16_f32", 0b011, F16x2),
        ("v_cvt_pknorm_i16_f32", 0b011, Int),
        ("v_cvt_pknorm_u16_f32", 0b011, Int),
        ("v_cvt_pk_u8_f32", 0b001, Int),
        ("v_cvt_f32_ubyte0", 0, F32),
        ("v_cubeid_f32", 0b111, F32),
        ("v_cubesc_f32", 0b111, F32),
        ("v_cubetc_f32", 0b111, F32),
        ("v_cubema_f32", 0b111, F32),
        ("v_sqrt_f32", 0b001, F32),
        ("v_rsq_f32", 0b001, F32),
        ("v_rcp_f32", 0b001, F32),
        ("v_exp_f32", 0b001, F32),
        ("v_log_f32", 0b001, F32),
        ("v_sin_f32", 0b001, F32),
        ("v_cos_f32", 0b001, F32),
    ];
    let values: [u32; 40] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x3F80_0000, // 1
        0xBF80_0000, // -1
        0x3F00_0000, // 0.5
        0xBF00_0000, // -0.5
        0x3FC0_0000, // 1.5
        0x4020_0000, // 2.5
        0xC020_0000, // -2.5
        0x4060_0000, // 3.5
        0x3DCC_CCCD, // 0.1
        0xBDCC_CCCD, // -0.1
        0x0000_0001, // the smallest denormal (also the integer 1)
        0x007F_FFFF, // the largest denormal
        0x8000_0001, // a negative denormal
        0x0080_0000, // the smallest normal
        0x7F7F_FFFF, // the largest finite
        0xFF7F_FFFF,
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0000, // a quiet NaN
        0xFFC0_0000, // a negative quiet NaN
        0x7F80_0001, // a signaling NaN
        0x4F00_0000, // 2^31
        0x4F80_0000, // 2^32
        0xCF00_0000, // -2^31
        0x4F7F_FFFF, // the largest float below 2^32
        0x477F_E000, // 65504, the largest f16
        0x477F_F000, // 65520, which rounds to the f16 infinity
        0x3F80_1000, // 1 + 2^-11: an f16 tie
        0x3F80_1800, // 1 + 3 * 2^-12: rounds up to nearest, down toward zero
        0x3F80_3000, // 1 + 3 * 2^-11: a tie to the even f16 above
        0x4049_0FDB, // pi
        0x42C8_8000, // 100.25
        0x5015_02F9, // 1e10
        0xD015_02F9, // -1e10
        2,
        3,
        0xFFFF_FFFF, // -1 as an integer
        0xFFFF_FF6B, // -149 as an integer
    ];
    let n = values.len();
    let cases: Vec<[u32; 4]> = (0..48)
        .map(|i| {
            [
                values[i % n],
                values[(i * 7 + 3) % n],
                values[(i * 13 + 5) % n],
                values[(i * 3 + 1) % n],
            ]
        })
        .collect();
    compare_float_lowerings(&ops, &cases);
}

/// Packed f16 arithmetic (VOP3P, each half on its own) on each SM 6.6
/// device (WARP on CI) against the evaluator: both halves rounded to
/// nearest even by the lowering's own conversion, so to the bit but for
/// NaN bits. Every operand's two halves come from the same edge cases.
#[test]
fn packed_f16_lowerings_agree_with_the_evaluator_on_the_host() {
    use super::synth::{vgpr, vop3p};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let ops: Vec<(&str, Vec<u32>, u8, Domain)> = ["v_pk_add_f16", "v_pk_mul_f16", "v_pk_fma_f16"]
        .into_iter()
        .map(|name| {
            let op = crate::gfx10::opcode(crate::gfx10::Encoding::Vop3p, name)
                .unwrap_or_else(|| panic!("{name}: no opcode"));
            let body = vop3p(op as u8, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec();
            (name, body, 0, Domain::F16x2)
        })
        .collect();
    let halves: [u32; 20] = [
        0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x0E00, 0x3E00, 0x1400, 0x7BFF,
        0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x1000, 0x1800, 0x5BFF, 0x2E66, 0xB401,
    ];
    let n = halves.len();
    let cases: Vec<[u32; 4]> = (0..60)
        .map(|i| {
            let h = |k: usize| halves[k % n] | halves[(k * 11 + 7) % n] << 16;
            [h(i), h(i * 7 + 3), h(i * 13 + 5), 0]
        })
        .collect();
    compare_lowered_bodies(&ops, &cases);
}

/// f16 arithmetic on each SM 6.6 device (WARP on CI) against the
/// evaluator: its results are rounded to nearest even by the lowering's own
/// conversion (D3D's rounds toward zero), so they must agree to the bit but
/// for NaN bits. Operands are halves in the low 16 bits (the high bits
/// filled, to be ignored): ties after 1.0, denormals, the largest finite
/// half, infinities, NaNs, zeros of both signs.
#[test]
fn f16_lowerings_agree_with_the_evaluator_on_the_host() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let ops = [
        ("v_add_f16", 0, Domain::F16Lo),
        ("v_sub_f16", 0, Domain::F16Lo),
        ("v_mul_f16", 0, Domain::F16Lo),
        ("v_fma_f16", 0, Domain::F16Lo),
    ];
    let halves: [u32; 20] = [
        0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x0E00, 0x3E00, 0x1400, 0x7BFF,
        0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x1000, 0x1800, 0x5BFF, 0x2E66, 0xB401,
    ];
    let n = halves.len();
    let cases: Vec<[u32; 4]> = (0..60)
        .map(|i| {
            let h = |k: usize| halves[k % n] | 0xABCD_0000;
            [h(i), h(i * 7 + 3), h(i * 13 + 5), 0]
        })
        .collect();
    compare_float_lowerings(&ops, &cases);
}

/// Probes (GPU-1) on each SM 6.6 device (WARP on CI): a store through the
/// descriptor at heap index `s1 >> 5`, with three UAVs in the heap and the
/// probe constants saying the heap has 3 slots (the index is in range) or
/// 2 (it is past the heap, though the device's heap holds it, so nothing
/// faults). The probe buffer the device leaves matches the evaluator's
/// word for word, and the store lands at index 2 either way: nothing is
/// clamped.
#[test]
fn probes_record_heap_indices_past_the_bound_heap_on_the_host() {
    use super::synth::{buffer_store_dword, eval_probed, lower_valu, user_data};
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // s_load_dwordx4 s[12:15], s[4:5], s1; buffer_store_dword v3 (= s2).
    let mut body = vec![0x3D << 26 | 0x02 << 18 | 12 << 6 | 2, 1 << 25];
    body.extend(buffer_store_dword(3, 12, 0));
    super::TEST_PROBES.set(true);
    let lowered = lower_valu(&body);
    super::TEST_PROBES.set(false);
    let l = lowered.unwrap_or_else(|e| panic!("does not lower with probes: {e}"));
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil.clone(),
        user_data: l.user_data.clone(),
        samplers: l.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs.clone(),
        root_srvs: l.root_srvs.clone(),
        probe: l.probe,
    };
    assert!(shader.probe.is_some());
    let param = Host::probe_param(&[&shader]);
    let dwords = super::probe::DWORDS as u32;
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Some(host) = test_host(label, warp) else {
            continue;
        };
        let kind = if host.adapter.1 {
            "software"
        } else {
            "hardware"
        };
        eprintln!("{label}: on {} ({kind})", host.adapter.0);
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for slots in [3u32, 2] {
            let inputs = [0, 64, 0x5EED, 0];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(user_data(&l, inputs))),
                            (param, Root::Constants(vec![slots, 16])),
                            (param + 1, Root::Uav(dwords)),
                        ],
                        views: &[View::RawUav(4), View::RawUav(4), View::RawUav(4)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[2][0], 0x5EED,
                "{label}, {slots} slots: the store at index 2"
            );
            let extra = [(1, vec![0; 16]), (2, vec![0; 16])];
            let (_, want, _) = eval_probed(&l, inputs, [slots, 16], &extra)
                .unwrap_or_else(|e| panic!("evaluator: {e}"));
            assert_eq!(out[3], want, "{label}, {slots} slots: the probe buffer");
            let past = want[super::probe::PAST_HEAP];
            assert_eq!(
                past > 0,
                slots == 2,
                "{label}, {slots} slots: {past} counted"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
