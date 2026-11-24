//! Lowered VALU and SALU code run by the module evaluator
//! ([`super::bitcode::eval`]) against references: on any machine, without
//! a GPU.
//!
//! The first tests replay cases `valu_tests` checks on a host GPU and WARP
//! (expected values the hardware gave), so the evaluator's choices are
//! known to agree with real drivers on them. The rest check instructions
//! against references written from AMD's RDNA2 ISA descriptions.

use super::bitcode::eval::f16_to_f32;
use super::synth::{
    LITERAL, eval_cases, eval_valu, eval_wave, lower_valu, s_mov_b32, sgpr, sop1, sop2, sopc, sopk,
    sopp, vgpr, vop1, vop2, vop3, vop3b, vop3p,
};

/// A named instruction with its opcode and a reference, in test tables.
type Op1 = (&'static str, u16, fn(u32) -> u32);
type Op3 = (&'static str, u16, fn(u32, u32, u32) -> u32);
type Sop1 = (&'static str, u8, fn(u32) -> u32);
type Sop2 = (&'static str, u8, fn(u32, u32) -> u32);
type SccOp = (&'static str, u8, fn(u32, u32) -> bool);

/// Runs `body` for each case and compares `v0` with `want(inputs)`.
fn check_v0(what: &str, body: &[u32], cases: &[[u32; 4]], want: impl Fn([u32; 4]) -> u32) {
    for (inputs, got) in cases.iter().zip(eval_cases(body, cases)) {
        let want = want(*inputs);
        assert_eq!(
            got[0], want,
            "{what}({inputs:08x?}) = {:#010x}, want {want:#010x}",
            got[0]
        );
    }
}

/// A spread of interesting dwords: zero, ones, signs, small and large
/// magnitudes, and a few float patterns.
const DWORDS: [u32; 16] = [
    0,
    1,
    2,
    3,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0xFFFF_FFFE,
    0x1234_5678,
    0xDEAD_BEEF,
    0x3F80_0000, // 1.0
    0xC2F6_E979, // -123.456
];

/// Every pair (and a third operand) from [`DWORDS`].
fn pairs() -> Vec<[u32; 4]> {
    let mut v = Vec::new();
    for (i, &a) in DWORDS.iter().enumerate() {
        for &b in &DWORDS {
            v.push([a, b, DWORDS[(i * 7 + 3) % DWORDS.len()], 0]);
        }
    }
    v
}

// --- Cases the GPU tests check on hardware (valu_tests) -----------------

#[test]
fn hardware_cases_v_med3_f32_with_nan() {
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;
    let cases: &[([u32; 4], Option<f32>)] = &[
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        ([nan, nan, nan, 0], None),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    let got = eval_cases(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)), &inputs);
    for ((inputs, want), got) in cases.iter().zip(got) {
        let got = f32::from_bits(got[0]);
        match want {
            Some(w) => assert_eq!(got, *w, "v_med3_f32{inputs:08x?}"),
            None => assert!(got.is_nan(), "v_med3_f32{inputs:08x?} = {got}"),
        }
    }
}

#[test]
fn hardware_cases_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    for (name, op, want) in [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ] {
        check_v0(
            name,
            &vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)),
            &cases,
            |_| want,
        );
    }
}

#[test]
fn hardware_cases_v_cmp_lg_is_false_for_nan() {
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([f(f32::INFINITY), f(f32::NEG_INFINITY), 0, 0], 1),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0),
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0),
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        for ((inputs, want), got) in cases.iter().zip(eval_cases(&body, &inputs)) {
            assert_eq!(got[0], *want, "{what}{inputs:08x?}");
        }
    }
}

#[test]
fn hardware_cases_v_fract_f32_never_returns_one() {
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;
    let cases: &[(u32, Option<u32>)] = &[
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    for ((x, want), got) in cases
        .iter()
        .zip(eval_cases(&[vop1(0x020, 0, vgpr(1))], &inputs))
    {
        match want {
            Some(w) => assert_eq!(got[0], *w, "v_fract_f32({x:#010x})"),
            None => assert!(f32::from_bits(got[0]).is_nan(), "v_fract_f32({x:#010x})"),
        }
    }
}

#[test]
fn hardware_cases_v_cvt_u16_and_i16_f16_saturate() {
    let cases: &[(u16, u16, u16)] = &[
        (0x0000, 0, 0),
        (0x8000, 0, 0),
        (0x3E00, 1, 1),
        (0xBE00, 0, 0xFFFF),
        (0x57D0, 125, 125),
        (0x7800, 32768, 32767),
        (0x77FF, 32752, 32752),
        (0xF800, 0, 0x8000),
        (0xF801, 0, 0x8000),
        (0x7BFF, 65504, 32767),
        (0xFBFF, 0, 0x8000),
        (0x7C00, 0xFFFF, 32767),
        (0xFC00, 0, 0x8000),
        (0x7E00, 0, 0),
        (0xFE00, 0, 0),
        (0x0001, 0, 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
    for (op, unsigned) in [(0x052u16, true), (0x053, false)] {
        let body = [vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))];
        for (c, got) in cases.iter().zip(eval_cases(&body, &inputs)) {
            let want = if unsigned { c.1 } else { c.2 };
            assert_eq!(got[0], u32::from(want), "op {op:#x}({:#06x})", c.0);
        }
    }
}

#[test]
fn hardware_cases_v_sin_and_v_cos_f32_reduce_large_turns() {
    let f = f32::to_bits;
    let r = std::f32::consts::FRAC_1_SQRT_2;
    for (op, cases) in [
        (
            0x035u16,
            [
                (0.0, 0.0),
                (0.25, 1.0),
                (255.0, 0.0),
                (255.25, 1.0),
                (-255.25, -1.0),
                (10_000.125, r),
            ],
        ),
        (
            0x036,
            [
                (0.0, 1.0),
                (0.25, 0.0),
                (255.0, 1.0),
                (255.5, -1.0),
                (-255.5, -1.0),
                (10_000.125, r),
            ],
        ),
    ] {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
        for ((x, want), got) in cases
            .iter()
            .zip(eval_cases(&[vop1(op, 0, vgpr(1))], &inputs))
        {
            let got = f32::from_bits(got[0]);
            assert!(
                (got - want).abs() < 1e-5,
                "op {op:#x}({x}) = {got}, want {want}"
            );
        }
    }
}

#[test]
fn hardware_cases_bfe_signed_and_unsigned() {
    let signed: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
        ([0x8000_0000, 4, 32, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = signed.iter().map(|c| c.0).collect();
    let body = vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3));
    for ((inputs, want), got) in signed.iter().zip(eval_cases(&body, &inputs)) {
        assert_eq!(got[0], *want, "v_bfe_i32{inputs:08x?}");
    }
    // s_bfe_i32 s6, s0, s1 (s1 = offset[4:0] | width[22:16]): 7-bit widths.
    let s_body = [sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))];
    let s_inputs: Vec<[u32; 4]> = signed
        .iter()
        .map(|&(i, _)| [i[0], (i[1] & 31) | ((i[2] & 0x7F) << 16), 0, 0])
        .collect();
    let s_want = |k: usize| if k == 7 { 0xf800_0000 } else { signed[k].1 };
    for (k, got) in eval_cases(&s_body, &s_inputs).into_iter().enumerate() {
        assert_eq!(got[0], s_want(k), "s_bfe_i32{:08x?}", s_inputs[k]);
    }
}

#[test]
fn hardware_cases_v_cmp_class_f16() {
    let mut body = vop3(0x08f, 6, vgpr(1), vgpr(2), 0).to_vec();
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(6)));
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0),
        (0xFDFF, 0),
        (0x7E00, 1),
        (0xFE01, 1),
        (0xFC00, 2),
        (0xBC00, 3),
        (0x8001, 4),
        (0x8000, 5),
        (0x0000, 6),
        (0x0001, 7),
        (0x3C00, 8),
        (0x7C00, 9),
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    for ((inputs, want), got) in cases.iter().zip(eval_cases(&body, &inputs)) {
        assert_eq!(got[0], *want, "v_cmp_class_f16{inputs:08x?}");
    }
}

#[test]
fn hardware_cases_packed_add_selects_halves_of_inline_constants() {
    for (words, want) in [
        ([0xcc0f4000u32, 0x1a0202f8], 0x0000_3118u32),
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ] {
        let got = eval_cases(&words, &[[0; 4]]);
        assert_eq!(got[0][0], want, "{words:08x?}");
    }
}

// --- Instructions checked against ISA references -------------------------

#[test]
fn integer_vop2_and_vop3_match_references() {
    let cases = pairs();
    let ops: &[Op3] = &[
        ("v_add_nc_u32", 0x125, |a, b, _| a.wrapping_add(b)),
        ("v_sub_nc_u32", 0x126, |a, b, _| a.wrapping_sub(b)),
        ("v_subrev_nc_u32", 0x127, |a, b, _| b.wrapping_sub(a)),
        ("v_mul_lo_u32", 0x169, |a, b, _| a.wrapping_mul(b)),
        ("v_mul_hi_u32", 0x16a, |a, b, _| {
            ((u64::from(a) * u64::from(b)) >> 32) as u32
        }),
        ("v_mul_hi_i32", 0x16c, |a, b, _| {
            ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
        }),
        ("v_lshlrev_b32", 0x11a, |a, b, _| b << (a & 31)),
        ("v_lshrrev_b32", 0x116, |a, b, _| b >> (a & 31)),
        ("v_ashrrev_i32", 0x118, |a, b, _| {
            ((b as i32) >> (a & 31)) as u32
        }),
        ("v_max_i32", 0x112, |a, b, _| {
            (a as i32).max(b as i32) as u32
        }),
        ("v_min_u32", 0x113, |a, b, _| a.min(b)),
        ("v_mul_u32_u24", 0x10b, |a, b, _| {
            (a & 0xFF_FFFF).wrapping_mul(b & 0xFF_FFFF)
        }),
        ("v_mul_i32_i24", 0x109, |a, b, _| {
            let s = |x: u32| ((x << 8) as i32) >> 8;
            s(a).wrapping_mul(s(b)) as u32
        }),
        ("v_bfm_b32", 0x363, |a, b, _| {
            (((1u64 << (a & 31)) - 1) as u32) << (b & 31)
        }),
        ("v_alignbit_b32", 0x14e, |a, b, c| {
            ((u64::from(a) << 32 | u64::from(b)) >> (c & 31)) as u32
        }),
        ("v_alignbyte_b32", 0x14f, |a, b, c| {
            ((u64::from(a) << 32 | u64::from(b)) >> (8 * (c & 3))) as u32
        }),
        ("v_bfi_b32", 0x14a, |a, b, c| (a & b) | (!a & c)),
        ("v_lshl_add_u32", 0x346, |a, b, c| {
            (a << (b & 31)).wrapping_add(c)
        }),
        ("v_add3_u32", 0x36d, |a, b, c| {
            a.wrapping_add(b).wrapping_add(c)
        }),
        ("v_med3_u32", 0x159, |a, b, c| {
            let mut v = [a, b, c];
            v.sort_unstable();
            v[1]
        }),
        ("v_med3_i32", 0x158, |a, b, c| {
            let mut v = [a as i32, b as i32, c as i32];
            v.sort_unstable();
            v[1] as u32
        }),
    ];
    for &(name, op, reference) in ops {
        check_v0(
            name,
            &vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)),
            &cases,
            |[a, b, c, _]| reference(a, b, c),
        );
    }
}

#[test]
fn bit_counting_matches_references() {
    let cases: Vec<[u32; 4]> = DWORDS.iter().map(|&x| [x, 5, 0, 0]).collect();
    let ops: &[Op1] = &[
        ("v_bfrev_b32", 0x038, u32::reverse_bits),
        ("v_ffbh_u32", 0x039, |x| {
            if x == 0 { u32::MAX } else { x.leading_zeros() }
        }),
        ("v_ffbl_b32", 0x03a, |x| {
            if x == 0 { u32::MAX } else { x.trailing_zeros() }
        }),
        ("v_ffbh_i32", 0x03b, |x| {
            let y = if (x as i32) < 0 { !x } else { x };
            if y == 0 { u32::MAX } else { y.leading_zeros() }
        }),
    ];
    for &(name, op, reference) in ops {
        check_v0(name, &[vop1(op, 0, vgpr(1))], &cases, |[x, ..]| {
            reference(x)
        });
    }
    // v_bcnt_u32_b32 v0, v1, v2: bits of v1 plus v2.
    check_v0(
        "v_bcnt_u32_b32",
        &vop3(0x364, 0, vgpr(1), vgpr(2), 0),
        &cases,
        |[x, y, ..]| x.count_ones() + y,
    );
}

#[test]
fn literal_operands_reach_the_lowering() {
    // v_add_nc_u32 v0, 0x12345678, v1 (VOP2 0x25 with a literal).
    let body = [vop2(0x25, 0, LITERAL, 1), 0x1234_5678];
    check_v0(
        "v_add_nc_u32 literal",
        &body,
        &[[1, 0, 0, 0], [u32::MAX, 0, 0, 0]],
        |[a, ..]| a.wrapping_add(0x1234_5678),
    );
}

#[test]
fn scalar_bit_operations_match_references() {
    let cases = pairs();
    // SOP1 op s6, s0; v_mov_b32 v0, s6.
    let sop1_ops: &[Sop1] = &[
        ("s_brev_b32", 0x0b, u32::reverse_bits),
        ("s_bcnt1_i32_b32", 0x0f, u32::count_ones),
        ("s_ff1_i32_b32", 0x13, |x| {
            if x == 0 { u32::MAX } else { x.trailing_zeros() }
        }),
        ("s_flbit_i32_b32", 0x15, |x| {
            if x == 0 { u32::MAX } else { x.leading_zeros() }
        }),
        ("s_not_b32", 0x07, |x| !x),
        ("s_abs_i32", 0x34, |x| (x as i32).unsigned_abs()),
    ];
    for &(name, op, reference) in sop1_ops {
        check_v0(
            name,
            &[sop1(op, 6, 0), vop1(0x001, 0, sgpr(6))],
            &cases,
            |[a, ..]| reference(a),
        );
    }
    // SOP2 op s6, s0, s1; v_mov_b32 v0, s6.
    let sop2_ops: &[Sop2] = &[
        ("s_add_u32", 0x00, u32::wrapping_add),
        ("s_mul_i32", 0x26, u32::wrapping_mul),
        ("s_lshl_b32", 0x1e, |a, b| a << (b & 31)),
        ("s_ashr_i32", 0x22, |a, b| ((a as i32) >> (b & 31)) as u32),
        ("s_bfm_b32", 0x24, |a, b| {
            (((1u64 << (a & 31)) - 1) as u32) << (b & 31)
        }),
        ("s_bfe_u32", 0x27, |a, b| {
            let (o, w) = (b & 31, (b >> 16) & 0x7F);
            if w == 0 {
                0
            } else {
                ((u64::from(a) >> o) & ((1u64 << w.min(32)) - 1)) as u32
            }
        }),
        ("s_lshl2_add_u32", 0x2f, |a, b| (a << 2).wrapping_add(b)),
        ("s_mul_hi_u32", 0x35, |a, b| {
            ((u64::from(a) * u64::from(b)) >> 32) as u32
        }),
        ("s_andn2_b32", 0x14, |a, b| a & !b),
        ("s_xnor_b32", 0x1c, |a, b| !(a ^ b)),
    ];
    for &(name, op, reference) in sop2_ops {
        check_v0(
            name,
            &[sop2(op, 6, 0, 1), vop1(0x001, 0, sgpr(6))],
            &cases,
            |[a, b, ..]| reference(a, b),
        );
    }
}

#[test]
fn half_conversions_match_references() {
    // v_cvt_f32_f16 v0, v1 and v_cvt_f16_f32 v0, v1 (v0's high half 0).
    let halves: Vec<[u32; 4]> = (0..=u16::MAX)
        .step_by(97)
        .chain([0x7C00, 0xFC00, 0x7BFF, 0x0001, 0x8001, 0x03FF])
        .map(|h| [u32::from(h), 0, 0, 0])
        .collect();
    for (inputs, got) in halves
        .iter()
        .zip(eval_cases(&[vop1(0x00b, 0, vgpr(1))], &halves))
    {
        let want = f16_to_f32(inputs[0] as u16);
        let got = f32::from_bits(got[0]);
        assert!(
            got == want || (got.is_nan() && want.is_nan()),
            "v_cvt_f32_f16({:#06x}) = {got}, want {want}",
            inputs[0]
        );
    }
}

#[test]
fn programs_lower_for_the_evaluator() {
    // The harness itself: an empty body stores the untouched registers.
    let l = lower_valu(&[]).expect("an empty body lowers");
    let got = super::synth::eval_valu(&l, [1, 2, 3, 4]).expect("runs");
    assert_eq!(got, [0, 0, 0, 0], "v0, v5-v7 start at zero");
}

// --- Instructions added with the evaluator --------------------------------

/// The null SGPR (no carry out).
const NULL: u8 = 0x7D;

fn sext(bits: u32, x: u32) -> i32 {
    ((x << (32 - bits)) as i32) >> (32 - bits)
}

#[test]
fn added_integer_ops_match_references() {
    let cases = pairs();
    let ops: &[Op3] = &[
        ("v_mul_hi_u32_u24", 0x10c, |a, b, _| {
            ((u64::from(a & 0xFF_FFFF) * u64::from(b & 0xFF_FFFF)) >> 32) as u32
        }),
        ("v_mul_hi_i32_i24", 0x10a, |a, b, _| {
            ((i64::from(sext(24, a)) * i64::from(sext(24, b))) >> 32) as u32
        }),
        ("v_add_nc_i32", 0x37f, |a, b, _| a.wrapping_add(b)),
        ("v_sub_nc_i32", 0x376, |a, b, _| a.wrapping_sub(b)),
        ("v_mul_lo_i32", 0x16b, |a, b, _| a.wrapping_mul(b)),
        ("v_mad_u32_u16", 0x373, |a, b, c| {
            (a & 0xFFFF).wrapping_mul(b & 0xFFFF).wrapping_add(c)
        }),
        ("v_mad_i32_i16", 0x375, |a, b, c| {
            sext(16, a).wrapping_mul(sext(16, b)).wrapping_add(c as i32) as u32
        }),
        ("v_cvt_pk_u16_u32", 0x36a, |a, b, _| {
            a.min(0xFFFF) | b.min(0xFFFF) << 16
        }),
        ("v_cvt_pk_i16_i32", 0x36b, |a, b, _| {
            let sat = |x: u32| (x as i32).clamp(-32768, 32767) as u32 & 0xFFFF;
            sat(a) | sat(b) << 16
        }),
        ("v_sad_u8", 0x15a, |a, b, c| {
            let d: u32 = (0..4)
                .map(|i| ((a >> (8 * i)) & 0xFF).abs_diff((b >> (8 * i)) & 0xFF))
                .sum();
            d.wrapping_add(c)
        }),
        ("v_sad_hi_u8", 0x15b, |a, b, c| {
            let d: u32 = (0..4)
                .map(|i| ((a >> (8 * i)) & 0xFF).abs_diff((b >> (8 * i)) & 0xFF))
                .sum();
            (d << 16).wrapping_add(c)
        }),
        ("v_sad_u16", 0x15c, |a, b, c| {
            let d = (a & 0xFFFF).abs_diff(b & 0xFFFF) + (a >> 16).abs_diff(b >> 16);
            d.wrapping_add(c)
        }),
        ("v_sad_u32", 0x15d, |a, b, c| a.abs_diff(b).wrapping_add(c)),
        ("v_lerp_u8", 0x14d, |a, b, c| {
            (0..4)
                .map(|i| {
                    let f = |x: u32| (x >> (8 * i)) & 0xFF;
                    ((f(a) + f(b) + (f(c) & 1)) >> 1) << (8 * i)
                })
                .sum()
        }),
    ];
    for &(name, op, reference) in ops {
        check_v0(
            name,
            &vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)),
            &cases,
            |[a, b, c, _]| reference(a, b, c),
        );
    }
}

#[test]
fn added_16_bit_integer_ops_match_references() {
    let cases = pairs();
    let ops: &[Op3] = &[
        ("v_add_nc_i16", 0x30d, |a, b, _| a.wrapping_add(b)),
        ("v_sub_nc_i16", 0x30e, |a, b, _| a.wrapping_sub(b)),
        ("v_mad_u16", 0x340, |a, b, c| {
            a.wrapping_mul(b).wrapping_add(c)
        }),
        ("v_mad_i16", 0x35e, |a, b, c| {
            a.wrapping_mul(b).wrapping_add(c)
        }),
        ("v_min3_u16", 0x353, |a, b, c| {
            (a & 0xFFFF).min(b & 0xFFFF).min(c & 0xFFFF)
        }),
        ("v_max3_u16", 0x356, |a, b, c| {
            (a & 0xFFFF).max(b & 0xFFFF).max(c & 0xFFFF)
        }),
        ("v_min3_i16", 0x352, |a, b, c| {
            sext(16, a).min(sext(16, b)).min(sext(16, c)) as u32
        }),
        ("v_max3_i16", 0x355, |a, b, c| {
            sext(16, a).max(sext(16, b)).max(sext(16, c)) as u32
        }),
        ("v_med3_u16", 0x359, |a, b, c| {
            let mut v = [a & 0xFFFF, b & 0xFFFF, c & 0xFFFF];
            v.sort_unstable();
            v[1]
        }),
        ("v_med3_i16", 0x358, |a, b, c| {
            let mut v = [sext(16, a), sext(16, b), sext(16, c)];
            v.sort_unstable();
            v[1] as u32
        }),
    ];
    for &(name, op, reference) in ops {
        // The result is the low half; v0's high half (zero) is kept.
        check_v0(
            name,
            &vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)),
            &cases,
            |[a, b, c, _]| reference(a, b, c) & 0xFFFF,
        );
    }
}

#[test]
fn v_mad_u64_u32_and_v_mad_i64_i32_match_references() {
    // v_mad_*64_*32 v[6:7], null, v1, v2, v[3:4]: the addend is inputs 2-3.
    let mut cases = pairs();
    cases
        .iter_mut()
        .enumerate()
        .for_each(|(i, c)| c[3] = DWORDS[i % DWORDS.len()]);
    let got_u = eval_cases(&vop3b(0x176, 6, NULL, vgpr(1), vgpr(2), vgpr(3)), &cases);
    let got_i = eval_cases(&vop3b(0x177, 6, NULL, vgpr(1), vgpr(2), vgpr(3)), &cases);
    for ((c, u), i) in cases.iter().zip(got_u).zip(got_i) {
        let addend = u64::from(c[2]) | u64::from(c[3]) << 32;
        let want_u = (u64::from(c[0]) * u64::from(c[1])).wrapping_add(addend);
        let want_i =
            (i64::from(c[0] as i32) * i64::from(c[1] as i32)).wrapping_add(addend as i64) as u64;
        let got = |r: [u32; 4]| u64::from(r[2]) | u64::from(r[3]) << 32;
        assert_eq!(got(u), want_u, "v_mad_u64_u32{c:08x?}");
        assert_eq!(got(i), want_i, "v_mad_i64_i32{c:08x?}");
    }
    // The carry out (s20), selected into v0.
    let mut body = vop3b(0x176, 6, 20, vgpr(1), vgpr(2), vgpr(3)).to_vec();
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
    check_v0("v_mad_u64_u32 carry", &body, &cases, |[a, b, lo, hi]| {
        let sum = u128::from(a) * u128::from(b) + (u128::from(hi) << 32 | u128::from(lo));
        u32::from(sum > u128::from(u64::MAX))
    });
}

/// A spread of f32 bit patterns: zeros, denormals, normals around 1, the
/// largest finite, infinities, NaNs.
const FLOATS: [u32; 18] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x807F_FFFF,
    0x0080_0000,
    0x3F00_0000, // 0.5
    0x3F80_0000, // 1
    0xBF80_0000, // -1
    0x3F7F_FFFF,
    0x4049_0FDB, // pi
    0xC2F6_E979, // -123.456
    0x3E99_999A, // 0.3
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0x3F2A_AAAB, // 2/3
];

#[test]
fn v_ldexp_f32_matches_a_reference() {
    let exps: [i32; 16] = [
        0,
        1,
        -1,
        23,
        -23,
        126,
        -126,
        127,
        -127,
        149,
        -149,
        254,
        -254,
        300,
        i32::MAX,
        i32::MIN,
    ];
    let mut cases = Vec::new();
    for &x in &FLOATS {
        for &n in &exps {
            cases.push([x, n as u32, 0, 0]);
        }
    }
    let got = eval_cases(&vop3(0x362, 0, vgpr(1), vgpr(2), 0), &cases);
    for (c, got) in cases.iter().zip(got) {
        let x = f32::from_bits(c[0]);
        let n = (c[1] as i32).clamp(-400, 400);
        // Exact in f64, then rounded once.
        let want = (f64::from(x) * 2f64.powi(n)) as f32;
        let got = f32::from_bits(got[0]);
        if want.is_nan() {
            assert!(got.is_nan(), "ldexp({x:e}, {n}) = {got:e}");
        } else if want.is_subnormal() {
            // Two roundings are allowed for denormal results.
            let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(ulps <= 1, "ldexp({x:e}, {n}) = {got:e}, want {want:e}");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "ldexp({x:e}, {n}) = {got:e}, want {want:e}"
            );
        }
    }
}

#[test]
fn v_frexp_f32_matches_a_reference() {
    let cases: Vec<[u32; 4]> = FLOATS.iter().map(|&x| [x, 0, 0, 0]).collect();
    let mant = eval_cases(&[vop1(0x040, 0, vgpr(1))], &cases);
    let exp = eval_cases(&[vop1(0x03f, 0, vgpr(1))], &cases);
    for ((c, m), e) in cases.iter().zip(mant).zip(exp) {
        let x = f32::from_bits(c[0]);
        // Through f64, where every f32 (denormals too) is normal.
        let (want_m, want_e) = if x == 0.0 || !x.is_finite() {
            (x, 0)
        } else {
            let bits = f64::from(x).to_bits();
            let e = ((bits >> 52) & 0x7FF) as i32 - 1022;
            ((f64::from(x) / 2f64.powi(e)) as f32, e)
        };
        let got_m = f32::from_bits(m[0]);
        if want_m.is_nan() {
            assert!(got_m.is_nan(), "frexp_mant({x:e}) = {got_m:e}");
        } else {
            assert_eq!(
                got_m.to_bits(),
                want_m.to_bits(),
                "frexp_mant({x:e}) = {got_m:e}"
            );
        }
        assert_eq!(e[0] as i32, want_e, "frexp_exp({x:e})");
    }
}

#[test]
fn v_cvt_pknorm_f32_match_a_reference() {
    let mut extra = vec![
        0x3F00_0000u32, // 0.5: 16383.5 rounds to even
        0xBF00_0000,
        0x3E80_0000, // 0.25
        0x3800_0000, // 2^-15
        0x3780_0000, // 2^-16
        0x477F_FF00, // 65535
    ];
    extra.extend(FLOATS);
    let mut cases = Vec::new();
    for &a in &extra {
        for &b in &[0x3F80_0000u32, 0x3E99_999A, 0xBE99_999A] {
            cases.push([a, b, 0, 0]);
        }
    }
    let norm = |x: u32, signed: bool| -> u32 {
        let x = f32::from_bits(x);
        if x.is_nan() {
            return 0;
        }
        let (lo, scale) = if signed {
            (-1.0, 32767.0)
        } else {
            (0.0, 65535.0)
        };
        // Exact in f64 (24 + 16 bits), rounded once.
        let p = f64::from(x.clamp(lo, 1.0)) * scale;
        (p.round_ties_even() as i32) as u32 & 0xFFFF
    };
    for (op, signed) in [(0x368u16, true), (0x369, false)] {
        check_v0(
            if signed {
                "v_cvt_pknorm_i16_f32"
            } else {
                "v_cvt_pknorm_u16_f32"
            },
            &vop3(op, 0, vgpr(1), vgpr(2), 0),
            &cases,
            |[a, b, ..]| norm(a, signed) | norm(b, signed) << 16,
        );
    }
}

#[test]
fn dot_products_match_references() {
    let cases = pairs();
    let field = |x: u32, width: u32, i: u32, signed: bool| -> i64 {
        let f = (x >> (width * i)) & ((1u64 << width) - 1) as u32;
        if signed {
            i64::from(sext(width, f))
        } else {
            i64::from(f)
        }
    };
    let dot = |a: u32, b: u32, width: u32, signed: bool| -> u32 {
        (0..32 / width)
            .map(|i| field(a, width, i, signed) * field(b, width, i, signed))
            .sum::<i64>() as u32
    };
    for (name, op, width, signed) in [
        ("v_dot2_i32_i16", 0x14u8, 16u32, true),
        ("v_dot2_u32_u16", 0x15, 16, false),
        ("v_dot4_i32_i8", 0x16, 8, true),
        ("v_dot4_u32_u8", 0x17, 8, false),
        ("v_dot8_u32_u4", 0x19, 4, false),
    ] {
        check_v0(
            name,
            &vop3p(op, 0, vgpr(1), vgpr(2), vgpr(3)),
            &cases,
            |[a, b, c, _]| dot(a, b, width, signed).wrapping_add(c),
        );
    }
    // v_mov_b32 v0, v3; v_dot4c_i32_i8 v0, v1, v2 (accumulates into v0).
    check_v0(
        "v_dot4c_i32_i8",
        &[vop1(0x001, 0, vgpr(3)), vop2(0x0d, 0, vgpr(1), 2)],
        &cases,
        |[a, b, c, _]| dot(a, b, 8, true).wrapping_add(c),
    );
}

#[test]
fn added_scalar_ops_match_references() {
    let cases = pairs();
    let sop1_ops: &[Sop1] = &[
        ("s_sext_i32_i8", 0x19, |x| sext(8, x) as u32),
        ("s_sext_i32_i16", 0x1a, |x| sext(16, x) as u32),
        ("s_bcnt0_i32_b32", 0x0d, u32::count_zeros),
        ("s_ff0_i32_b32", 0x11, |x| {
            if x == u32::MAX {
                u32::MAX
            } else {
                x.trailing_ones()
            }
        }),
    ];
    for &(name, op, reference) in sop1_ops {
        check_v0(
            name,
            &[sop1(op, 6, 0), vop1(0x001, 0, sgpr(6))],
            &cases,
            |[a, ..]| reference(a),
        );
    }
    let sop2_ops: &[Sop2] = &[
        ("s_absdiff_i32", 0x2c, |a, b| {
            (a as i32).wrapping_sub(b as i32).wrapping_abs() as u32
        }),
        ("s_pack_ll_b32_b16", 0x32, |a, b| a & 0xFFFF | b << 16),
        ("s_pack_lh_b32_b16", 0x33, |a, b| {
            a & 0xFFFF | b & 0xFFFF_0000
        }),
        ("s_pack_hh_b32_b16", 0x34, |a, b| a >> 16 | b & 0xFFFF_0000),
    ];
    for &(name, op, reference) in sop2_ops {
        check_v0(
            name,
            &[sop2(op, 6, 0, 1), vop1(0x001, 0, sgpr(6))],
            &cases,
            |[a, b, ..]| reference(a, b),
        );
    }
}

/// Runs a scalar body leaving a 64-bit result in s[6:7] (v6, v7) and
/// compares it with `want(s[0:1], s2)`.
fn check_s64(what: &str, body: &[u32], want: impl Fn(u64, u32) -> u64) {
    let mut full = body.to_vec();
    full.extend([vop1(0x001, 6, sgpr(6)), vop1(0x001, 7, sgpr(7))]);
    let mut cases = Vec::new();
    for &a in &DWORDS {
        for &b in &[0u32, 1, 0x8000_0000, 0xFFFF_FFFF, 0x0F0F_1234] {
            for &c in &[
                0u32,
                1,
                31,
                32,
                33,
                63,
                64,
                0x0008_0004,
                0x0020_0010,
                0x0040_0020,
                0x007F_003F,
            ] {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (c, got) in cases.iter().zip(eval_cases(&full, &cases)) {
        let x = u64::from(c[0]) | u64::from(c[1]) << 32;
        let want = want(x, c[2]);
        let got = u64::from(got[2]) | u64::from(got[3]) << 32;
        assert_eq!(got, want, "{what}(s[0:1] = {x:#018x}, s2 = {:#x})", c[2]);
    }
}

#[test]
fn added_64_bit_scalar_ops_match_references() {
    check_s64("s_bfm_b64", &[sop2(0x25, 6, 2, 0)], |x, c| {
        // S0 = s2 (width), S1 = s0 (offset).
        let (w, o) = (c & 63, (x as u32) & 63);
        ((1u64 << w).wrapping_sub(1)) << o
    });
    let bfe = |x: u64, c: u32, signed: bool| -> u64 {
        let (o, w) = (c & 63, (c >> 16) & 0x7F);
        if w == 0 {
            return 0;
        }
        if w + o >= 64 {
            return if signed {
                ((x as i64) >> o) as u64
            } else {
                x >> o
            };
        }
        let t = x << (64 - w - o);
        if signed {
            ((t as i64) >> (64 - w)) as u64
        } else {
            t >> (64 - w)
        }
    };
    check_s64("s_bfe_u64", &[sop2(0x29, 6, 0, 2)], move |x, c| {
        bfe(x, c, false)
    });
    check_s64("s_bfe_i64", &[sop2(0x2a, 6, 0, 2)], move |x, c| {
        bfe(x, c, true)
    });
    check_s64("s_brev_b64", &[sop1(0x0c, 6, 0)], |x, _| x.reverse_bits());
    check_s64(
        "s_bitset0_b64",
        &[sop1(0x03, 6, 0), sop1(0x03, 7, 1), sop1(0x1c, 6, 2)],
        |x, c| x & !(1u64 << (c & 63)),
    );
    check_s64(
        "s_bitset1_b64",
        &[sop1(0x03, 6, 0), sop1(0x03, 7, 1), sop1(0x1e, 6, 2)],
        |x, c| x | 1u64 << (c & 63),
    );
    check_s64("s_bitreplicate_b64_b32", &[sop1(0x3b, 6, 2)], |_, c| {
        (0..32).fold(0u64, |r, i| r | (u64::from(c >> i & 1) * 3) << (2 * i))
    });
    // s_cmp_lg_u32 s2, 0 sets SCC; s_cmov_b64 s[6:7], s[0:1] moves on it
    // over s[6:7] = 0 (never written: zero).
    check_s64(
        "s_cmov_b64",
        &[sopc(0x07, 2, 0x80), sop1(0x06, 6, 0)],
        |x, c| {
            if c != 0 { x } else { 0 }
        },
    );
    // 32-bit results of 64-bit sources, in s6 (s7 never written: zero).
    check_s64("s_bcnt0_i32_b64", &[sop1(0x0e, 6, 0)], |x, _| {
        u64::from(x.count_zeros())
    });
    check_s64("s_ff0_i32_b64", &[sop1(0x12, 6, 0)], |x, _| {
        if x == u64::MAX {
            0xFFFF_FFFF
        } else {
            u64::from(x.trailing_ones())
        }
    });
    check_s64("s_flbit_i32_b64", &[sop1(0x16, 6, 0)], |x, _| {
        if x == 0 {
            0xFFFF_FFFF
        } else {
            u64::from(x.leading_zeros())
        }
    });
    check_s64("s_flbit_i32_i64", &[sop1(0x18, 6, 0)], |x, _| {
        let y = if (x as i64) < 0 { !x } else { x };
        if y == 0 {
            0xFFFF_FFFF
        } else {
            u64::from(y.leading_zeros())
        }
    });
}

#[test]
fn scalar_conditional_moves_follow_scc() {
    // s_cmp_eq_u32 s0, s1; s_mov_b32 s6, s2; s_cmov_b32 s6, s3 / s_cmovk_i32 s6, -5.
    let cases = pairs();
    let mut cmov = vec![sopc(0x06, 0, 1), sop1(0x03, 6, 2), sop1(0x05, 6, 3)];
    cmov.push(vop1(0x001, 0, sgpr(6)));
    check_v0(
        "s_cmov_b32",
        &cmov,
        &cases,
        |[a, b, c, d]| if a == b { d } else { c },
    );
    let mut cmovk = vec![sopc(0x06, 0, 1), sop1(0x03, 6, 2), sopk(0x02, 6, 0xFFFB)];
    cmovk.push(vop1(0x001, 0, sgpr(6)));
    check_v0("s_cmovk_i32", &cmovk, &cases, |[a, b, c, _]| {
        if a == b { (-5i32) as u32 } else { c }
    });
}

/// SCC after `s_min_*` and `s_max_*` is the comparison, and after
/// `s_lshl*_add_u32` the unsigned overflow (RDNA2 ISA), not "the result is
/// not zero" (what the lowering wrote before).
#[test]
fn scalar_scc_follows_the_isa() {
    let cases = pairs();
    // op s6, s0, s1; s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7.
    let ops: &[SccOp] = &[
        ("s_min_u32", 0x07, |a, b| a < b),
        ("s_min_i32", 0x06, |a, b| (a as i32) < (b as i32)),
        ("s_max_u32", 0x09, |a, b| a > b),
        ("s_max_i32", 0x08, |a, b| (a as i32) > (b as i32)),
        ("s_lshl1_add_u32", 0x2e, |a, b| {
            (u64::from(a) << 1) + u64::from(b) > 0xFFFF_FFFF
        }),
        ("s_lshl2_add_u32", 0x2f, |a, b| {
            (u64::from(a) << 2) + u64::from(b) > 0xFFFF_FFFF
        }),
        ("s_lshl4_add_u32", 0x31, |a, b| {
            (u64::from(a) << 4) + u64::from(b) > 0xFFFF_FFFF
        }),
        ("s_absdiff_i32", 0x2c, |a, b| a != b),
    ];
    for &(name, op, reference) in ops {
        let body = [
            sop2(op, 6, 0, 1),
            sop2(0x0a, 7, 0x81, 0x80),
            vop1(0x001, 0, sgpr(7)),
        ];
        check_v0(&format!("SCC of {name}"), &body, &cases, |[a, b, ..]| {
            u32::from(reference(a, b))
        });
    }
}

// --- Sweeps of existing instructions against the ISA ----------------------

/// The VOP3 opcode of `name` (VOPC, VOP2 and VOP1 operations included).
fn vop3_op(name: &str) -> u16 {
    crate::gfx10::opcode(crate::gfx10::Encoding::Vop3, name)
        .unwrap_or_else(|| panic!("{name} is not in the VOP3 table"))
}

/// `name` writes lane mask s20 from v1, v2 (64-bit forms: v[1:2], v[3:4]);
/// v_cndmask_b32 turns it into v0.
fn compare_body(name: &str, wide: bool) -> Vec<u32> {
    let b = if wide { vgpr(3) } else { vgpr(2) };
    let mut body = vop3(vop3_op(name), 20, vgpr(1), b, 0).to_vec();
    body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
    body
}

/// What predicate `p` (AMD's names) says for an ordering of two values:
/// `ord` is `None` when they are unordered (a NaN).
fn predicate(p: &str, ord: Option<std::cmp::Ordering>) -> bool {
    use std::cmp::Ordering::*;
    let is = |o: &[std::cmp::Ordering]| ord.is_some_and(|x| o.contains(&x));
    match p {
        "f" => false,
        "t" | "tru" => true,
        "lt" => is(&[Less]),
        "eq" => is(&[Equal]),
        "le" => is(&[Less, Equal]),
        "gt" => is(&[Greater]),
        "lg" => is(&[Less, Greater]),
        "ne" => !is(&[Equal]),
        "ge" => is(&[Greater, Equal]),
        "o" => ord.is_some(),
        "u" => ord.is_none(),
        "nge" => !is(&[Greater, Equal]),
        "nlg" => !is(&[Less, Greater]),
        "ngt" => !is(&[Greater]),
        "nle" => !is(&[Less, Equal]),
        "neq" => !is(&[Equal]),
        "nlt" => !is(&[Less]),
        p => panic!("predicate {p}"),
    }
}

const FLOAT_PREDICATES: [&str; 16] = [
    "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt",
    "tru",
];
const INT_PREDICATES: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];

#[test]
fn float_compares_match_the_isa() {
    let f32s = [
        0u32,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x0000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x7F80_0001,
        0x4049_0FDB,
    ];
    let f16s = [
        0u32, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x4248,
    ];
    let f16 = |h: u32| f16_to_f32(h as u16);
    for (ty, values, as_f) in [
        ("f32", &f32s, (|x: u32| f32::from_bits(x)) as fn(u32) -> f32),
        ("f16", &f16s, f16),
    ] {
        let mut cases = Vec::new();
        for &a in values {
            for &b in values {
                // High halves set, which 16-bit compares must ignore.
                let junk = if ty == "f16" { 0xA5A5_0000 } else { 0 };
                cases.push([a | junk, b | junk, 0, 0]);
            }
        }
        for p in FLOAT_PREDICATES {
            let name = format!("v_cmp_{p}_{ty}");
            check_v0(&name, &compare_body(&name, false), &cases, |[a, b, ..]| {
                let (x, y) = (
                    as_f(a & if ty == "f16" { 0xFFFF } else { !0 }),
                    as_f(b & if ty == "f16" { 0xFFFF } else { !0 }),
                );
                u32::from(predicate(p, x.partial_cmp(&y)))
            });
        }
    }
}

#[test]
fn integer_compares_match_the_isa() {
    let cases = pairs();
    for (ty, bits, signed) in [
        ("i32", 32, true),
        ("u32", 32, false),
        ("i16", 16, true),
        ("u16", 16, false),
    ] {
        for p in INT_PREDICATES {
            let name = format!("v_cmp_{p}_{ty}");
            // The 16-bit forms have no f and t.
            if crate::gfx10::opcode(crate::gfx10::Encoding::Vop3, &name).is_none() {
                assert_eq!(bits, 16, "{name}");
                continue;
            }
            check_v0(&name, &compare_body(&name, false), &cases, |[a, b, ..]| {
                let ord = if signed {
                    sext(bits, a).cmp(&sext(bits, b))
                } else {
                    let m = if bits == 32 { !0 } else { (1 << bits) - 1 };
                    (a & m).cmp(&(b & m))
                };
                u32::from(predicate(p, Some(ord)))
            });
        }
    }
}

#[test]
fn wide_integer_compares_match_the_isa() {
    // v[1:2] against v[3:4].
    let mut cases = Vec::new();
    for &a in &[
        0u64,
        1,
        u64::MAX,
        1 << 63,
        (1 << 32) - 1,
        1 << 32,
        0x1234_5678_9ABC_DEF0,
    ] {
        for &b in &[0u64, 1, u64::MAX, 1 << 63, 1 << 32, 0x1234_5678_0000_0000] {
            cases.push([a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]);
        }
    }
    for (ty, signed) in [("i64", true), ("u64", false)] {
        for p in INT_PREDICATES {
            let name = format!("v_cmp_{p}_{ty}");
            let l = lower_valu(&compare_body(&name, true))
                .unwrap_or_else(|e| panic!("{name} does not lower: {e}"));
            for c in &cases {
                let a = u64::from(c[0]) | u64::from(c[1]) << 32;
                let b = u64::from(c[2]) | u64::from(c[3]) << 32;
                let ord = if signed {
                    (a as i64).cmp(&(b as i64))
                } else {
                    a.cmp(&b)
                };
                let got = super::synth::eval_valu(&l, *c).expect("runs");
                assert_eq!(
                    got[0],
                    u32::from(predicate(p, Some(ord))),
                    "{name}({a:#x}, {b:#x})"
                );
            }
        }
    }
}

#[test]
fn v_cmp_class_f32_matches_the_isa() {
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // signalling NaN
        (0xFFBF_FFFF, 0),
        (0x7FC0_0000, 1), // quiet NaN
        (0xFFC0_0001, 1),
        (0xFF80_0000, 2),
        (0xBF80_0000, 3),
        (0x8000_0001, 4),
        (0x8000_0000, 5),
        (0x0000_0000, 6),
        (0x0000_0001, 7),
        (0x3F80_0000, 8),
        (0x7F80_0000, 9),
    ];
    let mut cases = Vec::new();
    for &(bits, class) in classes {
        cases.push([bits, 1 << class, 0, 0]);
        cases.push([bits, 0x3FF & !(1 << class), 0, 0]);
    }
    check_v0(
        "v_cmp_class_f32",
        &compare_body("v_cmp_class_f32", false),
        &cases,
        |[x, mask, ..]| {
            let class = classes.iter().find(|c| c.0 == x).expect("listed").1;
            u32::from(mask >> class & 1 != 0)
        },
    );
}

/// `f32` to binary16 bits rounded toward zero: magnitudes past the largest
/// finite half give it, infinities and NaNs stay.
fn f16_toward_zero(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    if x.is_nan() {
        return sign | 0x7E00 | ((b >> 13) & 0x3FF);
    }
    if x.is_infinite() {
        return sign | 0x7C00;
    }
    let a = x.abs();
    if a >= 65520.0 || f64::from(a) > 65504.0 {
        return sign | 0x7BFF;
    }
    // The largest half not above |x|, by search over the ordered encodings.
    let (mut lo, mut hi) = (0u32, 0x7BFF);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f16_to_f32(mid as u16) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    sign | lo
}

#[test]
fn conversions_match_the_isa() {
    let cases: Vec<[u32; 4]> = FLOATS
        .iter()
        .chain(&[
            0x4F00_0000u32, // 2^31
            0xCF00_0001,    // just below -2^31
            0x4F80_0000,    // 2^32
            0x3F00_0000,    // 0.5
            0xBF00_0000,    // -0.5
            0x3FC0_0000,    // 1.5
            0xC020_0000,    // -2.5
            0x4B7F_FFFF,    // 2^24 - 0.5
        ])
        .map(|&x| [x, x.rotate_left(7), 0, 0])
        .collect();
    fn f(x: u32) -> f32 {
        f32::from_bits(x)
    }
    let ops: &[Op1] = &[
        ("v_cvt_i32_f32", 0, |x| f(x) as i32 as u32),
        ("v_cvt_u32_f32", 0, |x| f(x) as u32),
        ("v_cvt_flr_i32_f32", 0, |x| f(x).floor() as i32 as u32),
        ("v_cvt_rpi_i32_f32", 0, |x| {
            (f(x) + 0.5).floor() as i32 as u32
        }),
        ("v_cvt_f32_i32", 0, |x| (x as i32 as f32).to_bits()),
        ("v_cvt_f32_u32", 0, |x| (x as f32).to_bits()),
        ("v_cvt_f32_ubyte0", 0, |x| ((x & 0xFF) as f32).to_bits()),
        ("v_cvt_f32_ubyte1", 0, |x| {
            ((x >> 8 & 0xFF) as f32).to_bits()
        }),
        ("v_cvt_f32_ubyte2", 0, |x| {
            ((x >> 16 & 0xFF) as f32).to_bits()
        }),
        ("v_cvt_f32_ubyte3", 0, |x| ((x >> 24) as f32).to_bits()),
        ("v_cvt_f16_f32", 0, |x| {
            u32::from(super::bitcode::eval::f32_to_f16(f(x)))
        }),
        ("v_trunc_f32", 0, |x| f(x).trunc().to_bits()),
        ("v_floor_f32", 0, |x| f(x).floor().to_bits()),
        ("v_ceil_f32", 0, |x| f(x).ceil().to_bits()),
        ("v_rndne_f32", 0, |x| f(x).round_ties_even().to_bits()),
    ];
    for &(name, _, reference) in ops {
        let body = vop3(vop3_op(name), 0, vgpr(1), 0, 0);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let want = reference(c[0]);
            let (gf, wf) = (f32::from_bits(got[0]), f32::from_bits(want));
            let float_out =
                !name.contains("_i32_") && !name.contains("_u32_") && name != "v_cvt_f16_f32";
            if float_out && gf.is_nan() && wf.is_nan() {
                continue;
            }
            assert_eq!(got[0], want, "{name}({:#010x} = {:e})", c[0], f(c[0]));
        }
    }
    // v_cvt_pkrtz_f16_f32 v0, v1, v2: both toward zero, v1 in the low half.
    check_v0(
        "v_cvt_pkrtz_f16_f32",
        &vop3(vop3_op("v_cvt_pkrtz_f16_f32"), 0, vgpr(1), vgpr(2), 0),
        &cases,
        |[a, b, ..]| f16_toward_zero(f(a)) | f16_toward_zero(f(b)) << 16,
    );
}

#[test]
fn float_output_modifiers_match_the_isa() {
    let mut cases = Vec::new();
    for &a in &FLOATS {
        for &b in &[0x3F80_0000u32, 0xBF00_0000, 0x3E80_0000, 0x7FC0_0000] {
            cases.push([a, b, 0, 0]);
        }
    }
    for (omod, factor) in [(0u32, 1.0f32), (1, 2.0), (2, 4.0), (3, 0.5)] {
        for clamp in [false, true] {
            // v_add_f32_e64 v0, v1, v2 with omod and clamp.
            let [mut w0, mut w1] = vop3(vop3_op("v_add_f32"), 0, vgpr(1), vgpr(2), 0);
            w0 |= u32::from(clamp) << 15;
            w1 |= omod << 27;
            check_v0(
                &format!("v_add_f32 omod {omod} clamp {clamp}"),
                &[w0, w1],
                &cases,
                |[a, b, ..]| {
                    let mut r = (f32::from_bits(a) + f32::from_bits(b)) * factor;
                    if clamp {
                        r = if r.is_nan() || r <= 0.0 {
                            0.0
                        } else {
                            r.min(1.0)
                        };
                    }
                    if r.is_nan() { 0x7FC0_0000 } else { r.to_bits() }
                },
            );
        }
    }
}

#[test]
fn vector_carries_match_the_isa() {
    let cases = pairs();
    // op v0, s20, v1, v2; v_cndmask_b32 v5, 0, 1, s20.
    for (name, add, rev) in [
        ("v_add_co_u32", true, false),
        ("v_sub_co_u32", false, false),
        ("v_subrev_co_u32", false, true),
    ] {
        let mut body = vop3b(vop3_op(name), 0, 20, vgpr(1), vgpr(2), 0).to_vec();
        body.extend_from_slice(&vop3(0x101, 5, 0x80, 0x81, sgpr(20)));
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let (a, b) = if rev { (c[1], c[0]) } else { (c[0], c[1]) };
            let (r, carry) = if add {
                a.overflowing_add(b)
            } else {
                a.overflowing_sub(b)
            };
            assert_eq!([got[0], got[1]], [r, u32::from(carry)], "{name}{c:08x?}");
        }
    }
    // v_cmp_ne_u32 s22, v3, 0 (carry in: v3 != 0); op v0, s20, v1, v2, s22.
    for (name, add, rev) in [
        ("v_add_co_ci_u32", true, false),
        ("v_sub_co_ci_u32", false, false),
        ("v_subrev_co_ci_u32", false, true),
    ] {
        let mut body = vop3(vop3_op("v_cmp_ne_u32"), 22, vgpr(3), 0x80, 0).to_vec();
        body.extend_from_slice(&vop3b(vop3_op(name), 0, 20, vgpr(1), vgpr(2), sgpr(22)));
        body.extend_from_slice(&vop3(0x101, 5, 0x80, 0x81, sgpr(20)));
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let (a, b) = if rev { (c[1], c[0]) } else { (c[0], c[1]) };
            let cin = u64::from(c[2] != 0);
            let (wide, r) = if add {
                let s = u64::from(a) + u64::from(b) + cin;
                (s > 0xFFFF_FFFF, s as u32)
            } else {
                let borrow = u64::from(b) + cin > u64::from(a);
                (borrow, a.wrapping_sub(b).wrapping_sub(cin as u32))
            };
            assert_eq!([got[0], got[1]], [r, u32::from(wide)], "{name}{c:08x?}");
        }
    }
}

#[test]
fn scalar_64_bit_operations_match_the_isa() {
    // s[0:1] and s[2:3] are the inputs; results in s[6:7], SCC in s12
    // (s[8:11] hold the harness's buffer descriptor).
    let mut cases = Vec::new();
    for &a in &[
        0u64,
        1,
        u64::MAX,
        1 << 63,
        0xFFFF_FFFF,
        1 << 32,
        0x1234_5678_9ABC_DEF0,
    ] {
        for &b in &[0u64, 1, 31, 32, 33, 63, u64::MAX, 0x0F0F_0F0F_F0F0_F0F0] {
            cases.push([a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]);
        }
    }
    let read = |r: [u32; 4]| (u64::from(r[2]) | u64::from(r[3]) << 32, r[1]);
    let tail = [
        sop2(0x0a, 12, 0x81, 0x80), // s_cselect_b32 s12, 1, 0
        vop1(0x001, 5, sgpr(12)),
        vop1(0x001, 6, sgpr(6)),
        vop1(0x001, 7, sgpr(7)),
    ];
    type Op64 = (&'static str, u8, bool, fn(u64, u64) -> u64);
    let ops: &[Op64] = &[
        ("s_and_b64", 0x0f, true, |a, b| a & b),
        ("s_or_b64", 0x11, true, |a, b| a | b),
        ("s_xor_b64", 0x13, true, |a, b| a ^ b),
        ("s_andn2_b64", 0x15, true, |a, b| a & !b),
        ("s_orn2_b64", 0x17, true, |a, b| a | !b),
        ("s_nand_b64", 0x19, true, |a, b| !(a & b)),
        ("s_nor_b64", 0x1b, true, |a, b| !(a | b)),
        ("s_xnor_b64", 0x1d, true, |a, b| !(a ^ b)),
        ("s_lshl_b64", 0x1f, false, |a, b| a << (b & 63)),
        ("s_lshr_b64", 0x21, false, |a, b| a >> (b & 63)),
        ("s_ashr_i64", 0x23, false, |a, b| {
            ((a as i64) >> (b & 63)) as u64
        }),
    ];
    for &(name, op, pair, reference) in ops {
        // The shifts' amount is the 32-bit s2.
        let mut body = vec![sop2(op, 6, 0, 2)];
        body.extend(tail);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let a = u64::from(c[0]) | u64::from(c[1]) << 32;
            let b = if pair {
                u64::from(c[2]) | u64::from(c[3]) << 32
            } else {
                u64::from(c[2])
            };
            let want = reference(a, b);
            assert_eq!(
                read(got),
                (want, u32::from(want != 0)),
                "{name}({a:#x}, {b:#x})"
            );
        }
    }
    // A 64-bit add and subtract through the carry: s_add_u32 s6, s0, s2;
    // s_addc_u32 s7, s1, s3 (and s_sub_u32 / s_subb_u32); SCC: the carry out.
    for (name, lo, hi, add) in [("add", 0x00u8, 0x04u8, true), ("sub", 0x01, 0x05, false)] {
        let mut body = vec![sop2(lo, 6, 0, 2), sop2(hi, 7, 1, 3)];
        body.extend(tail);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let a = u64::from(c[0]) | u64::from(c[1]) << 32;
            let b = u64::from(c[2]) | u64::from(c[3]) << 32;
            let (r, carry) = if add {
                a.overflowing_add(b)
            } else {
                a.overflowing_sub(b)
            };
            assert_eq!(
                read(got),
                (r, u32::from(carry)),
                "64-bit {name}({a:#x}, {b:#x})"
            );
        }
    }
    // s_cmp_eq_u32 s0, s2; s_cselect_b64 s[6:7], s[0:1], s[2:3].
    let mut body = vec![sopc(0x06, 0, 2), sop2(0x0b, 6, 0, 2)];
    body.extend(tail);
    for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
        let a = u64::from(c[0]) | u64::from(c[1]) << 32;
        let b = u64::from(c[2]) | u64::from(c[3]) << 32;
        let want = if c[0] == c[2] { a } else { b };
        assert_eq!(read(got).0, want, "s_cselect_b64({a:#x}, {b:#x})");
    }
    // s_bitcmp0/1_b32 s0, s2 and _b64 s[0:1], s2: SCC is the bit's test.
    for (name, op, wide, set) in [
        ("s_bitcmp0_b32", 0x0cu8, false, false),
        ("s_bitcmp1_b32", 0x0d, false, true),
        ("s_bitcmp0_b64", 0x0e, true, false),
        ("s_bitcmp1_b64", 0x0f, true, true),
    ] {
        let mut body = vec![sopc(op, 0, 2)];
        body.extend(tail);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let a = u64::from(c[0]) | u64::from(c[1]) << 32;
            let bit = if wide { c[2] & 63 } else { c[2] & 31 };
            let want = (a >> bit & 1 != 0) == set;
            assert_eq!(got[1], u32::from(want), "{name}({a:#x}, {})", c[2]);
        }
    }
}

/// VOP2 with SDWA (src0 = 0xF9): `op vdst, v[src0], v[vsrc1]` and the SDWA
/// dword: selects (0-3 byte, 4-5 word, 6 dword), destination padding (0
/// zeros, 1 sign, 2 keep) and source sign extension.
fn vop2_sdwa(
    op: u8,
    vdst: u8,
    src0: u8,
    vsrc1: u8,
    dst: (u8, u8),
    sel0: (u8, bool),
    sel1: (u8, bool),
) -> [u32; 2] {
    [
        vop2(op, vdst, 0xF9, vsrc1),
        u32::from(src0)
            | u32::from(dst.0) << 8
            | u32::from(dst.1) << 11
            | u32::from(sel0.0) << 16
            | u32::from(sel0.1) << 19
            | u32::from(sel1.0) << 24
            | u32::from(sel1.1) << 27,
    ]
}

/// The field `sel` of `x`, zero- or sign-extended.
fn sdwa_field(x: u32, sel: u8, sext: bool) -> u32 {
    let (width, at) = match sel {
        0..=3 => (8, 8 * u32::from(sel)),
        4 | 5 => (16, 16 * u32::from(sel - 4)),
        _ => return x,
    };
    let f = (x >> at) & ((1 << width) - 1);
    if sext {
        (((f << (32 - width)) as i32) >> (32 - width)) as u32
    } else {
        f
    }
}

#[test]
fn sdwa_selects_match_the_isa() {
    let cases: Vec<[u32; 4]> = [0x8081_7F01u32, 0xFFFF_0000, 0x1234_ABCD, 0x0000_8000]
        .iter()
        .flat_map(|&a| [0x8000_00FFu32, 0x7FFF_7FFF, 0x0102_0304].map(|b| [a, b, 0xCAFE_F00D, 0]))
        .collect();
    // v_mov_b32 v0, v3 first, so a kept destination has old bits to keep.
    for dst_sel in [0u8, 1, 3, 4, 5, 6] {
        for unused in [0u8, 1, 2] {
            for (s0, x0) in [(0u8, false), (1, true), (5, true), (4, false), (6, false)] {
                for (s1, x1) in [(2u8, true), (4, false), (6, false)] {
                    let mut body = vec![vop1(0x001, 0, vgpr(3))];
                    // v_add_nc_u32 (VOP2 0x25) v0, v1, v2 with SDWA.
                    body.extend(vop2_sdwa(
                        0x25,
                        0,
                        1,
                        2,
                        (dst_sel, unused),
                        (s0, x0),
                        (s1, x1),
                    ));
                    let what = format!("SDWA dst {dst_sel}/{unused} src {s0}/{x0} {s1}/{x1}");
                    check_v0(&what, &body, &cases, |[a, b, old, _]| {
                        let r = sdwa_field(a, s0, x0).wrapping_add(sdwa_field(b, s1, x1));
                        if dst_sel == 6 {
                            return r;
                        }
                        let (width, at) = if dst_sel < 4 {
                            (8, 8 * u32::from(dst_sel))
                        } else {
                            (16, 16 * u32::from(dst_sel - 4))
                        };
                        let m = ((1u64 << width) - 1) as u32;
                        let field = (r & m) << at;
                        match unused {
                            0 => field,
                            1 => ((((r & m) << (32 - width)) as i32) >> (32 - width - at)) as u32,
                            _ => old & !(m << at) | field,
                        }
                    });
                }
            }
        }
    }
}

#[test]
fn float_arithmetic_matches_the_isa() {
    let mut cases = Vec::new();
    for &a in &FLOATS {
        for &b in &FLOATS {
            cases.push([a, b, 0x3F80_0000, 0]);
        }
    }
    let f = |x: u32| f32::from_bits(x);
    let same = |got: u32, want: f32| got == want.to_bits() || (want.is_nan() && f(got).is_nan());
    type FloatOp = (&'static str, fn(f32, f32) -> f32);
    let ops: &[FloatOp] = &[
        ("v_add_f32", |a, b| a + b),
        ("v_sub_f32", |a, b| a - b),
        ("v_subrev_f32", |a, b| b - a),
        ("v_mul_f32", |a, b| a * b),
        // D3D9's: 0 times anything (infinity and NaN too) is 0.
        ("v_mul_legacy_f32", |a, b| {
            if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
        }),
    ];
    for &(name, reference) in ops {
        let body = vop3(vop3_op(name), 0, vgpr(1), vgpr(2), 0);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let want = reference(f(c[0]), f(c[1]));
            // The sign of a legacy zero product is not pinned down.
            let ok = same(got[0], want)
                || (name == "v_mul_legacy_f32" && want == 0.0 && f(got[0]) == 0.0);
            assert!(
                ok,
                "{name}({:e}, {:e}) = {:e}, want {want:e}",
                f(c[0]),
                f(c[1]),
                f(got[0])
            );
        }
    }
    // max and min: a NaN operand gives the other; RDNA2 orders -0 below +0.
    for (name, max) in [("v_max_f32", true), ("v_min_f32", false)] {
        let body = vop3(vop3_op(name), 0, vgpr(1), vgpr(2), 0);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let (a, b) = (f(c[0]), f(c[1]));
            let want = if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else if a == 0.0 && b == 0.0 {
                // -0 below +0: max is -0 only if both are, min if either is.
                let negative = if max {
                    a.is_sign_negative() && b.is_sign_negative()
                } else {
                    a.is_sign_negative() || b.is_sign_negative()
                };
                if negative { -0.0 } else { 0.0 }
            } else if a == b || (a > b) == max {
                a
            } else {
                b
            };
            assert!(
                same(got[0], want),
                "{name}({a:e}, {b:e}) = {:e}, want {want:e}",
                f(got[0])
            );
        }
    }
}

#[test]
fn packed_16_bit_integers_match_the_isa() {
    let cases = pairs();
    let lanes = |a: u32, b: u32, op: fn(u16, u16) -> u16| {
        let lo = op(a as u16, b as u16);
        let hi = op((a >> 16) as u16, (b >> 16) as u16);
        u32::from(lo) | u32::from(hi) << 16
    };
    type Pk = (&'static str, fn(u16, u16) -> u16);
    let ops: &[Pk] = &[
        ("v_pk_add_u16", u16::wrapping_add),
        ("v_pk_sub_u16", u16::wrapping_sub),
        ("v_pk_mul_lo_u16", u16::wrapping_mul),
        ("v_pk_max_u16", u16::max),
        ("v_pk_min_u16", u16::min),
        ("v_pk_max_i16", |a, b| (a as i16).max(b as i16) as u16),
        ("v_pk_min_i16", |a, b| (a as i16).min(b as i16) as u16),
        ("v_pk_lshlrev_b16", |a, b| b << (a & 15)),
        ("v_pk_lshrrev_b16", |a, b| b >> (a & 15)),
        ("v_pk_ashrrev_i16", |a, b| ((b as i16) >> (a & 15)) as u16),
    ];
    for &(name, op) in ops {
        let code = crate::gfx10::opcode(crate::gfx10::Encoding::Vop3p, name)
            .unwrap_or_else(|| panic!("{name}")) as u8;
        check_v0(
            name,
            &vop3p(code, 0, vgpr(1), vgpr(2), 0),
            &cases,
            |[a, b, ..]| lanes(a, b, op),
        );
    }
}

/// `v_perm_b32`'s byte select (RDNA2 ISA): bytes 0-3 of S1, then 4-7 of S0;
/// 8-11 the sign of bytes 1 and 3 of S1, then of S0, as 0x00 or 0xFF; 12
/// is 0x00, anything above 0xFF.
fn perm_reference(s0: u32, s1: u32, sel: u32) -> u32 {
    let data = u64::from(s0) << 32 | u64::from(s1);
    (0..4)
        .map(|i| {
            let s = (sel >> (8 * i)) & 0xFF;
            let byte = match s {
                0..=7 => (data >> (8 * s)) & 0xFF,
                8..=11 => {
                    let bit = [15, 31, 47, 63][s as usize - 8];
                    if data >> bit & 1 != 0 { 0xFF } else { 0 }
                }
                12 => 0,
                _ => 0xFF,
            };
            (byte as u32) << (8 * i)
        })
        .sum()
}

#[test]
fn v_perm_b32_matches_the_isa() {
    let selectors = [
        0x0706_0504u32,
        0x0302_0100,
        0x0C0C_0C0C,
        0x0B0A_0908,
        0x0D0E_FF10,
        0x0001_0806,
        0x0405_0607,
        0x0C08_0901,
    ];
    let mut cases = Vec::new();
    for &a in &[0x8081_7F01u32, 0x1234_5678, 0xFFFF_0000, 0x0080_8000] {
        for &b in &[0x00FF_80FFu32, 0xDEAD_BEEF, 0x7F7F_7F7F] {
            for &s in &selectors {
                cases.push([a, b, s, 0]);
            }
        }
    }
    // A computed selector (v3), and each selector as a literal.
    check_v0(
        "v_perm_b32 (dynamic)",
        &vop3(vop3_op("v_perm_b32"), 0, vgpr(1), vgpr(2), vgpr(3)),
        &cases,
        |[a, b, s, _]| perm_reference(a, b, s),
    );
    for &s in &selectors {
        let mut body = vop3(vop3_op("v_perm_b32"), 0, vgpr(1), vgpr(2), LITERAL).to_vec();
        body.push(s);
        let mine: Vec<[u32; 4]> = cases.iter().filter(|c| c[2] == s).copied().collect();
        check_v0(
            &format!("v_perm_b32 {s:#010x}"),
            &body,
            &mine,
            |[a, b, ..]| perm_reference(a, b, s),
        );
    }
}

#[test]
fn f16_arithmetic_rounds_once_as_the_isa() {
    // Every product or sum of two halves is exact in f32 and rounds once to
    // f16 (round to nearest even), as the hardware does.
    let halves = [
        0x0000u32, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0xFBFF, 0x7C00,
        0xFC00, 0x7E00, 0x3555, 0x4248, 0x1400,
    ];
    let mut cases = Vec::new();
    for &a in &halves {
        for &b in &halves {
            cases.push([a, b, 0, 0]);
        }
    }
    let to_h = |x: f32| u32::from(super::bitcode::eval::f32_to_f16(x));
    let h = |x: u32| f16_to_f32(x as u16);
    type HalfOp = (&'static str, fn(f32, f32) -> f32);
    let ops: &[HalfOp] = &[
        ("v_add_f16", |a, b| a + b),
        ("v_sub_f16", |a, b| a - b),
        ("v_subrev_f16", |a, b| b - a),
        ("v_mul_f16", |a, b| a * b),
    ];
    for &(name, op) in ops {
        let body = vop3(vop3_op(name), 0, vgpr(1), vgpr(2), 0);
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let want = op(h(c[0]), h(c[1]));
            let got_h = got[0] & 0xFFFF;
            if want.is_nan() {
                assert!(
                    h(got_h).is_nan(),
                    "{name}({:#06x}, {:#06x}) = {got_h:#06x}",
                    c[0],
                    c[1]
                );
            } else {
                assert_eq!(got_h, to_h(want), "{name}({:#06x}, {:#06x})", c[0], c[1]);
            }
            assert_eq!(got[0] >> 16, 0, "{name} keeps the high half");
        }
    }
}

#[test]
fn multiply_add_operand_orders_match_the_isa() {
    // Small integers, so products and sums are exact.
    let k = 7.0f32;
    let cases: Vec<[u32; 4]> = [
        (2.0f32, 3.0f32, 5.0f32),
        (-4.0, 0.5, 1.0),
        (10.0, -3.0, -2.0),
    ]
    .iter()
    .map(|&(a, b, c)| [a.to_bits(), b.to_bits(), c.to_bits(), 0])
    .collect();
    let vop2_op = |name: &str| {
        crate::gfx10::opcode(crate::gfx10::Encoding::Vop2, name).unwrap_or_else(|| panic!("{name}"))
            as u8
    };
    // op v0, v1, v2, K: madmk/fmamk compute v1 * K + v2, madak/fmaak v1 * v2 + K.
    for (name, k_multiplies) in [
        ("v_madmk_f32", true),
        ("v_fmamk_f32", true),
        ("v_madak_f32", false),
        ("v_fmaak_f32", false),
    ] {
        let body = [vop2(vop2_op(name), 0, vgpr(1), 2), k.to_bits()];
        check_v0(name, &body, &cases, |[a, b, ..]| {
            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
            if k_multiplies { a * k + b } else { a * b + k }.to_bits()
        });
    }
    // v_mov_b32 v0, v3; op v0, v1, v2: v0 accumulates v1 * v2.
    for name in ["v_mac_f32", "v_fmac_f32"] {
        let body = [vop1(0x001, 0, vgpr(3)), vop2(vop2_op(name), 0, vgpr(1), 2)];
        check_v0(name, &body, &cases, |[a, b, c, _]| {
            (f32::from_bits(a) * f32::from_bits(b) + f32::from_bits(c)).to_bits()
        });
    }
}

/// `words` (a VOP3, VOP3B or VOP3P instruction) with its clamp bit set.
fn clamped(mut words: [u32; 2]) -> [u32; 2] {
    words[0] |= 1 << 15;
    words
}

#[test]
fn integer_clamp_saturates_as_the_isa() {
    let cases = pairs();
    let ops: &[Op3] = &[
        ("v_add_nc_u32", 0, |a, b, _| a.saturating_add(b)),
        ("v_sub_nc_u32", 0, |a, b, _| a.saturating_sub(b)),
        ("v_subrev_nc_u32", 0, |a, b, _| b.saturating_sub(a)),
        ("v_add_nc_i32", 0, |a, b, _| {
            (a as i32).saturating_add(b as i32) as u32
        }),
        ("v_sub_nc_i32", 0, |a, b, _| {
            (a as i32).saturating_sub(b as i32) as u32
        }),
        ("v_add_nc_u16", 0, |a, b, _| {
            (a as u16).saturating_add(b as u16).into()
        }),
        ("v_sub_nc_u16", 0, |a, b, _| {
            (a as u16).saturating_sub(b as u16).into()
        }),
        ("v_add_nc_i16", 0, |a, b, _| {
            u32::from((a as i16).saturating_add(b as i16) as u16)
        }),
        ("v_sub_nc_i16", 0, |a, b, _| {
            u32::from((a as i16).saturating_sub(b as i16) as u16)
        }),
    ];
    for &(name, _, reference) in ops {
        let body = clamped(vop3(vop3_op(name), 0, vgpr(1), vgpr(2), 0));
        // 16-bit results land in the low half (the high half, zero, kept).
        check_v0(&format!("{name} clamp"), &body, &cases, |[a, b, c, _]| {
            reference(a, b, c)
        });
    }
    // The carry forms: saturated where the carry (borrow) leaves, the
    // carry itself as without clamp.
    for (name, add, rev) in [
        ("v_add_co_u32", true, false),
        ("v_sub_co_u32", false, false),
        ("v_subrev_co_u32", false, true),
    ] {
        let mut body = clamped(vop3b(vop3_op(name), 0, 20, vgpr(1), vgpr(2), 0)).to_vec();
        body.extend_from_slice(&vop3(0x101, 5, 0x80, 0x81, sgpr(20)));
        for (c, got) in cases.iter().zip(eval_cases(&body, &cases)) {
            let (a, b) = if rev { (c[1], c[0]) } else { (c[0], c[1]) };
            let (r, carry) = if add {
                (a.saturating_add(b), a.checked_add(b).is_none())
            } else {
                (a.saturating_sub(b), a < b)
            };
            assert_eq!(
                [got[0], got[1]],
                [r, u32::from(carry)],
                "{name} clamp{c:08x?}"
            );
        }
    }
}

#[test]
fn dot_product_clamp_saturates_the_exact_sum() {
    let cases = pairs();
    let field = |x: u32, width: u32, i: u32, signed: bool| -> i64 {
        let f = (x >> (width * i)) & ((1u64 << width) - 1) as u32;
        if signed {
            i64::from(sext(width, f))
        } else {
            i64::from(f)
        }
    };
    for (name, op, width, signed) in [
        ("v_dot2_i32_i16", 0x14u8, 16u32, true),
        ("v_dot2_u32_u16", 0x15, 16, false),
        ("v_dot4_i32_i8", 0x16, 8, true),
        ("v_dot4_u32_u8", 0x17, 8, false),
        ("v_dot8_u32_u4", 0x19, 4, false),
    ] {
        let body = clamped(vop3p(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        check_v0(&format!("{name} clamp"), &body, &cases, |[a, b, c, _]| {
            let acc = if signed {
                i64::from(c as i32)
            } else {
                i64::from(c)
            };
            let sum = acc
                + (0..32 / width)
                    .map(|i| field(a, width, i, signed) * field(b, width, i, signed))
                    .sum::<i64>();
            if signed {
                sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32 as u32
            } else {
                sum.clamp(0, i64::from(u32::MAX)) as u32
            }
        });
    }
}

#[test]
fn half_dot_products_match_the_isa() {
    // Values whose products and sums are exact in f32, then infinities and
    // NaNs.
    let h = |x: f32| u32::from(super::bitcode::eval::f32_to_f16(x));
    let pack = |lo: f32, hi: f32| h(lo) | h(hi) << 16;
    let cases: Vec<[u32; 4]> = [
        ((1.0, 2.0), (3.0, 4.0), 0.5),
        ((-1.5, 0.25), (2.0, -8.0), -7.0),
        ((1024.0, 0.5), (1024.0, 3.0), 1.0),
        ((0.0, -0.0), (5.0, 5.0), -0.0),
        ((f32::INFINITY, 1.0), (1.0, 1.0), 2.0),
        ((f32::INFINITY, 1.0), (0.0, 1.0), 2.0),
        ((f32::NAN, 1.0), (1.0, 1.0), 2.0),
        ((1.0, 1.0), (1.0, 1.0), f32::NAN),
    ]
    .iter()
    .map(|&((a0, a1), (b0, b1), c)| [pack(a0, a1), pack(b0, b1), c.to_bits(), 0])
    .collect();
    let want = |[a, b, c, _]: [u32; 4]| {
        let f = |x: u32| f64::from(f16_to_f32(x as u16));
        let exact = f(a) * f(b) + f(a >> 16) * f(b >> 16) + f64::from(f32::from_bits(c));
        exact as f32
    };
    let check = |name: &str, body: &[u32]| {
        for (c, got) in cases.iter().zip(eval_cases(body, &cases)) {
            let (got, want) = (f32::from_bits(got[0]), want(*c));
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{name}{c:08x?} = {got}, want {want}"
            );
        }
    };
    check("v_dot2_f32_f16", &vop3p(0x13, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_mov_b32 v0, v3; v_dot2c_f32_f16 v0, v1, v2.
    let vop2_op = crate::gfx10::opcode(crate::gfx10::Encoding::Vop2, "v_dot2c_f32_f16")
        .expect("v_dot2c_f32_f16") as u8;
    check(
        "v_dot2c_f32_f16",
        &[vop1(0x001, 0, vgpr(3)), vop2(vop2_op, 0, vgpr(1), 2)],
    );
}

// --- Waves: cross-lane instructions, lanes in lockstep --------------------

/// Distinct per-lane inputs.
fn lane_inputs(seed: u32) -> Vec<u32> {
    (0..64u32)
        .map(|i| i.wrapping_mul(0x9E37_79B9) ^ seed)
        .collect()
}

/// `body` run as one wave (32 or 64 lanes): each lane's `v4`.
fn run_wave(body: &[u32], wave64: bool, a: &[u32], b: &[u32], k: [u32; 4]) -> Vec<u32> {
    let lanes = if wave64 { 64 } else { 32 };
    eval_wave(body, lanes, wave64, a, b, k).unwrap_or_else(|e| panic!("{body:08x?}: {e}"))
}

#[test]
fn a_wave_runs_each_lane_on_its_own_data() {
    let (a, b) = (lane_inputs(1), lane_inputs(2));
    for wave64 in [false, true] {
        // v_add_nc_u32 v4, v1, v2.
        let got = run_wave(&[vop2(0x25, 4, vgpr(1), 2)], wave64, &a, &b, [0; 4]);
        for (l, g) in got.iter().enumerate() {
            assert_eq!(*g, a[l].wrapping_add(b[l]), "wave64 {wave64}, lane {l}");
        }
    }
}

#[test]
fn lane_reads_and_writes_match_the_isa() {
    let (a, b) = (lane_inputs(3), lane_inputs(4));
    for wave64 in [false, true] {
        let n: u32 = if wave64 { 64 } else { 32 };
        for k in [0u32, 5, n - 1] {
            // v_readlane_b32 s20, v1, k; v_mov_b32 v4, s20.
            let mut body = vop3(vop3_op("v_readlane_b32"), 20, vgpr(1), 128 + k as u16, 0).to_vec();
            body.push(vop1(0x001, 4, sgpr(20)));
            let got = run_wave(&body, wave64, &a, &b, [0; 4]);
            assert!(
                got.iter().all(|&g| g == a[k as usize]),
                "readlane {k}, wave64 {wave64}: {got:x?}"
            );
            // v_mov_b32 v4, v1; v_writelane_b32 v4, s0, k: lane k takes s0.
            let mut body = vec![vop1(0x001, 4, vgpr(1))];
            body.extend(vop3(
                vop3_op("v_writelane_b32"),
                4,
                sgpr(0),
                128 + k as u16,
                0,
            ));
            let got = run_wave(&body, wave64, &a, &b, [0xABCD_0123, 0, 0, 0]);
            for (l, g) in got.iter().enumerate() {
                let want = if l as u32 == k { 0xABCD_0123 } else { a[l] };
                assert_eq!(*g, want, "writelane {k}, wave64 {wave64}, lane {l}");
            }
        }
        // v_readfirstlane_b32 s20, v1; v_mov_b32 v4, s20.
        let got = run_wave(
            &[vop1(0x002, 20, vgpr(1)), vop1(0x001, 4, sgpr(20))],
            wave64,
            &a,
            &b,
            [0; 4],
        );
        assert!(
            got.iter().all(|&g| g == a[0]),
            "readfirstlane, wave64 {wave64}"
        );
    }
}

#[test]
fn mbcnt_counts_the_mask_below_each_lane() {
    let (a, b) = (lane_inputs(5), lane_inputs(6));
    for wave64 in [false, true] {
        let n = if wave64 { 64 } else { 32 };
        // v_mbcnt_lo_u32_b32 v4, -1, 0 (and _hi with -1 in wave64): the lane.
        let mut body = vop3(vop3_op("v_mbcnt_lo_u32_b32"), 4, 193, 128, 0).to_vec();
        if wave64 {
            body.extend(vop3(vop3_op("v_mbcnt_hi_u32_b32"), 4, 193, vgpr(4), 0));
        }
        let got = run_wave(&body, wave64, &a, &b, [0; 4]);
        assert_eq!(
            got,
            (0..n).collect::<Vec<u32>>(),
            "lane index, wave64 {wave64}"
        );
        // v_cmp_gt_u32 s[20:21], v1, v2; mbcnt over that mask: the lanes
        // below with a > b.
        let mut body = vop3(vop3_op("v_cmp_gt_u32"), 20, vgpr(1), vgpr(2), 0).to_vec();
        body.extend(vop3(vop3_op("v_mbcnt_lo_u32_b32"), 4, sgpr(20), 128, 0));
        if wave64 {
            body.extend(vop3(vop3_op("v_mbcnt_hi_u32_b32"), 4, sgpr(21), vgpr(4), 0));
        }
        let got = run_wave(&body, wave64, &a, &b, [0; 4]);
        for (l, g) in got.iter().enumerate() {
            let want = (0..l).filter(|&i| a[i] > b[i]).count() as u32;
            assert_eq!(*g, want, "mbcnt of a compare, wave64 {wave64}, lane {l}");
        }
    }
}

/// The lane DPP16 control `ctrl` reads for lane `l`, if it lies in the
/// row (RDNA2 ISA).
fn dpp16_source(ctrl: u16, l: usize) -> Option<usize> {
    let (row, pos) = (l & !15, l & 15);
    let n = usize::from(ctrl & 15);
    match ctrl {
        0..=0xFF => Some((l & !3) | (usize::from(ctrl) >> (2 * (l & 3)) & 3)),
        0x101..=0x10F => (pos + n < 16).then_some(l + n),
        0x111..=0x11F => (pos >= n).then(|| l - n),
        0x121..=0x12F => Some(row | ((pos + 16 - n) & 15)),
        0x140 => Some(row | (15 - pos)),
        0x141 => Some(row | (pos & 8) | (7 - (pos & 7))),
        0x150..=0x15F => Some(row | n),
        0x160..=0x16F => Some(row | (pos ^ n)),
        _ => unreachable!("{ctrl:#x}"),
    }
}

#[test]
fn dpp_reads_the_isa_lanes() {
    let (a, b) = (lane_inputs(7), lane_inputs(8));
    let mut ctrls: Vec<u16> = vec![0x00, 0xE4, 0x1B, 0x4E, 0xB1, 0xFF, 0x39];
    ctrls.extend((1..16).flat_map(|n| [0x100 + n, 0x110 + n, 0x120 + n]));
    ctrls.extend([
        0x140, 0x141, 0x150, 0x153, 0x15F, 0x160, 0x161, 0x16A, 0x16F,
    ]);
    for wave64 in [false, true] {
        for &ctrl in &ctrls {
            for (row_mask, bank_mask, bound) in
                [(0xF, 0xF, false), (0xF, 0xF, true), (0x5, 0xA, false)]
            {
                // v_mov_b32 v4, v2; v_mov_b32_dpp v4, v1 ctrl.
                let ext = 1
                    | u32::from(ctrl) << 8
                    | u32::from(bound) << 19
                    | (bank_mask as u32) << 24
                    | (row_mask as u32) << 28;
                let body = [vop1(0x001, 4, vgpr(2)), vop1(0x001, 4, 0xFA), ext];
                let got = run_wave(&body, wave64, &a, &b, [0; 4]);
                for (l, g) in got.iter().enumerate() {
                    let enabled =
                        row_mask >> (l / 16) & 1 != 0 && bank_mask >> (l / 4 % 4) & 1 != 0;
                    let want = match (enabled, dpp16_source(ctrl, l)) {
                        (false, _) => b[l],
                        (true, Some(s)) => a[s],
                        (true, None) if bound => 0,
                        (true, None) => b[l],
                    };
                    assert_eq!(
                        *g, want,
                        "dpp {ctrl:#x} rows {row_mask:#x} banks {bank_mask:#x} bound {bound}, wave64 {wave64}, lane {l}"
                    );
                }
            }
        }
        // DPP8: lane l reads lane sel[l % 8] of its group of eight.
        for sel in [0o76543210u32, 0o01234567, 0o00000000, 0o35172604] {
            let body = [vop1(0x001, 4, 0xE9), 1 | sel << 8];
            let got = run_wave(&body, wave64, &a, &b, [0; 4]);
            for (l, g) in got.iter().enumerate() {
                let s = (l & !7) | (sel >> (3 * (l % 8)) & 7) as usize;
                assert_eq!(*g, a[s], "dpp8 {sel:o}, wave64 {wave64}, lane {l}");
            }
        }
    }
}

#[test]
fn permlane16_gathers_within_rows_as_the_isa() {
    let (a, b) = (lane_inputs(9), lane_inputs(10));
    let selectors = [
        (0x7654_3210u32, 0xFEDC_BA98u32),
        (0x0123_4567, 0x89AB_CDEF),
        (0, 0),
        (0xF0F0_F0F0, 0x1357_9BDF),
    ];
    for wave64 in [false, true] {
        for cross in [false, true] {
            let name = if cross {
                "v_permlanex16_b32"
            } else {
                "v_permlane16_b32"
            };
            for (lo, hi) in selectors {
                // op v4, v1, s0, s1 (the selectors as root constants).
                let body = vop3(vop3_op(name), 4, vgpr(1), sgpr(0), sgpr(1));
                let got = run_wave(&body, wave64, &a, &b, [lo, hi, 0, 0]);
                for (l, g) in got.iter().enumerate() {
                    let pos = l & 15;
                    let sel = ((u64::from(hi) << 32 | u64::from(lo)) >> (4 * pos) & 15) as usize;
                    let row = if cross { (l & !15) ^ 16 } else { l & !15 };
                    assert_eq!(
                        *g,
                        a[row | sel],
                        "{name} {lo:#x}:{hi:#x}, wave64 {wave64}, lane {l}"
                    );
                }
            }
        }
    }
}

/// DS `op vdst, v[addr], v[data0] offset:offset` (offset0 and offset1 as
/// one 16-bit offset).
fn ds(op: u8, vdst: u8, addr: u8, data0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | u32::from(op) << 18 | u32::from(offset),
        u32::from(addr) | u32::from(data0) << 8 | u32::from(vdst) << 24,
    ]
}

#[test]
fn ds_swizzle_reads_the_isa_lanes() {
    let (a, b) = (lane_inputs(11), lane_inputs(12));
    let quads = [0x80E4u16, 0x801B, 0x8000, 0x80FF, 0x8039];
    // and, or, xor masks (5 bits each) packed as bits 0-4, 5-9, 10-14.
    let masks = [
        (0x1F, 0, 1),
        (0x1F, 0, 0x10),
        (0x10, 0x03, 0),
        (0x00, 0x1F, 0),
        (0x1C, 0, 3),
        (0x15, 0x0A, 0x11),
    ];
    for wave64 in [false, true] {
        for q in quads {
            // ds_swizzle_b32 v4, v1 offset:q.
            let got = run_wave(&ds(0x35, 4, 1, 0, q), wave64, &a, &b, [0; 4]);
            for (l, g) in got.iter().enumerate() {
                let src = (l & !3) | (usize::from(q) >> (2 * (l & 3)) & 3);
                assert_eq!(*g, a[src], "swizzle {q:#06x}, wave64 {wave64}, lane {l}");
            }
        }
        for (and, or, xor) in masks {
            let offset = and | or << 5 | xor << 10;
            let got = run_wave(&ds(0x35, 4, 1, 0, offset), wave64, &a, &b, [0; 4]);
            for (l, g) in got.iter().enumerate() {
                let j = (((l & 0x1F) & usize::from(and)) | usize::from(or)) ^ usize::from(xor);
                let src = j | (l & 0x20);
                assert_eq!(
                    *g, a[src],
                    "swizzle {offset:#06x}, wave64 {wave64}, lane {l}"
                );
            }
        }
    }
    // Rotate and FFT modes are refused, not taken for quad permutes.
    for offset in [0xC000u16, 0xC421, 0xE000, 0xFFFF] {
        let e = eval_wave(&ds(0x35, 4, 1, 0, offset), 32, false, &a, &b, [0; 4]).unwrap_err();
        assert!(e.contains("rotate or FFT"), "{offset:#06x}: {e}");
    }
}

#[test]
fn ds_bpermute_reads_the_addressed_lanes() {
    let a = lane_inputs(13);
    for wave64 in [false, true] {
        let n = if wave64 { 64 } else { 32 };
        for offset in [0u16, 4, 0x100] {
            // Byte addresses in v2: a lane pattern (reversed, rotated and
            // repeated), past the wave too (they wrap).
            let addresses: Vec<u32> = (0..64u32)
                .map(|l| match l % 3 {
                    0 => 4 * (n - 1 - l % n),
                    1 => 4 * ((l + 5) % n) + 2,
                    _ => 4 * (l + 70),
                })
                .collect();
            // ds_bpermute_b32 v4, v2, v1 offset:offset.
            let got = run_wave(&ds(0xB3, 4, 2, 1, offset), wave64, &a, &addresses, [0; 4]);
            for (l, g) in got.iter().enumerate() {
                let src = ((addresses[l] + u32::from(offset)) / 4 % n) as usize;
                assert_eq!(
                    *g, a[src],
                    "bpermute offset {offset}, wave64 {wave64}, lane {l}"
                );
            }
        }
    }
}

#[test]
fn packed_half_floats_round_each_half_as_the_isa() {
    let halves = [
        0x0000u16, 0x8000, 0x3C00, 0xBC00, 0x3555, 0x4248, 0x7BFF, 0x0001, 0x7C00, 0xFC00, 0x7E00,
        0x1400,
    ];
    let mut cases = Vec::new();
    for (i, &a) in halves.iter().enumerate() {
        for (j, &b) in halves.iter().enumerate() {
            let c = halves[(i + 2 * j) % halves.len()];
            let pack = |lo: u16, hi: u16| u32::from(lo) | u32::from(hi) << 16;
            cases.push([pack(a, b), pack(b, c), pack(c, a), 0]);
        }
    }
    let h = |x: u32| f16_to_f32(x as u16);
    let to_h = |x: f32| u32::from(super::bitcode::eval::f32_to_f16(x));
    type PkF = (&'static str, fn(f32, f32, f32) -> f32);
    let ops: &[PkF] = &[
        ("v_pk_add_f16", |a, b, _| a + b),
        ("v_pk_mul_f16", |a, b, _| a * b),
        // Products of halves are exact in f32: one rounding to f16.
        ("v_pk_fma_f16", |a, b, c| {
            (f64::from(a) * f64::from(b) + f64::from(c)) as f32
        }),
    ];
    // op_sel_hi as encoded by default (each source's high half for the high
    // result), then with op_sel swapping which half each result reads.
    for &(name, op) in ops {
        let code = crate::gfx10::opcode(crate::gfx10::Encoding::Vop3p, name)
            .unwrap_or_else(|| panic!("{name}")) as u8;
        for (opsel, opsel_hi) in [(0u32, 7u32), (7, 0), (1, 7), (5, 2)] {
            let [mut w0, mut w1] = vop3p(code, 0, vgpr(1), vgpr(2), vgpr(3));
            // op_sel bits 11-13; op_sel_hi bit 14 (source 2) and word1 bits 27-28.
            w0 = (w0 & !(0xF << 11)) | opsel << 11 | (opsel_hi >> 2 & 1) << 14;
            w1 = (w1 & !(3 << 27)) | (opsel_hi & 3) << 27;
            let got = eval_cases(&[w0, w1], &cases);
            for (c, g) in cases.iter().zip(got) {
                let half = |k: usize, high: bool| if high { c[k] >> 16 } else { c[k] & 0xFFFF };
                let result = |sel: u32| {
                    let x: Vec<f32> = (0..3).map(|k| h(half(k, sel >> k & 1 != 0))).collect();
                    op(x[0], x[1], x[2])
                };
                for (part, sel) in [(0, opsel), (1, opsel_hi)] {
                    let want = result(sel);
                    let got = (g[0] >> (16 * part)) & 0xFFFF;
                    if want.is_nan() {
                        assert!(
                            h(got).is_nan(),
                            "{name} op_sel {opsel:03b}/{opsel_hi:03b}{c:08x?}"
                        );
                    } else {
                        assert_eq!(
                            got,
                            to_h(want),
                            "{name} op_sel {opsel:03b}/{opsel_hi:03b}{c:08x?} half {part}"
                        );
                    }
                }
            }
        }
    }
}

/// DS `op vdst, v[addr], v[data0], v[data1] offset0 offset1`.
fn ds2(op: u8, vdst: u8, addr: u8, data0: u8, data1: u8, offset0: u8, offset1: u8) -> [u32; 2] {
    [
        0x36 << 26 | u32::from(op) << 18 | u32::from(offset1) << 8 | u32::from(offset0),
        u32::from(addr) | u32::from(data0) << 8 | u32::from(data1) << 16 | u32::from(vdst) << 24,
    ]
}

/// `body` as one wave with 2 KiB of group-shared memory: each lane's `v4`.
fn run_lds(body: &[u32], wave64: bool, a: &[u32], b: &[u32], k: [u32; 4]) -> Vec<u32> {
    let lanes = if wave64 { 64 } else { 32 };
    let native = super::synth::wave_program_lds(body, lanes, wave64, 4);
    super::synth::eval_native(&native, lanes, a, b, k)
        .unwrap_or_else(|e| panic!("{body:08x?}: {e}"))
}

const S_BARRIER: u32 = 0xBF8A_0000;

#[test]
fn group_shared_memory_moves_data_between_lanes() {
    let (a, b) = (lane_inputs(14), lane_inputs(15));
    for wave64 in [false, true] {
        let n = if wave64 { 64 } else { 32 };
        // ds_write_b32 v9, v1 offset:16; s_barrier; v10 = ((lane + 1) % n) * 4;
        // ds_read_b32 v4, v10 offset:16: the next lane's a.
        let mut body = ds(0x0D, 0, 9, 1, 16).to_vec();
        body.push(S_BARRIER);
        body.push(vop2(0x25, 10, 0x84, 9)); // v_add_nc_u32 v10, 4, v9
        body.extend([vop2(0x1B, 10, LITERAL, 10), 4 * n - 1]); // v_and_b32
        body.extend(ds(0x36, 4, 10, 0, 16));
        let got = run_lds(&body, wave64, &a, &b, [0; 4]);
        for (l, g) in got.iter().enumerate() {
            assert_eq!(
                *g,
                a[(l + 1) % n as usize],
                "neighbour, wave64 {wave64}, lane {l}"
            );
        }
        // write2: a at the lane's dword, b 64 dwords on; read2 back with the
        // offsets swapped (v4 the second): b. Then the st64 forms (offsets
        // in 64-dword units).
        for (write, read, far) in [(0x0Eu8, 0x37u8, 64u8), (0x0F, 0x38, 1)] {
            let mut body = ds2(write, 0, 9, 1, 2, 0, far).to_vec();
            body.push(S_BARRIER);
            body.extend(ds2(read, 4, 9, 0, 0, far, 0));
            let got = run_lds(&body, wave64, &a, &b, [0; 4]);
            assert_eq!(
                got,
                b[..n as usize].to_vec(),
                "read2/write2 {write:#x}, wave64 {wave64}"
            );
        }
    }
}

#[test]
fn group_shared_atomics_combine_every_lane() {
    let (a, b) = (lane_inputs(16), lane_inputs(17));
    type Lds = (&'static str, u8, u32, fn(u32, u32) -> u32);
    let ops: &[Lds] = &[
        ("ds_add_u32", 0x00, 0x1234, u32::wrapping_add),
        ("ds_max_u32", 0x08, 0, u32::max),
        ("ds_min_i32", 0x05, 0x7FFF_FFFF, |x, y| {
            (x as i32).min(y as i32) as u32
        }),
        ("ds_and_b32", 0x09, u32::MAX, |x, y| x & y),
        ("ds_or_b32", 0x0A, 0, |x, y| x | y),
        ("ds_xor_b32", 0x0B, 0x5555_AAAA, |x, y| x ^ y),
    ];
    for wave64 in [false, true] {
        let n = if wave64 { 64 } else { 32 };
        for &(name, op, init, combine) in ops {
            // v11 = 0, v12 = s0 (the initial value); every lane writes it to
            // dword 0, then applies the atomic with its a; then reads dword 0.
            let mut body = vec![vop1(0x001, 11, 128), vop1(0x001, 12, sgpr(0))];
            body.extend(ds(0x0D, 0, 11, 12, 0));
            body.push(S_BARRIER);
            body.extend(ds(op, 0, 11, 1, 0));
            body.push(S_BARRIER);
            body.extend(ds(0x36, 4, 11, 0, 0));
            let got = run_lds(&body, wave64, &a, &b, [init, 0, 0, 0]);
            let want = a[..n].iter().fold(init, |acc, &x| combine(acc, x));
            assert!(
                got.iter().all(|&g| g == want),
                "{name}, wave64 {wave64}: {got:x?}, want {want:#x}"
            );
        }
    }
}

/// Lowers `body` with the loop limit at `limit` (on this thread) and runs
/// it per case, as [`eval_cases`] does.
fn eval_cases_limited(body: &[u32], limit: Option<u32>, cases: &[[u32; 4]]) -> Vec<[u32; 4]> {
    super::TEST_LOOP_LIMIT.set(limit);
    let lowered = lower_valu(body);
    super::TEST_LOOP_LIMIT.set(None);
    let l = lowered.unwrap_or_else(|e| panic!("{body:08x?} does not lower: {e}"));
    cases
        .iter()
        .map(|&inputs| {
            eval_valu(&l, inputs)
                .unwrap_or_else(|e| panic!("{body:08x?} on {inputs:08x?} does not run: {e}"))
        })
        .collect()
}

/// The result store did not happen: the program left before it (a lane
/// stopped by the loop limit leaves the program; in compute nothing more
/// is stored).
const NOT_STORED: u32 = 0xDEAD_BEEF;

const S_CBRANCH_SCC1: u8 = 0x05;
const S_SUB_U32: u8 = 0x01;
const S_CMP_LG_U32: u8 = 0x07;
const V_ADD_NC_U32: u8 = 0x25;

/// `v0 = 0; do { v0 += 1; s12 -= 1 } while (s12 != 0)` with `s12 = s0`:
/// s0 iterations, and 2^32 for s0 = 0 (the count wraps), as a count read
/// from memory that holds zeros makes the console run.
fn counted_loop() -> Vec<u32> {
    vec![
        s_mov_b32(12, 0),              // s12 = s0
        vop1(0x001, 0, 128),           // v0 = 0
        vop2(V_ADD_NC_U32, 0, 129, 0), // L: v0 += 1
        sop2(S_SUB_U32, 12, 12, 129),  // s12 -= 1
        sopc(S_CMP_LG_U32, 12, 128),   // scc = s12 != 0
        sopp(S_CBRANCH_SCC1, -4),      // to L
    ]
}

/// Loops run their exact count; the limit lets a loop of `limit`
/// iterations finish and stops the next one, whose lane leaves the
/// program. (The ISA's loop is a back edge taken while SCC holds; the
/// limit counts back edges per entry into the loop.)
#[test]
fn loops_run_their_count_and_stop_at_the_limit() {
    let body = counted_loop();
    for (n, want) in [(1, 1), (2, 2), (7, 7), (8, 8)] {
        let got = eval_cases_limited(&body, Some(8), &[[n, 0, 0, 0]]);
        assert_eq!(got[0][0], want, "{n} iterations under a limit of 8");
    }
    for n in [9, 100, 0] {
        let got = eval_cases_limited(&body, Some(8), &[[n, 0, 0, 0]]);
        assert_eq!(got[0], [NOT_STORED; 4], "{n} iterations under a limit of 8");
    }
    // Without the test limit, the default one: counts far below it run.
    let got = eval_cases_limited(&body, None, &[[1000, 0, 0, 0]]);
    assert_eq!(got[0][0], 1000);
}

/// The limit counts per entry into a loop: an inner loop entered many
/// times runs its count each time, however many iterations it adds up to.
#[test]
fn the_loop_limit_counts_per_entry() {
    // s13 = s1; v0 = 0; O: s12 = s0; I: v0 += 1; s12 -= 1; while s12;
    // s13 -= 1; while s13: v0 = s0 * s1.
    let body = [
        s_mov_b32(13, 1),
        vop1(0x001, 0, 128),
        s_mov_b32(12, 0),              // O:
        vop2(V_ADD_NC_U32, 0, 129, 0), // I:
        sop2(S_SUB_U32, 12, 12, 129),
        sopc(S_CMP_LG_U32, 12, 128),
        sopp(S_CBRANCH_SCC1, -4), // to I
        sop2(S_SUB_U32, 13, 13, 129),
        sopc(S_CMP_LG_U32, 13, 128),
        sopp(S_CBRANCH_SCC1, -8), // to O
    ];
    for (inner, outer) in [(6, 4), (8, 8), (1, 8), (3, 1)] {
        let got = eval_cases_limited(&body, Some(8), &[[inner, outer, 0, 0]]);
        assert_eq!(got[0][0], inner * outer, "{inner} x {outer}");
    }
    for (inner, outer) in [(9, 2), (2, 9)] {
        let got = eval_cases_limited(&body, Some(8), &[[inner, outer, 0, 0]]);
        assert_eq!(got[0], [NOT_STORED; 4], "{inner} x {outer}");
    }
}

/// The case the limit is for, at its real value: a count of zero read
/// from memory that is not mirrored yet would run 2^32 times; the
/// translation leaves the loop after 2^20 and the lane leaves the program.
#[test]
fn a_wrapped_count_stops_at_the_default_limit() {
    assert_eq!(super::loop_limit(), 1 << 20);
    let got = eval_cases_limited(&counted_loop(), None, &[[0, 0, 0, 0]]);
    assert_eq!(got[0], [NOT_STORED; 4]);
}

const S_CBRANCH_SCC0: u8 = 0x04;
const S_CMP_EQ_U32: u8 = 0x06;

/// The counted loop with its back edge on the branch's other side
/// (`s_cbranch_scc0` on `s12 == 0`), and one that a branch can skip, so
/// that its exit is a join with a block of its own: both run their count
/// and are stopped at the limit the same way.
#[test]
fn loops_of_other_shapes_stop_at_the_limit_too() {
    let inverted = [
        s_mov_b32(12, 0),
        vop1(0x001, 0, 128),
        vop2(V_ADD_NC_U32, 0, 129, 0), // L:
        sop2(S_SUB_U32, 12, 12, 129),
        sopc(S_CMP_EQ_U32, 12, 128),
        sopp(S_CBRANCH_SCC0, -4), // to L while s12 != 0
    ];
    // v0 = 0; if (s0 != 0) do { v0 += 1 } while (--s12); v0 += 64.
    let skippable = [
        s_mov_b32(12, 0),
        vop1(0x001, 0, 128),
        sopc(S_CMP_EQ_U32, 12, 128),
        sopp(S_CBRANCH_SCC1, 4),       // to J
        vop2(V_ADD_NC_U32, 0, 129, 0), // L:
        sop2(S_SUB_U32, 12, 12, 129),
        sopc(S_CMP_LG_U32, 12, 128),
        sopp(S_CBRANCH_SCC1, -4),      // to L
        vop2(V_ADD_NC_U32, 0, 192, 0), // J: v0 += 64
    ];
    for (n, want) in [(1, 1), (5, 5), (8, 8)] {
        let got = eval_cases_limited(&inverted, Some(8), &[[n, 0, 0, 0]]);
        assert_eq!(got[0][0], want, "inverted, {n} iterations");
        let got = eval_cases_limited(&skippable, Some(8), &[[n, 0, 0, 0]]);
        assert_eq!(got[0][0], want + 64, "skippable, {n} iterations");
    }
    let got = eval_cases_limited(&skippable, Some(8), &[[0, 0, 0, 0]]);
    assert_eq!(got[0][0], 64, "skipped");
    for n in [9, 0] {
        let got = eval_cases_limited(&inverted, Some(8), &[[n, 0, 0, 0]]);
        assert_eq!(got[0], [NOT_STORED; 4], "inverted, {n} iterations");
    }
    let got = eval_cases_limited(&skippable, Some(8), &[[9, 0, 0, 0]]);
    assert_eq!(got[0], [NOT_STORED; 4], "skippable, 9 iterations");
}

/// The MIMG opcode of `name`.
fn mimg_op(name: &str) -> u16 {
    crate::gfx10::opcode(crate::gfx10::Encoding::Mimg, name).unwrap_or_else(|| panic!("{name}"))
}

/// A texture's dwords as bytes.
fn texel_bytes(texels: &[u32]) -> Vec<u8> {
    texels.iter().flat_map(|t| t.to_le_bytes()).collect()
}

/// An image atomic's reference: the texel's new value from its old one,
/// the data and (cmpswap) the compare value.
type ImageAtomic = (&'static str, fn(u64, u64, u64, u32) -> u64);

/// The image atomics the lowering takes, by the ISA's definitions
/// (IMAGE_ATOMIC_*: the texel becomes `op(old, data)`, cmpswap stores its
/// first data dword where the texel equals its second); `bits` is the
/// width (32, or 64 for the two-dword forms).
const IMAGE_ATOMICS: &[ImageAtomic] = &[
    ("image_atomic_swap", |_, x, _, _| x),
    (
        "image_atomic_cmpswap",
        |m, x, c, _| if m == c { x } else { m },
    ),
    ("image_atomic_add", |m, x, _, _| m.wrapping_add(x)),
    ("image_atomic_sub", |m, x, _, _| m.wrapping_sub(x)),
    ("image_atomic_smin", |m, x, _, bits| {
        let s = |v: u64| ((v << (64 - bits)) as i64) >> (64 - bits);
        if s(m) <= s(x) { m } else { x }
    }),
    ("image_atomic_umin", |m, x, _, _| m.min(x)),
    ("image_atomic_smax", |m, x, _, bits| {
        let s = |v: u64| ((v << (64 - bits)) as i64) >> (64 - bits);
        if s(m) >= s(x) { m } else { x }
    }),
    ("image_atomic_umax", |m, x, _, _| m.max(x)),
    ("image_atomic_and", |m, x, _, _| m & x),
    ("image_atomic_or", |m, x, _, _| m | x),
    ("image_atomic_xor", |m, x, _, _| m ^ x),
];

/// 32-bit image atomics on a 4x4 R32 texture UAV: the addressed texel
/// becomes the operation's result, the others stay, and with glc the old
/// value comes back in the data VGPR. (`v[1:2]` = (x, y) and `v3` (`v[3:4]`
/// for cmpswap: data, compare) come from the root constants.)
#[test]
fn image_atomics_on_a_typed_texture() {
    let texels: Vec<u32> = (0..16u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    for &(name, reference) in IMAGE_ATOMICS {
        let cmpswap = name.ends_with("cmpswap");
        for glc in [false, true] {
            let mut body =
                super::synth::mimg(mimg_op(name), 3, 1, 12, 1 + 2 * u8::from(cmpswap), 1, glc)
                    .to_vec();
            body.push(vop1(0x001, 0, vgpr(3))); // v0 = v3
            let l = super::synth::lower_native(&super::synth::image_program(&body))
                .unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            for (x, y, data) in [(2u32, 3u32, 5u32), (0, 0, 0x8000_0001), (3, 1, 0xFFFF_FFFF)] {
                let i = (y * 4 + x) as usize;
                // A compare that matches, and one that does not.
                for cmp in [texels[i], texels[i] ^ 1] {
                    let (out, after) = super::synth::eval_image(
                        &l,
                        [x, y, data, cmp],
                        [4, 4, 1],
                        texel_bytes(&texels),
                    )
                    .unwrap_or_else(|e| panic!("{name}: does not run: {e}"));
                    let old = texels[i];
                    let want =
                        reference(u64::from(old), u64::from(data), u64::from(cmp), 32) as u32;
                    let mut texels_after = texels.clone();
                    texels_after[i] = want;
                    assert_eq!(
                        after,
                        texel_bytes(&texels_after),
                        "{name} at ({x}, {y}) with {data:#x}, compare {cmp:#x}"
                    );
                    let back = if glc { old } else { data };
                    assert_eq!(out[0], back, "{name}, glc {glc}: returned");
                }
            }
        }
    }
}

/// 64-bit image atomics (dmask 0x3; cmpswap 0xF) on a 4x4 R64 texture UAV,
/// translated only with the test switch: the data VGPRs hold the value's
/// low then high dword (cmpswap: new, then compare), and with glc the old
/// value comes back in the first two. Root constants: the value (or new)
/// in s[0:1], cmpswap's compare in s[2:3]; the coordinates are constants.
#[test]
fn wide_image_atomics_on_a_typed_texture() {
    let texels: Vec<u64> = (0..16u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let bytes = |t: &[u64]| -> Vec<u8> { t.iter().flat_map(|t| t.to_le_bytes()).collect() };
    for &(name, reference) in IMAGE_ATOMICS {
        let cmpswap = name.ends_with("cmpswap");
        let dmask = if cmpswap { 0xF } else { 0x3 };
        for (x, y) in [(2u32, 3u32), (0, 0), (3, 1)] {
            let i = (y * 4 + x) as usize;
            // v[5:6] = (x, y); the atomic on v[1:2] (cmpswap v[1:4]); the
            // returned value from v[1:2] into v0 and v5.
            let mut body = vec![
                vop1(0x001, 5, 128 + x as u16),
                vop1(0x001, 6, 128 + y as u16),
            ];
            body.extend(super::synth::mimg(mimg_op(name), 1, 5, 12, dmask, 1, true));
            body.push(vop1(0x001, 0, vgpr(1)));
            body.push(vop1(0x001, 5, vgpr(2)));
            super::TEST_IMAGE_ATOMICS_64.set(true);
            let lowered = super::synth::lower_native(&super::synth::image_program(&body));
            super::TEST_IMAGE_ATOMICS_64.set(false);
            let l = lowered.unwrap_or_else(|e| panic!("{name}: does not lower: {e}"));
            for value in [5u64, 0x8000_0000_0000_0001, u64::MAX, 0x1_0000_0000] {
                for cmp in [texels[i], texels[i] ^ 1 << 40] {
                    let k = [
                        value as u32,
                        (value >> 32) as u32,
                        cmp as u32,
                        (cmp >> 32) as u32,
                    ];
                    let (out, after) = super::synth::eval_image(&l, k, [4, 4, 1], bytes(&texels))
                        .unwrap_or_else(|e| panic!("{name}: does not run: {e}"));
                    let old = texels[i];
                    let mut want = texels.clone();
                    want[i] = reference(old, value, cmp, 64);
                    assert_eq!(
                        after,
                        bytes(&want),
                        "{name} at ({x}, {y}) with {value:#x}, compare {cmp:#x}"
                    );
                    let back = u64::from(out[1]) << 32 | u64::from(out[0]);
                    assert_eq!(back, old, "{name}: returned");
                }
            }
        }
    }
}

/// A 64-bit image atomic declares what it needs (DXC's rules): 64-bit
/// integer operations, 64-bit atomics on a typed resource and on a
/// descriptor-heap resource, in the shader flags and SFI0. Without the
/// test switch it is refused, as before.
#[test]
fn wide_image_atomics_declare_their_features() {
    let mut body =
        super::synth::mimg(mimg_op("image_atomic_umax"), 1, 5, 12, 0x3, 1, true).to_vec();
    body.push(vop1(0x001, 0, vgpr(1)));
    let native = super::synth::image_program(&body);
    let refused = super::synth::lower_native(&native).map(|_| ()).unwrap_err();
    assert!(
        refused.to_string().contains("a 64-bit image atomic"),
        "{refused}"
    );
    super::TEST_IMAGE_ATOMICS_64.set(true);
    let lowered = super::synth::lower_native(&native);
    super::TEST_IMAGE_ATOMICS_64.set(false);
    let l = lowered.expect("lowers with the switch");
    let container = crate::dxbc::Container::parse(&l.dxil).expect("a container");
    let sfi0 = container
        .part(crate::dxbc::FourCC::SFI0)
        .expect("SFI0")
        .data;
    let features = u64::from_le_bytes(sfi0.try_into().expect("8 bytes"));
    for (bit, what) in [
        (0x8000, "Int64Ops"),
        (0x40_0000, "AtomicInt64OnTypedResource"),
        (0x1000_0000, "AtomicInt64OnHeapResource"),
    ] {
        assert_ne!(features & bit, 0, "SFI0 {features:#x}: no {what}");
    }
    // The GPU layer refuses such a pipeline on a device without them.
    let shader = |l: &super::lower::Lowered| super::HostShader {
        stage: crate::HwStage::Cs,
        dxil: l.dxil.clone(),
        user_data: l.user_data.clone(),
        samplers: l.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs.clone(),
        root_srvs: l.root_srvs.clone(),
    };
    assert!(shader(&l).needs_typed_atomics_64());
    // A 32-bit atomic needs none of them.
    let mut narrow =
        super::synth::mimg(mimg_op("image_atomic_umax"), 1, 5, 12, 0x1, 1, true).to_vec();
    narrow.push(vop1(0x001, 0, vgpr(1)));
    let l = super::synth::lower_native(&super::synth::image_program(&narrow)).expect("lowers");
    let container = crate::dxbc::Container::parse(&l.dxil).expect("a container");
    let sfi0 = container
        .part(crate::dxbc::FourCC::SFI0)
        .expect("SFI0")
        .data;
    let features = u64::from_le_bytes(sfi0.try_into().expect("8 bytes"));
    assert_eq!(features & 0x1040_8000, 0, "SFI0 {features:#x}");
    assert!(!shader(&l).needs_typed_atomics_64());
}
