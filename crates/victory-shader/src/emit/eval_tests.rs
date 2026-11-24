//! Lowered VALU and SALU code run by the module evaluator
//! ([`super::bitcode::eval`]) against references: on any machine, without
//! a GPU.
//!
//! The first tests replay cases `valu_tests` checks on a host GPU and WARP
//! (expected values the hardware gave), so the evaluator's choices are
//! known to agree with real drivers on them. The rest check instructions
//! against references written from AMD's RDNA2 ISA descriptions.

use super::bitcode::eval::f16_to_f32;
use super::synth::{LITERAL, eval_cases, lower_valu, sgpr, sop1, sop2, vgpr, vop1, vop2, vop3};

/// A named instruction with its opcode and a reference, in test tables.
type Op1 = (&'static str, u16, fn(u32) -> u32);
type Op3 = (&'static str, u16, fn(u32, u32, u32) -> u32);
type Sop1 = (&'static str, u8, fn(u32) -> u32);
type Sop2 = (&'static str, u8, fn(u32, u32) -> u32);

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
