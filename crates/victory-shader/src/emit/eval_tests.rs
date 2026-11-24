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
    LITERAL, eval_cases, lower_valu, sgpr, sop1, sop2, sopc, sopk, vgpr, vop1, vop2, vop3, vop3b,
    vop3p,
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
