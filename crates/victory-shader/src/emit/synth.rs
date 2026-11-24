//! Synthetic GFX10 compute programs for tests: instruction encoders and a
//! one-thread program around a body of VALU or SALU code, lowered, and run
//! by [`super::bitcode::eval`] (any machine) or on a host GPU
//! (`valu_tests`, Windows).

use super::bitcode::eval::{self, Env};
use super::lower::{self, Lowered};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

pub(super) const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `op sdst, src0`.
pub(super) const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// SOP1 `s_mov_b32 sdst, src0`.
pub(super) const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    sop1(0x03, sdst, src0)
}

/// SOP2 `op sdst, src0, src1`.
pub(super) const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
pub(super) const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
pub(super) const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1` (`vsrc1` a VGPR number).
pub(super) const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
pub(super) const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// VOP3B `op vdst, sdst, src0, src1, src2` (no modifiers): the VOP3 form
/// with a scalar destination (a carry or a lane mask) in bits 8-14.
pub(super) const fn vop3b(
    op: u16,
    vdst: u8,
    sdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, src2);
    [w0 | (sdst as u32 & 0x7F) << 8, w1]
}

/// VOP3P `op vdst, src0, src1, src2` with the default op_sel_hi (each
/// source's high half for the high result) and no other modifiers.
pub(super) const fn vop3p(op: u8, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32 & 0x7F) << 16 | 1 << 14 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18 | 3 << 27,
    ]
}

/// SOPK `op sdst, simm16`.
pub(super) const fn sopk(op: u8, sdst: u8, imm: u16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | imm as u32
}

/// SOPC `op src0, src1`.
pub(super) const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
pub(super) const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

pub(super) const fn sgpr(n: u8) -> u16 {
    n as u16
}

pub(super) const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// The source code of a literal constant (the dword after the instruction).
pub(super) const LITERAL: u16 = 0xFF;

/// A one-thread compute program that receives 4 dwords of root constants
/// in `s0..s3` (copied to `v1..v4`), runs `body`, and stores `v0`, `v5`,
/// `v6` and `v7` (the results: a 64-bit one in `v[6:7]`) to dwords 0-3 of
/// the raw UAV at heap index 0.
pub(super) fn valu_program(body: &[u32]) -> NativeProgram {
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
    for (k, v) in [0u8, 5, 6, 7].into_iter().enumerate() {
        words.extend_from_slice(&buffer_store_dword(v, 8, 4 * k as u16));
    }
    words.push(S_ENDPGM);

    NativeProgram {
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
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Lowers [`valu_program`] around `body`.
pub(super) fn lower_valu(body: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let native = valu_program(body);
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
}

/// The user-data dwords of a [`valu_program`] given its root constants.
pub(super) fn user_data(l: &Lowered, inputs: [u32; 4]) -> Vec<u32> {
    let n = l.user_data.iter().map(|u| u.index + 1).max().unwrap_or(0);
    let mut ud = vec![0u32; n.next_multiple_of(4) as usize];
    for u in &l.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => inputs[u.dword as usize],
            (RootType::Table, 1) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    ud
}

/// Runs a lowered [`valu_program`] on the evaluator: `v0`, `v5`, `v6` and
/// `v7` at the end, or why it could not run.
pub(super) fn eval_valu(l: &Lowered, inputs: [u32; 4]) -> Result<[u32; 4], String> {
    let mut env = Env {
        user_data: user_data(l, inputs),
        ..Env::default()
    };
    // A marker the program overwrites: a store that never happens shows.
    env.heap.insert(
        0,
        [0xDEAD_BEEFu32; 4]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    );
    eval::run(&l.module, &mut env)?;
    if env.out_of_bounds != 0 {
        return Err(format!("{} accesses out of bounds", env.out_of_bounds));
    }
    let out = &env.heap[&0];
    let dword = |k: usize| u32::from_le_bytes(out[4 * k..4 * k + 4].try_into().expect("4 bytes"));
    Ok([dword(0), dword(1), dword(2), dword(3)])
}

/// Lowers `body` and runs it on the evaluator for each input set: the
/// results per case ([`eval_valu`]). Panics with the reason when either
/// step fails.
pub(super) fn eval_cases(body: &[u32], cases: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let l = lower_valu(body).unwrap_or_else(|e| panic!("{body:08x?} does not lower: {e}"));
    cases
        .iter()
        .map(|&inputs| {
            eval_valu(&l, inputs)
                .unwrap_or_else(|e| panic!("{body:08x?} on {inputs:08x?} does not run: {e}"))
        })
        .collect()
}
