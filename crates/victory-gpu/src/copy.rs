//! Copies between host copies of title memory run by the GPU, in order with
//! the list's other GPU work (invariant C5, docs/architecture.md).
//!
//! Buffer copies run on the CPU when their submission starts, in step order
//! with the list's other CPU work: a fill of upload memory (a constant
//! buffer of a fresh table) after a copy reads what the copy wrote, as the
//! title expects. That order is wrong where shaders recorded earlier in the
//! same list write the copy's source: the CPU takes the bytes from before
//! they ran. Those copies are recorded here instead, as a compute pass of
//! ours (HLSL compiled by the system's compiler, as the indirect clamp is)
//! between the two host copies, through unordered access as translated
//! shaders use them, between global UAV barriers. The destination range is
//! declared GPU-written, so after the submission what changed goes back
//! into title memory (`shadow::Shadow::copy_back`).
//!
//! The pass copies rows between pitches too (texture-to-buffer copies).
//! Each thread writes one destination dword, reading and merging the bytes
//! of it outside the copied range, so any byte alignment is copied exactly;
//! rows of a multi-row copy must start on dwords so that no two rows share
//! one.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{PCSTR, s};

use crate::host::{HostList, host};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), UAV(u0), UAV(u1)"
cbuffer Copy : register(b0) {
    uint from;       // source byte offset of the first row
    uint to;         // destination byte offset of the first row
    uint bytes;      // bytes per row
    uint from_pitch; // source bytes from a row to the next
    uint to_pitch;   // destination bytes from a row to the next
    uint from_slice; // source bytes from a slice to the next
    uint to_slice;   // destination bytes from a slice to the next
    uint unused;
};
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint s0 = from + id.y * from_pitch + id.z * from_slice;
    uint d0 = to + id.y * to_pitch + id.z * to_slice;
    // One destination dword per thread: the bytes of it the row covers.
    uint at = (d0 & ~3u) + id.x * 4;
    uint end = d0 + bytes;
    if (at >= end) return;
    uint lo = max(at, d0);
    uint hi = min(at + 4, end);
    uint value;
    if (lo == at && hi == at + 4 && ((s0 ^ d0) & 3u) == 0) {
        value = src.Load(s0 + (at - d0));
    } else {
        value = dst.Load(at);
        for (uint b = lo; b < hi; b++) {
            uint s = s0 + (b - d0);
            uint byte = (src.Load(s & ~3u) >> ((s & 3u) * 8)) & 0xFFu;
            uint shift = (b - at) * 8;
            value = (value & ~(0xFFu << shift)) | (byte << shift);
        }
    }
    dst.Store(at, value);
}
"#;

/// A compute pipeline of ours and its root signature (inside the shader).
pub(crate) struct Compute {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Compute {}
// SAFETY: as above.
unsafe impl Sync for Compute {}

/// Compiles `source` (a cs_5_1 `main` with its root signature) into a
/// compute pipeline on the host device; `label` names it in compiler errors.
pub(crate) fn compute(source: &str, label: PCSTR) -> windows::core::Result<Compute> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            label,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("compute pass: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Compute { root, pso })
    }
}

fn copier() -> Option<&'static Compute> {
    static COPIER: std::sync::OnceLock<Option<Compute>> = std::sync::OnceLock::new();
    COPIER
        .get_or_init(|| match compute(SOURCE, s!("copy")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU copy pass: copies run on the CPU");
                None
            }
        })
        .as_ref()
}

/// What a copy moves, as byte offsets in the source and destination
/// buffers: `rows` rows of `bytes` in each of `slices` slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) bytes: u64,
    pub(crate) rows: u32,
    pub(crate) slices: u32,
    pub(crate) from_pitch: u64,
    pub(crate) to_pitch: u64,
    pub(crate) from_slice: u64,
    pub(crate) to_slice: u64,
}

impl Region {
    /// `bytes` bytes from `from` to `to`.
    pub(crate) fn bytes(from: u64, to: u64, bytes: u64) -> Region {
        Region {
            from,
            to,
            bytes,
            rows: 1,
            slices: 1,
            from_pitch: 0,
            to_pitch: 0,
            from_slice: 0,
            to_slice: 0,
        }
    }
}

/// The most destination bytes one dispatch row covers: 65,535 groups of 64
/// threads of one dword each.
const ROW_MAX: u64 = 65_535 * 64 * 4;

/// The dispatches (as their root constants and group counts) that copy
/// `r`, or `None` if the pass cannot: offsets past 32 bits, more than
/// 65,535 rows or slices, or rows of a multi-row copy that do not start on
/// dwords (two rows could share one). Single rows longer than a dispatch
/// covers are cut at dwords of the destination.
pub(crate) fn dispatches(r: &Region) -> Option<Vec<([u32; 8], [u32; 3])>> {
    let last_from = r.from
        + r.from_pitch * u64::from(r.rows.max(1) - 1)
        + r.from_slice * u64::from(r.slices.max(1) - 1)
        + r.bytes;
    let last_to = r.to
        + r.to_pitch * u64::from(r.rows.max(1) - 1)
        + r.to_slice * u64::from(r.slices.max(1) - 1)
        + r.bytes;
    if r.bytes == 0 || r.rows == 0 || r.slices == 0 {
        return Some(Vec::new());
    }
    if last_from > u64::from(u32::MAX) || last_to > u64::from(u32::MAX) {
        return None;
    }
    if r.rows > 65_535 || r.slices > 65_535 {
        return None;
    }
    let several = r.rows > 1 || r.slices > 1;
    if several
        && (!r.to.is_multiple_of(4)
            || !r.to_pitch.is_multiple_of(4)
            || !r.to_slice.is_multiple_of(4)
            || (r.rows > 1 && r.to_pitch < r.bytes)
            || (r.slices > 1 && r.to_slice < r.to_pitch * u64::from(r.rows - 1) + r.bytes))
    {
        return None;
    }
    let constants = |from: u64, to: u64, bytes: u64| {
        [
            from as u32,
            to as u32,
            bytes as u32,
            r.from_pitch as u32,
            r.to_pitch as u32,
            r.from_slice as u32,
            r.to_slice as u32,
            0,
        ]
    };
    let groups = |to: u64, bytes: u64| ((to % 4 + bytes).div_ceil(4)).div_ceil(64) as u32;
    if several {
        if r.bytes > ROW_MAX {
            return None;
        }
        return Some(vec![(
            constants(r.from, r.to, r.bytes),
            [groups(r.to, r.bytes), r.rows, r.slices],
        )]);
    }
    let end = r.to + r.bytes;
    let mut out = Vec::new();
    let mut at = r.to;
    while at < end {
        let next = ((at & !3) + ROW_MAX).min(end);
        let from = r.from + (at - r.to);
        out.push((
            constants(from, at, next - at),
            [groups(at, next - at), 1, 1],
        ));
        at = next;
    }
    Some(out)
}

/// Records, in `list`, the copy of `r` from the buffer at host address
/// `src` to the one at `dst` (each resource's start), between global UAV
/// barriers. Replaces the list's compute root signature and pipeline:
/// callers make the title's draws and dispatches bind theirs again. False,
/// with nothing recorded, if the pass is not available or cannot copy `r`.
pub(crate) fn record(list: &HostList, src: u64, dst: u64, r: &Region) -> bool {
    let (Some(c), Some(dispatches)) = (copier(), dispatches(r)) else {
        return false;
    };
    let l = &list.list;
    // SAFETY: an open list, our root signature and pipeline, and buffers
    // the caller's list keeps alive (host copies it refreshes or holds).
    unsafe {
        // Earlier shader writes first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootUnorderedAccessView(1, src);
        l.SetComputeRootUnorderedAccessView(2, dst);
        for (constants, [x, y, z]) in dispatches {
            l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            l.Dispatch(x, y, z);
        }
        // Later work reads what the pass wrote.
        l.ResourceBarrier(&[uav_barrier()]);
    }
    true
}

/// Records the copy of `len` bytes of title memory from `from` to `to` as
/// a GPU copy between their host copies: both brought up to date when the
/// list is submitted, the destination declared written by the GPU (copied
/// back into title memory once the submission has run). False, with
/// nothing recorded, if either range has no host copy or the pass cannot
/// copy it; the caller copies on the CPU then.
pub(crate) fn buffers(list: &HostList, to: u64, from: u64, len: u64) -> bool {
    let (Some(src), Some(dst)) = (crate::shadow::find(from, len), crate::shadow::find(to, len))
    else {
        return false;
    };
    let r = Region::bytes(src.offset(from), dst.offset(to), len);
    if copier().is_none() || dispatches(&r).is_none() {
        return false;
    }
    src.refresh(list, from, len);
    dst.refresh(list, to, len);
    dst.gpu_writes(list, to, len);
    record(list, src.base(), dst.base(), &r)
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single row is cut at destination dwords, every 16,776,960 bytes,
    /// each piece's source following it; any alignment.
    #[test]
    fn long_copies_are_cut_at_destination_dwords() {
        let d = dispatches(&Region::bytes(3, 6, 40_000_000)).unwrap();
        assert_eq!(d.len(), 3);
        let (c0, g0) = d[0];
        assert_eq!((c0[0], c0[1]), (3, 6));
        assert_eq!(
            c0[2] as u64,
            ROW_MAX - 4 + 2,
            "to the first dword boundary past ROW_MAX"
        );
        assert_eq!(g0, [65_535, 1, 1]);
        let (c1, _) = d[1];
        assert_eq!(c1[1] % 4, 0);
        assert_eq!(c1[1] - c0[1], c0[2]);
        assert_eq!(c1[0] - c0[0], c0[2]);
        let total: u64 = d.iter().map(|(c, _)| u64::from(c[2])).sum();
        assert_eq!(total, 40_000_000);
        assert!(dispatches(&Region::bytes(0, 0, 0)).unwrap().is_empty());
        assert_eq!(dispatches(&Region::bytes(1, 2, 3)).unwrap()[0].1, [1, 1, 1]);
    }

    /// Rows and slices go to one dispatch, rows as y and slices as z; rows
    /// that could share a destination dword, offsets past 32 bits and too
    /// many rows are refused.
    #[test]
    fn rows_need_dword_starts_and_fit_one_dispatch() {
        let r = Region {
            from: 0,
            to: 512,
            bytes: 64,
            rows: 4,
            slices: 2,
            from_pitch: 64,
            to_pitch: 256,
            from_slice: 256,
            to_slice: 1024,
        };
        let d = dispatches(&r).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].1, [1, 4, 2]);
        assert_eq!(d[0].0, [0, 512, 64, 64, 256, 256, 1024, 0]);
        assert!(
            dispatches(&Region { to: 514, ..r }).is_none(),
            "rows off dwords"
        );
        assert!(dispatches(&Region { to_pitch: 254, ..r }).is_none());
        assert!(
            dispatches(&Region { to_pitch: 32, ..r }).is_none(),
            "rows overlap"
        );
        assert!(
            dispatches(&Region { to_slice: 512, ..r }).is_none(),
            "slices overlap"
        );
        assert!(dispatches(&Region { rows: 70_000, ..r }).is_none());
        assert!(dispatches(&Region::bytes(0, 1 << 32, 4)).is_none());
    }
}
