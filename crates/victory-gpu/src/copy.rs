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

/// The furthest offset the pass may reach: its dword reads and writes go
/// up to 3 bytes past a copy's last byte, in 32-bit arithmetic.
const OFFSET_MAX: u64 = u32::MAX as u64 - 3;

/// The dispatches (as their root constants and group counts) that copy
/// `r`, or `None` if the pass cannot: offsets past 32 bits, more than
/// 65,535 rows or slices, or rows of a multi-row copy that do not start on
/// dwords (two rows could share one). Single rows longer than a dispatch
/// covers are cut at dwords of the destination.
pub(crate) fn dispatches(r: &Region) -> Option<Vec<([u32; 8], [u32; 3])>> {
    if r.bytes == 0 || r.rows == 0 || r.slices == 0 {
        return Some(Vec::new());
    }
    if r.rows > 65_535 || r.slices > 65_535 {
        return None;
    }
    // Past the last byte copied, each side (pitches are the title's).
    let end = |at: u64, pitch: u64, slice: u64| {
        let rows = pitch.checked_mul(u64::from(r.rows - 1))?;
        let slices = slice.checked_mul(u64::from(r.slices - 1))?;
        at.checked_add(rows)?
            .checked_add(slices)?
            .checked_add(r.bytes)
    };
    let last_from = end(r.from, r.from_pitch, r.from_slice)?;
    let last_to = end(r.to, r.to_pitch, r.to_slice)?;
    if last_from > OFFSET_MAX || last_to > OFFSET_MAX {
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
/// back into title memory once the submission has run). Overlapping ranges
/// of one host copy go through a buffer of ours, so that the copy reads
/// what was there before it, a memory move as the copy on the CPU makes
/// it, not what other threads of the pass already wrote. False, with
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
    // Both ranges lie in their buffers (found), so neither end overflows.
    let overlap = std::sync::Arc::ptr_eq(&src, &dst) && from < to + len && to < from + len;
    let staged = if overlap {
        let Some(staging) = staging(len) else {
            return false;
        };
        Some(staging)
    } else {
        None
    };
    src.refresh(list, from, len);
    dst.refresh(list, to, len);
    dst.gpu_writes(list, to, len);
    let Some(staging) = staged else {
        return record(list, src.base(), dst.base(), &r);
    };
    crate::host::name(&staging, || format!("overlapping copy {len:#x}"));
    list.keep(&staging);
    // SAFETY: a live buffer; written as unordered access from the common
    // state (a buffer's promotion), then read so.
    let between = unsafe { staging.GetGPUVirtualAddress() };
    record(list, src.base(), between, &Region::bytes(r.from, 0, len))
        && record(list, between, dst.base(), &Region::bytes(0, r.to, len))
}

/// Where a texture-to-buffer copy writes in title memory: the first byte,
/// the destination footprint's row and slice pitches, and the end of the
/// title buffer (nothing is written past it).
pub(crate) struct Destination {
    pub(crate) at: u64,
    pub(crate) row_pitch: u64,
    pub(crate) slice_pitch: u64,
    pub(crate) end: u64,
}

/// Records the copy of `subresource` of the host `texture` (within `area`,
/// a box as left, top, front, right, bottom, back; all of it if `None`) into
/// title memory laid out as `dst` says. The GPU copies the texture into a
/// staging buffer of ours at a host footprint (pitches the host accepts,
/// block rows for block-compressed formats), then our pass moves its rows
/// to the title buffer's host copy at the title's pitches; that range is
/// brought up to date first and declared GPU-written, so the copied bytes
/// reach title memory after the submission. The texture must already be in
/// the copy source state. Err, with nothing recorded, says what it cannot
/// copy.
pub(crate) fn texture_to_buffer(
    list: &HostList,
    texture: &ID3D12Resource,
    subresource: u32,
    area: Option<[u32; 6]>,
    dst: &Destination,
) -> Result<(), &'static str> {
    let device = &host().ok_or("no host device")?.device;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let three_d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let mips = u32::from(desc.MipLevels.max(1));
    let layers = if three_d {
        1
    } else {
        u32::from(desc.DepthOrArraySize.max(1))
    };
    let mip = subresource % mips;
    let plane = subresource / (mips * layers);
    let size = |n: u64| u32::try_from((n >> mip).max(1)).unwrap_or(u32::MAX);
    let whole = [
        0,
        0,
        0,
        size(desc.Width),
        size(u64::from(desc.Height)),
        if three_d {
            size(u64::from(desc.DepthOrArraySize))
        } else {
            1
        },
    ];
    let [left, top, front, right, bottom, back] = area.unwrap_or(whole);
    if right <= left || bottom <= top || back <= front {
        return Ok(());
    }
    // The box as a texture of its own: its footprint is the staging layout.
    let part = D3D12_RESOURCE_DESC {
        Dimension: desc.Dimension,
        Alignment: 0,
        Width: u64::from(right - left),
        Height: bottom - top,
        DepthOrArraySize: if three_d {
            u16::try_from(back - front).map_err(|_| "a box too deep")?
        } else {
            1
        },
        MipLevels: 1,
        Format: desc.Format,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
    // SAFETY: outputs for one subresource.
    unsafe {
        device.GetCopyableFootprints(
            &part,
            plane,
            1,
            0,
            Some(&mut footprint),
            Some(&mut rows),
            Some(&mut row_bytes),
            Some(&mut total),
        );
    }
    if rows == 0 || row_bytes == 0 || total == 0 || total == u64::MAX {
        return Err("the host has no footprint for the copied box");
    }
    let slices = footprint.Footprint.Depth.max(1);
    let pitch = u64::from(footprint.Footprint.RowPitch);
    // The title's pitches: checked, as everything it passes.
    let last = u64::from(slices - 1)
        .checked_mul(dst.slice_pitch)
        .zip(u64::from(rows - 1).checked_mul(dst.row_pitch))
        .and_then(|(s, r)| {
            dst.at
                .checked_add(s)?
                .checked_add(r)?
                .checked_add(row_bytes)
        });
    let Some(last) = last.filter(|&last| last <= dst.end) else {
        return Err("the copied box runs past the destination buffer");
    };
    let len = last - dst.at;
    let shadow = crate::shadow::find(dst.at, len).ok_or("the destination has no host copy")?;
    let r = Region {
        from: 0,
        to: shadow.offset(dst.at),
        bytes: row_bytes,
        rows,
        slices,
        from_pitch: pitch,
        to_pitch: dst.row_pitch,
        from_slice: pitch * u64::from(rows),
        to_slice: dst.slice_pitch,
    };
    if copier().is_none() {
        return Err("no GPU copy pass");
    }
    if dispatches(&r).is_none() {
        return Err("destination rows that do not start on dwords, or too many of them");
    }
    let staging = staging(total).ok_or("no staging buffer")?;
    crate::host::name(&staging, || format!("texture readback {total:#x}"));
    list.keep(&staging);
    shadow.refresh(list, dst.at, len);
    shadow.gpu_writes(list, dst.at, len);
    let from = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed for the call; the caller's list keeps the texture.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    };
    let to = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed for the call; the list keeps the buffer.
        pResource: unsafe { std::mem::transmute_copy(&staging) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    let area = D3D12_BOX {
        left,
        top,
        front,
        right,
        bottom,
        back,
    };
    // SAFETY: an open list; a live texture in the copy source state and our
    // buffer, promoted from the common state by the copy, then moved on.
    unsafe {
        list.list
            .CopyTextureRegion(&to, 0, 0, 0, &from, Some(&area));
        list.list.ResourceBarrier(&[crate::host::transition(
            &staging,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    // SAFETY: a live buffer.
    let from = unsafe { staging.GetGPUVirtualAddress() };
    if record(list, from, shadow.base(), &r) {
        Ok(())
    } else {
        Err("no GPU copy pass")
    }
}

/// A buffer of `bytes` in GPU memory our passes can write, in the common
/// state (as every buffer is created), rounded up to whole dwords: the
/// pass reads the dword holding a copy's last byte.
fn staging(bytes: u64) -> Option<ID3D12Resource> {
    let host = host()?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(4).next_multiple_of(4),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    match made {
        Ok(()) => buffer,
        Err(e) => {
            tracing::warn!(error = %e, bytes, "no staging buffer");
            None
        }
    }
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

    /// What titles pass is checked before any arithmetic on it: pitches
    /// whose products overflow and counts past a dispatch are refused, not
    /// wrapped (or a panic in an exported method); the pass's dword reads
    /// and writes past a copy's last byte stay within 32-bit offsets.
    #[test]
    fn title_sized_regions_are_refused_before_they_overflow() {
        let r = Region {
            rows: 65_535,
            from_pitch: u64::MAX / 2,
            ..Region::bytes(0, 0, 4)
        };
        assert!(dispatches(&r).is_none(), "the rows' span overflows");
        let r = Region {
            slices: 3,
            to_slice: u64::MAX,
            ..Region::bytes(0, 0, 4)
        };
        assert!(dispatches(&r).is_none(), "the slices' span overflows");
        let r = Region {
            rows: u32::MAX,
            ..Region::bytes(0, 0, 4)
        };
        assert!(dispatches(&r).is_none(), "more rows than a dispatch has");
        assert!(dispatches(&Region::bytes(u64::MAX - 1, 0, 4)).is_none());
        let last = u64::from(u32::MAX) - 3;
        assert!(dispatches(&Region::bytes(0, last - 4, 4)).is_some());
        assert!(dispatches(&Region::bytes(0, last - 3, 4)).is_none());
        assert!(dispatches(&Region::bytes(last - 3, 0, 4)).is_none());
    }
}
