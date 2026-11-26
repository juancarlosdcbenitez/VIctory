//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). Grand Theft Auto
        // VI keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 65),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
#[cfg(windows)]
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }

    /// Where the element holding texel `(x, y, z)` starts in the buffer, and
    /// the footprint's row and slice pitches (a slice is the footprint's
    /// rows of elements, blocks for block-compressed formats). `None` for a
    /// format whose element size is not known.
    pub(crate) fn texel(&self, x: u32, y: u32, z: u32) -> Option<(u64, u64, u64)> {
        let (element, block) = element(self.format)?;
        let row_pitch = u64::from(self.row_pitch);
        let slice_pitch = row_pitch * u64::from(self.height).div_ceil(block);
        let at = self.offset
            + u64::from(z) * slice_pitch
            + u64::from(y) / block * row_pitch
            + u64::from(x) / block * element;
        Some((at, row_pitch, slice_pitch))
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's size in an upload buffer: footprint (offset 0), rows
/// and bytes per row. Planes of depth-stencil formats are not told apart.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let (mip, slice) = (index % mips, index / mips);
    if slice >= slices {
        return None;
    }
    let (element, block) = element(desc.format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format: desc.format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(PITCH)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Device method: where each subresource goes in an upload buffer, rows
/// 256-byte and subresources 512-byte aligned, as on PC. Unknown sizes give
/// the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (Grand Theft Auto VI places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet.
    #[cfg(windows)]
    host: Option<std::sync::Arc<HostTexture>>,
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
}

/// The host GPU's copy of a title resource, if it has one.
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host.as_deref().map(|h| &h.texture)
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host.clone()
}

/// A host texture like the title's: same description, in GPU memory,
/// without the console's layout and flags.
#[cfg(windows)]
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    use windows::Win32::Graphics::Direct3D12::*;
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; Grand Theft Auto VI's shadow maps ask for it).
    // shortcut: the host copy loses UAV use; give it a second view-only
    // copy if draws write it as a UAV.
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    if d.flags & DEPTH != 0 {
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = pc_format(value[0]);
        value
    });
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description has the host layout (checked size, same
    // fields); the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            (&raw const d).cast(),
            D3D12_RESOURCE_STATES(
                crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32,
            ),
            // The same layout: format, then color or depth and stencil.
            clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast()),
            &mut texture,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no host texture");
    }
    if let Some(t) = &texture {
        crate::host::name(t, || {
            format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
        });
    }
    texture
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear);
    #[cfg(windows)]
    let host = host_texture(&desc, state, clear);
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: host.map(|texture| {
            let state = crate::host::legal_state(&texture, crate::host::texture_state(state));
            std::sync::Arc::new(HostTexture {
                texture,
                state: state.into(),
            })
        }),
    });
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if self
            .host
            .as_ref()
            .is_some_and(|h| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        crate::texture::forget(std::ptr::from_ref::<Resource>(self) as usize);
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from Grand Theft Auto VI's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From Grand Theft Auto VI's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, Grand Theft Auto VI's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// shortcut: never released (one host texture per distinct address and
/// description viewed); release them when the title unmaps or reuses the
/// memory once streaming grows VRAM use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&r) = map.get(&(address, *desc)) {
        return Some(r as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let mut out = std::ptr::null();
    let made = create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    map.insert((address, *desc), out as usize);
    resource(out).host.is_some().then_some(out)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as GTA VI asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        buffer_data_reaches_the_host_texture_with(false);
    }

    /// The same on asynchronous queues (experimental): the submission runs
    /// on the queue worker, and the title's way to know it ran, a signal
    /// after it and a fence wait, is what the test waits on before reading
    /// the texture back.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture_on_asynchronous_queues() {
        buffer_data_reaches_the_host_texture_with(true);
    }

    #[cfg(windows)]
    fn buffer_data_reaches_the_host_texture_with(asynchronous: bool) {
        use windows::Win32::Graphics::Direct3D12::*;
        crate::queue::asynchronous_for_tests(asynchronous);
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        // Copied first, so the source's upload memory is not the first of
        // the list's: a copy reading the wrong offset would leave these
        // bytes in the texture.
        let mut other = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut other
            ),
            S_OK
        );
        let (first, _) = buffer_bytes(other).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        unsafe { first.write_bytes(0xee, 1024) };

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        let from_other = Location {
            resource: other,
            ..from
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from_other, null.cast());
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        if asynchronous {
            let mut fence = null;
            let fence_iid = &crate::fence::FENCE_IIDS[0];
            assert_eq!(
                crate::fence::create(null, 0, 0, fence_iid, &mut fence),
                S_OK
            );
            assert_eq!(crate::queue::signal_for_tests(queue, fence, 1), S_OK);
            let value = slot::<extern "system" fn(This) -> u64>(fence, 0x40);
            let started = std::time::Instant::now();
            while value(fence) < 1 {
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(10),
                    "the queue worker never signalled after the submission"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            crate::queue::asynchronous_for_tests(false);
        }

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let before = host_copy(image)
            .unwrap()
            .state
            .swap(0x800, std::sync::atomic::Ordering::Relaxed);
        assert_ne!(before, 0x800, "the texture moves to the copy source");
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    /// Writes 256 dwords, `0xc0de0000 | index`, through the root UAV.
    #[cfg(windows)]
    const WRITER: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { o.Store(id.x * 4, 0xC0DE0000u | id.x); }
"#;

    /// A buffer copy whose source shaders earlier in the same list write
    /// (invariant C5): a compute pass writes buffer A, the title copies A to
    /// B in the same list, and B must hold what the pass wrote. Copied on
    /// the CPU at submission, as every copy was before, B gets A's bytes from
    /// before the pass ran; the GPU copy gets the pass's. A's own title
    /// memory gets the pass's bytes either way (copied back).
    #[cfg(windows)]
    #[test]
    fn a_copy_reads_what_earlier_shaders_in_its_list_wrote() {
        if crate::host::host().is_none() {
            assert!(
                !crate::warp::required(),
                "VICTORY_WARP_REQUIRED=1, but the GPU layer has no host device"
            );
            return;
        }
        let written = |bytes: &[u8]| {
            let (words, _) = bytes.as_chunks::<4>();
            words
                .iter()
                .enumerate()
                .all(|(i, w)| u32::from_le_bytes(*w) == 0xC0DE_0000 | i as u32)
        };
        // CopyBufferRegion, then CopyMemoryToMemoryX (title addresses).
        for memory in [false, true] {
            let (a, b) = copy_after_a_shader_write(true, memory);
            assert!(written(&a), "the pass wrote A: {:02x?}", &a[..8]);
            assert!(
                b.iter().all(|&x| x == 0x11),
                "on the CPU the copy took A's bytes from before the pass: {:02x?}",
                &b[..8]
            );
            let (a, b) = copy_after_a_shader_write(false, memory);
            assert!(written(&a), "the pass wrote A: {:02x?}", &a[..8]);
            assert!(
                written(&b),
                "the GPU copy took what the pass wrote: {:02x?}",
                &b[..8]
            );
        }
    }

    /// Title buffers A (bytes 0x11) and B (0x22); a list in which our own
    /// compute pass writes 1 KiB into A's host copy, declared written as a
    /// title dispatch's writes are, then the title's copy of A to B,
    /// CopyBufferRegion or (`memory`) CopyMemoryToMemoryX of their
    /// addresses (`cpu_copies`: on the CPU at submission, as before GPU
    /// copies); submitted through a title queue and waited for. A's and B's
    /// title memory afterwards.
    #[cfg(windows)]
    fn copy_after_a_shader_write(cpu_copies: bool, memory: bool) -> (Vec<u8>, Vec<u8>) {
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        let (mut a, mut b) = (null, null);
        for (out, fill) in [(&mut a, 0x11u8), (&mut b, 0x22)] {
            let desc = buffer(1024);
            assert_eq!(
                create_committed(null, &PROPERTIES, 0, &desc, 0, null.cast(), iid, out),
                S_OK
            );
            let (data, _) = buffer_bytes(*out).unwrap();
            // SAFETY: the buffer holds 1024 bytes.
            unsafe { data.write_bytes(fill, 1024) };
        }
        let at = |r: This| buffer_bytes(r).unwrap().0 as u64;
        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );
        let host_list = crate::list::host_list(list).expect("a host list");
        let writer = crate::copy::compute(WRITER, windows::core::s!("writer")).expect("a writer");
        let shadow = crate::shadow::find(at(a), 1024).expect("A's host copy");
        // As a title dispatch writing A: A's data first, its writes declared.
        shadow.refresh(host_list, at(a), 1024);
        shadow.gpu_writes(host_list, at(a), 1024);
        // SAFETY: an open list, our pipeline, a live host copy.
        unsafe {
            let l = &host_list.list;
            l.SetComputeRootSignature(&writer.root);
            l.SetPipelineState(&writer.pso);
            l.SetComputeRootUnorderedAccessView(0, shadow.gpu(at(a)));
            l.Dispatch(4, 1, 1);
        }
        host_list.keep(&writer.pso);
        host_list.keep(&writer.root);
        crate::list::CPU_COPIES_FOR_TESTS.set(cpu_copies);
        if memory {
            // Called directly: its slot depends on the title's list layout,
            // which a device asked for at the newest level (another test)
            // switches for every list made after it.
            crate::list::copy_memory(list, at(b), at(a), 1024, 0);
        } else {
            type CopyBuffer = extern "system" fn(This, This, u64, This, u64, u64);
            slot::<CopyBuffer>(list, 0x78)(list, b, 0, a, 0, 1024);
        }
        crate::list::CPU_COPIES_FOR_TESTS.set(false);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        let read = |r: This| {
            let (data, _) = buffer_bytes(r).unwrap();
            // SAFETY: the buffer holds 1024 bytes; its submission has run.
            unsafe { std::slice::from_raw_parts(data, 1024) }.to_vec()
        };
        (read(a), read(b))
    }

    /// A copy location as titles pass it (the host's layout).
    #[cfg(windows)]
    #[repr(C)]
    struct At {
        resource: This,
        kind: u32,
        footprint: PlacedFootprint,
    }

    /// A title texture made from `desc`, filled from `source` (a title
    /// buffer laid out as `up`) and copied back, the `area` box of it, into
    /// a title buffer of `len` bytes of 0x33 at texel `xyz` of footprint
    /// `down`, in one list submitted through a title queue: that buffer's
    /// title memory afterwards.
    #[cfg(windows)]
    fn texture_round_trip(
        desc: &ResourceDesc,
        (up, source): (PlacedFootprint, &[u8]),
        (down, xyz, area): (PlacedFootprint, (u32, u32, u32), [u32; 6]),
        len: u64,
    ) -> Vec<u8> {
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut from, mut to) = (null, null, null);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                desc,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        for (out, bytes) in [(&mut from, source.len() as u64), (&mut to, len)] {
            let d = buffer(bytes);
            assert_eq!(
                create_committed(null, &PROPERTIES, 0, &d, 0, null.cast(), iid, out),
                S_OK
            );
        }
        let (data, _) = buffer_bytes(from).unwrap();
        // SAFETY: the buffer holds the source's bytes.
        unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), data, source.len()) };
        let (data, _) = buffer_bytes(to).unwrap();
        // SAFETY: the buffer holds `len` bytes.
        unsafe { data.write_bytes(0x33, len as usize) };
        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );
        let subresource = PlacedFootprint {
            offset: 0,
            ..NO_FOOTPRINT
        };
        type Copy = extern "system" fn(This, *const At, u32, u32, u32, *const At, *const u32);
        let copy = slot::<Copy>(list, 0x80);
        let upload = (
            At {
                resource: image,
                kind: 0,
                footprint: subresource,
            },
            At {
                resource: from,
                kind: 1,
                footprint: up,
            },
        );
        copy(list, &upload.0, 0, 0, 0, &upload.1, null.cast());
        let back = (
            At {
                resource: to,
                kind: 1,
                footprint: down,
            },
            At {
                resource: image,
                kind: 0,
                footprint: subresource,
            },
        );
        let (x, y, z) = xyz;
        copy(list, &back.0, x, y, z, &back.1, area.as_ptr());
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        let (data, _) = buffer_bytes(to).unwrap();
        // SAFETY: the buffer holds `len` bytes; its submission has run.
        unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec()
    }

    /// Texture to buffer: an 8x4 RGBA8 texture whose texel (x, y) holds
    /// `[x, y, 0xa0 + x, 0xb0 + y]`; texels 2..6 x 1..3 of it are copied to
    /// texel (1, 1) of a footprint at offset 512 (row pitch 256) in a buffer
    /// of 0x33: exactly those rows, at those places, are written; every
    /// other byte keeps 0x33.
    #[cfg(windows)]
    #[test]
    fn texture_regions_reach_title_buffers() {
        if crate::host::host().is_none() {
            assert!(
                !crate::warp::required(),
                "VICTORY_WARP_REQUIRED=1, but the GPU layer has no host device"
            );
            return;
        }
        let texel = |x: usize, y: usize| [x as u8, y as u8, 0xA0 + x as u8, 0xB0 + y as u8];
        let up = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 8,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let mut source = vec![0u8; 1024];
        for y in 0..4 {
            for x in 0..8 {
                source[y * 256 + x * 4..][..4].copy_from_slice(&texel(x, y));
            }
        }
        let down = PlacedFootprint { offset: 512, ..up };
        let out = texture_round_trip(
            &texture(8, 4, 28, 1),
            (up, &source),
            (down, (1, 1, 0), [2, 1, 0, 6, 3, 1]),
            2048,
        );
        let mut want = vec![0x33u8; 2048];
        for y in 1..3 {
            for x in 1..5 {
                want[512 + y * 256 + x * 4..][..4].copy_from_slice(&texel(x + 1, y));
            }
        }
        for (i, (&got, &w)) in out.iter().zip(&want).enumerate() {
            assert_eq!(got, w, "byte {i:#x}");
        }
    }

    /// Texture to buffer in a block-compressed format: an 8x8 BC1 texture
    /// whose block (bx, by) holds `[bx, by, 0xc0, ..., 0xc5]`; the box of
    /// texels 4..8 x 0..8 (block column 1, both block rows) goes to a
    /// footprint at offset 1024: two 8-byte blocks, a row pitch apart.
    #[cfg(windows)]
    #[test]
    fn compressed_texture_regions_reach_title_buffers() {
        if crate::host::host().is_none() {
            assert!(
                !crate::warp::required(),
                "VICTORY_WARP_REQUIRED=1, but the GPU layer has no host device"
            );
            return;
        }
        let block = |bx: usize, by: usize| [bx as u8, by as u8, 0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5];
        let up = PlacedFootprint {
            offset: 0,
            format: 71,
            width: 8,
            height: 8,
            depth: 1,
            row_pitch: 256,
        };
        let mut source = vec![0u8; 1024];
        for by in 0..2 {
            for bx in 0..2 {
                source[by * 256 + bx * 8..][..8].copy_from_slice(&block(bx, by));
            }
        }
        let down = PlacedFootprint { offset: 1024, ..up };
        let out = texture_round_trip(
            &texture(8, 8, 71, 1),
            (up, &source),
            (down, (0, 0, 0), [4, 0, 0, 8, 8, 1]),
            2048,
        );
        let mut want = vec![0x33u8; 2048];
        for by in 0..2 {
            want[1024 + by * 256..][..8].copy_from_slice(&block(1, by));
        }
        for (i, (&got, &w)) in out.iter().zip(&want).enumerate() {
            assert_eq!(got, w, "byte {i:#x}");
        }
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    /// A title queue and title lists submitted through it together, for
    /// submissions where later lists depend on earlier lists' GPU work
    /// through title memory (invariant C5, `order`). Work at submission
    /// runs on the CPU before any GPU work, so as one part (the behaviour
    /// before parts) a later list's copy takes what was there before an
    /// earlier list's shaders ran; cut, each part runs and is copied back
    /// first.
    #[cfg(windows)]
    struct Lists {
        queue: This,
        lists: Vec<This>,
    }

    #[cfg(windows)]
    impl Lists {
        fn new() -> Lists {
            let null = std::ptr::null();
            let iid = guid(
                0x0ec870a6,
                0x5d7e,
                0x4c22,
                [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
            );
            let (desc, mut queue) = ([0u32; 4], null);
            assert_eq!(
                crate::queue::create(null, desc.as_ptr().cast(), &iid, &mut queue),
                S_OK
            );
            Lists {
                queue,
                lists: Vec::new(),
            }
        }

        /// A new title list, open, and its host list.
        fn list(&mut self) -> (This, &'static crate::host::HostList) {
            let null = std::ptr::null();
            let allocator_iid = guid(
                0x6102dee4,
                0xaf59,
                0x4b09,
                [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
            );
            let list_iid = guid(
                0x5b160d0f,
                0xac1b,
                0x4185,
                [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
            );
            let (mut allocator, mut list) = (null, null);
            assert_eq!(
                crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
                S_OK
            );
            assert_eq!(
                crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
                S_OK
            );
            self.lists.push(list);
            (list, crate::list::host_list(list).expect("a host list"))
        }

        /// Closes the lists and submits them together, in order.
        fn run(&self) {
            for &list in &self.lists {
                let close = slot::<extern "system" fn(This) -> HRESULT>(list, 0x48);
                assert_eq!(close(list), S_OK);
            }
            let submit = slot::<extern "system" fn(This, u32, *const This)>(self.queue, 0x50);
            submit(self.queue, self.lists.len() as u32, self.lists.as_ptr());
        }

        /// Waits, as a title does, for a fence the queue signals after what
        /// was submitted (run by the queue worker with asynchronous queues).
        fn wait(&self) {
            let null = std::ptr::null();
            let mut fence = null;
            let fence_iid = &crate::fence::FENCE_IIDS[0];
            assert_eq!(
                crate::fence::create(null, 0, 0, fence_iid, &mut fence),
                S_OK
            );
            assert_eq!(crate::queue::signal_for_tests(self.queue, fence, 1), S_OK);
            let value = slot::<extern "system" fn(This) -> u64>(fence, 0x40);
            let started = std::time::Instant::now();
            while value(fence) < 1 {
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(10),
                    "the queue worker never signalled after the submission"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        /// Runs the lists as one part (`cut` false) or cut where they depend
        /// on each other, on this thread; the parts the submission ran in.
        fn run_cut(&self, cut: bool) -> usize {
            crate::host::UNCUT_FOR_TESTS.set(!cut);
            self.run();
            crate::host::UNCUT_FOR_TESTS.set(false);
            crate::host::PARTS_FOR_TESTS.get()
        }
    }

    /// Whether to skip: no host device (a failure with
    /// `VICTORY_WARP_REQUIRED=1`).
    #[cfg(windows)]
    fn no_host_device() -> bool {
        if crate::host::host().is_some() {
            return false;
        }
        assert!(
            !crate::warp::required(),
            "VICTORY_WARP_REQUIRED=1, but the GPU layer has no host device"
        );
        true
    }

    /// A committed title buffer of `len` bytes of `fill`, and its address.
    #[cfg(windows)]
    fn filled(len: u64, fill: u8) -> (This, usize) {
        let null = std::ptr::null();
        let mut out = null;
        let desc = buffer(len);
        let iid = &RESOURCE_IIDS[0];
        assert_eq!(
            create_committed(null, &PROPERTIES, 0, &desc, 0, null.cast(), iid, &mut out),
            S_OK
        );
        let (data, _) = buffer_bytes(out).unwrap();
        // SAFETY: the buffer holds `len` bytes.
        unsafe { data.write_bytes(fill, len as usize) };
        (out, data as usize)
    }

    /// Title memory `[at, at + len)`, as the title reads it once the
    /// submission has run.
    #[cfg(windows)]
    fn title_bytes(at: usize, len: usize) -> Vec<u8> {
        // SAFETY: memory of the test's own buffers.
        unsafe { std::slice::from_raw_parts(at as *const u8, len) }.to_vec()
    }

    /// The bytes [`WRITER`] writes as dwords `range`.
    #[cfg(windows)]
    fn dwords(range: std::ops::Range<u32>) -> Vec<u8> {
        let bytes = |i: u32| (0xC0DE_0000 | i).to_le_bytes();
        range.flat_map(bytes).collect()
    }

    /// Our compute pass writes dwords `0xc0de0000 | i` over `len` bytes (a
    /// multiple of 256) of title memory from `at` on, through `copy` (a
    /// host copy holding them), declared written as a title dispatch's
    /// writes are.
    #[cfg(windows)]
    fn shaders_write(
        list: &crate::host::HostList,
        copy: &std::sync::Arc<crate::shadow::Shadow>,
        at: usize,
        len: usize,
    ) {
        let writer = crate::copy::compute(WRITER, windows::core::s!("writer")).expect("a writer");
        copy.refresh(list, at as u64, len as u64);
        copy.gpu_writes(list, at as u64, len as u64);
        // SAFETY: an open list, our pipeline, a live host copy.
        unsafe {
            let l = &list.list;
            l.SetComputeRootSignature(&writer.root);
            l.SetPipelineState(&writer.pso);
            l.SetComputeRootUnorderedAccessView(0, copy.gpu(at as u64));
            l.Dispatch((len / 256) as u32, 1, 1);
        }
        list.keep(&writer.pso);
        list.keep(&writer.root);
    }

    /// GPU work that reads `len` bytes of title memory at `from` and writes
    /// them at `to` (the ordered copy, through their host copies).
    #[cfg(windows)]
    fn gpu_copies(list: &crate::host::HostList, to: usize, from: usize, len: u64) {
        assert!(crate::copy::buffers(list, to as u64, from as u64, len));
    }

    /// The title's CopyBufferRegion.
    #[cfg(windows)]
    fn title_copies(list: This, (dst, at): (This, u64), (src, from): (This, u64), len: u64) {
        type CopyBuffer = extern "system" fn(This, This, u64, This, u64, u64);
        slot::<CopyBuffer>(list, 0x78)(list, dst, at, src, from, len);
    }

    /// Shaders of one list write A, the next list copies A to B, by
    /// CopyBufferRegion and by CopyMemoryToMemoryX: as one part the copy
    /// takes A's bytes from before the shaders ran; cut, what they wrote.
    #[cfg(windows)]
    #[test]
    fn copies_in_later_lists_read_what_earlier_lists_shaders_wrote() {
        if no_host_device() {
            return;
        }
        for memory in [false, true] {
            for cut in [false, true] {
                let (a, at_a) = filled(1024, 0x11);
                let (b, at_b) = filled(1024, 0x22);
                let mut lists = Lists::new();
                let (_, first) = lists.list();
                let copy = crate::shadow::find(at_a as u64, 1024).expect("A's host copy");
                shaders_write(first, &copy, at_a, 1024);
                let (second, _) = lists.list();
                if memory {
                    crate::list::copy_memory(second, at_b as u64, at_a as u64, 1024, 0);
                } else {
                    title_copies(second, (b, 0), (a, 0), 1024);
                }
                let parts = lists.run_cut(cut);
                let what = if memory { "memory copy" } else { "buffer copy" };
                assert_eq!(title_bytes(at_a, 1024), dwords(0..256), "{what}: A");
                let got = title_bytes(at_b, 1024);
                if cut {
                    assert_eq!(parts, 2, "{what}");
                    assert_eq!(got, dwords(0..256), "{what}: B has what they wrote");
                } else {
                    assert_eq!(parts, 1, "{what}");
                    assert!(
                        got.iter().all(|&x| x == 0x11),
                        "{what}, one part: A's bytes from before: {:02x?}",
                        &got[..8]
                    );
                }
            }
        }
    }

    /// A chain: shaders write A, a copy takes A to B, GPU work reads B into
    /// C, a copy takes C to D. Three parts: the copy after the shaders and
    /// the copy after the GPU copy each start one; GPU work after a copy
    /// needs none (the copy runs first).
    #[cfg(windows)]
    #[test]
    fn dependent_lists_run_in_as_many_parts_as_they_need() {
        if no_host_device() {
            return;
        }
        let (a, at_a) = filled(1024, 0x11);
        let (b, at_b) = filled(1024, 0x22);
        let (c, at_c) = filled(1024, 0x33);
        let (d, at_d) = filled(1024, 0x44);
        let mut lists = Lists::new();
        let (_, first) = lists.list();
        let copy = crate::shadow::find(at_a as u64, 1024).expect("A's host copy");
        shaders_write(first, &copy, at_a, 1024);
        let (second, _) = lists.list();
        title_copies(second, (b, 0), (a, 0), 1024);
        let (_, third) = lists.list();
        gpu_copies(third, at_c, at_b, 1024);
        let (fourth, _) = lists.list();
        title_copies(fourth, (d, 0), (c, 0), 1024);
        assert_eq!(lists.run_cut(true), 3);
        for (name, at) in [("B", at_b), ("C", at_c), ("D", at_d)] {
            assert_eq!(title_bytes(at, 1024), dwords(0..256), "{name}");
        }
    }

    /// Shaders write A's first half; a later list copies 512 bytes of A to B:
    /// from A's second half, sharing no byte, in one part; from 4 bytes
    /// before it, sharing the shaders' last dword, cut.
    #[cfg(windows)]
    #[test]
    fn only_lists_sharing_bytes_are_cut_apart() {
        if no_host_device() {
            return;
        }
        for (from, parts) in [(512usize, 1usize), (508, 2)] {
            let (a, at_a) = filled(1024, 0x11);
            let (b, at_b) = filled(1024, 0x22);
            let mut lists = Lists::new();
            let (_, first) = lists.list();
            let copy = crate::shadow::find(at_a as u64, 1024).expect("A's host copy");
            shaders_write(first, &copy, at_a, 512);
            let (second, _) = lists.list();
            title_copies(second, (b, 0), (a, from as u64), 512);
            assert_eq!(lists.run_cut(true), parts, "copying from {from}");
            let mut a_after = dwords(0..128);
            a_after.extend([0x11; 512]);
            assert_eq!(title_bytes(at_a, 1024), a_after);
            let mut want = a_after[from..from + 512].to_vec();
            want.extend([0x22; 512]);
            assert_eq!(title_bytes(at_b, 1024), want, "copying from {from}");
        }
    }

    /// Two buffers over the same memory, each with its own host copy: P, a
    /// heap's first 192 KiB, and Q, 64 KiB inside it. Shaders write where Q
    /// lies through P's copy; a later list's GPU work reads it through Q's
    /// copy, brought up to date from title memory only: as one part from
    /// before the shaders ran (the heap's zeros), cut from what they wrote.
    #[cfg(windows)]
    #[test]
    fn aliased_buffers_see_earlier_lists_shader_writes() {
        if no_host_device() {
            return;
        }
        for cut in [false, true] {
            let null = std::ptr::null();
            let heap = heap(4 * KIB_64);
            let place = |offset: u64, len: u64, out: &mut This| {
                let desc = buffer(len);
                let iid = &RESOURCE_IIDS[0];
                create_placed(null, heap, offset, &desc, 0, null.cast(), iid, out)
            };
            let (mut p, mut q) = (null, null);
            assert_eq!(place(0, 3 * KIB_64, &mut p), S_OK);
            assert_eq!(place(KIB_64, KIB_64, &mut q), S_OK);
            let at_p = buffer_bytes(p).unwrap().0 as u64;
            let at_q = buffer_bytes(q).unwrap().0 as usize;
            let (_, at_b) = filled(1024, 0x22);
            let whole = crate::shadow::find(at_p, 3 * KIB_64).expect("P's host copy");
            let inner = crate::shadow::find(at_q as u64, 1024).expect("Q's host copy");
            assert!(!std::sync::Arc::ptr_eq(&whole, &inner), "two host copies");
            let mut lists = Lists::new();
            let (_, first) = lists.list();
            shaders_write(first, &whole, at_q, 1024);
            let (_, second) = lists.list();
            gpu_copies(second, at_b, at_q, 1024);
            assert_eq!(lists.run_cut(cut), if cut { 2 } else { 1 });
            let got = title_bytes(at_b, 1024);
            if cut {
                assert_eq!(got, dwords(0..256), "through Q, what was written through P");
            } else {
                assert!(
                    got.iter().all(|&x| x == 0),
                    "one part: Q's copy from before: {:02x?}",
                    &got[..8]
                );
            }
        }
    }

    /// Copies over memory earlier lists' GPU work uses. Shaders write A, a
    /// later list copies W over A: as one part the copy runs first and the
    /// shaders' bytes, copied back after, win; cut, the copy's. GPU work
    /// reads Y into Z1, a later list copies X over Y, GPU work reads Y into
    /// Z2: as one part Y's host copy is brought up to date for the last
    /// reader before the first one runs, so both get X; cut, Z1 gets Y as
    /// it was.
    #[cfg(windows)]
    #[test]
    fn copies_over_memory_earlier_lists_use_wait_for_them() {
        if no_host_device() {
            return;
        }
        for cut in [false, true] {
            let parts = if cut { 2 } else { 1 };
            let (a, at_a) = filled(1024, 0x11);
            let (w, _) = filled(1024, 0x44);
            let mut lists = Lists::new();
            let (_, first) = lists.list();
            let copy = crate::shadow::find(at_a as u64, 1024).expect("A's host copy");
            shaders_write(first, &copy, at_a, 1024);
            let (second, _) = lists.list();
            title_copies(second, (a, 0), (w, 0), 1024);
            assert_eq!(lists.run_cut(cut), parts);
            let got = title_bytes(at_a, 1024);
            if cut {
                assert!(
                    got.iter().all(|&x| x == 0x44),
                    "the copy last: {:02x?}",
                    &got[..8]
                );
            } else {
                assert_eq!(got, dwords(0..256), "one part: the shaders' bytes last");
            }

            let (y, at_y) = filled(1024, 0x33);
            let (x, _) = filled(1024, 0x44);
            let (_, at_z1) = filled(1024, 0x55);
            let (_, at_z2) = filled(1024, 0x66);
            let mut lists = Lists::new();
            let (_, first) = lists.list();
            gpu_copies(first, at_z1, at_y, 1024);
            let (second, _) = lists.list();
            title_copies(second, (y, 0), (x, 0), 1024);
            let (_, third) = lists.list();
            gpu_copies(third, at_z2, at_y, 1024);
            assert_eq!(lists.run_cut(cut), parts);
            let z2 = title_bytes(at_z2, 1024);
            assert!(
                z2.iter().all(|&v| v == 0x44),
                "Z2 reads the copy: {:02x?}",
                &z2[..8]
            );
            let z1 = title_bytes(at_z1, 1024);
            let want = if cut { 0x33 } else { 0x44 };
            assert!(z1.iter().all(|&v| v == want), "Z1: {:02x?}", &z1[..8]);
        }
    }

    /// With asynchronous queues the queue worker runs the submission, cut
    /// as on the title's thread: the copy in the later list gets what the
    /// earlier list's shaders wrote.
    #[cfg(windows)]
    #[test]
    fn asynchronous_queues_cut_submissions_too() {
        if no_host_device() {
            return;
        }
        crate::queue::asynchronous_for_tests(true);
        let (a, at_a) = filled(1024, 0x11);
        let (b, at_b) = filled(1024, 0x22);
        let mut lists = Lists::new();
        let (_, first) = lists.list();
        let copy = crate::shadow::find(at_a as u64, 1024).expect("A's host copy");
        shaders_write(first, &copy, at_a, 1024);
        let (second, _) = lists.list();
        title_copies(second, (b, 0), (a, 0), 1024);
        lists.run();
        lists.wait();
        crate::queue::asynchronous_for_tests(false);
        assert_eq!(title_bytes(at_a, 1024), dwords(0..256));
        assert_eq!(title_bytes(at_b, 1024), dwords(0..256));
    }
}
