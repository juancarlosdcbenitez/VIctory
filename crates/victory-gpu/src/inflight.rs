//! What overlapping submissions will need in place of the CPU wait that
//! ends every submission today (`docs/architecture.md`, "GPU submission
//! and its invariants"; PERF-2): [`Retire`] keeps what a submission uses
//! alive until the host fence passes it (invariant O2), and [`InFlight`]
//! says which submission in flight writes a range of title memory, so
//! preparation can wait for exactly that one before reading it (C3).
//!
//! Nothing uses them yet: every submission still waits for the GPU, so
//! neither would have anything to do. They are tested on their own and,
//! for [`Retire`], against a real D3D12 queue on WARP.

#![allow(dead_code)] // the foundations of PERF-2, used by tests until submissions overlap

use std::collections::VecDeque;

/// Things submissions use, each kept until the host fence passes the value
/// its submission signals. Values only grow (one host timeline).
#[derive(Debug)]
pub(crate) struct Retire<T> {
    kept: VecDeque<(u64, T)>,
}

impl<T> Default for Retire<T> {
    fn default() -> Self {
        Self {
            kept: VecDeque::new(),
        }
    }
}

impl<T> Retire<T> {
    /// Keeps `item` until the fence reaches `value`.
    pub(crate) fn keep(&mut self, value: u64, item: T) {
        debug_assert!(
            self.kept.back().is_none_or(|&(last, _)| last <= value),
            "host fence values only grow"
        );
        self.kept.push_back((value, item));
    }

    /// What the work up to `completed` used, now free, oldest first.
    pub(crate) fn release(&mut self, completed: u64) -> Vec<T> {
        let mut free = Vec::new();
        while self
            .kept
            .front()
            .is_some_and(|&(value, _)| value <= completed)
        {
            free.extend(self.kept.pop_front().map(|(_, item)| item));
        }
        free
    }

    /// How many things are still kept.
    pub(crate) fn len(&self) -> usize {
        self.kept.len()
    }
}

/// Ranges of title memory that GPU work of submissions in flight writes,
/// each to be copied back when its submission finishes. Submissions are
/// numbered by the host fence value they signal.
#[derive(Debug, Default)]
pub(crate) struct InFlight {
    /// `(start, end, submission)`, submissions in increasing order.
    ranges: VecDeque<(usize, usize, u64)>,
}

impl InFlight {
    /// Submission `value` writes `[start, start + len)`.
    pub(crate) fn writes(&mut self, value: u64, start: usize, len: usize) {
        debug_assert!(
            self.ranges.back().is_none_or(|&(_, _, last)| last <= value),
            "submissions are recorded in order"
        );
        if len > 0 {
            self.ranges.push_back((start, start + len, value));
        }
    }

    /// Submissions up to `completed` have finished and been copied back.
    pub(crate) fn finished(&mut self, completed: u64) {
        while self.ranges.front().is_some_and(|&(_, _, v)| v <= completed) {
            self.ranges.pop_front();
        }
    }

    /// The newest submission in flight that writes any byte of
    /// `[start, start + len)`: what reading that range must wait for.
    pub(crate) fn latest_writer(&self, start: usize, len: usize) -> Option<u64> {
        let end = start + len;
        self.ranges
            .iter()
            .rev()
            .find(|&&(s, e, _)| s < end && start < e)
            .map(|&(_, _, v)| v)
    }

    /// Whether no submission in flight writes title memory.
    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_things_are_released_once_their_value_is_reached() {
        let mut r = Retire::default();
        r.keep(1, "a");
        r.keep(2, "b");
        r.keep(2, "c");
        r.keep(5, "d");
        assert!(r.release(0).is_empty());
        assert_eq!(r.release(1), ["a"]);
        assert_eq!(
            r.release(4),
            ["b", "c"],
            "everything up to the value, in order"
        );
        assert_eq!(r.len(), 1);
        assert!(r.release(4).is_empty(), "nothing twice");
        assert_eq!(r.release(9), ["d"]);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn reads_wait_for_the_newest_submission_writing_any_byte_of_them() {
        let mut f = InFlight::default();
        f.writes(3, 0x1000, 0x100);
        f.writes(4, 0x1080, 0x100);
        f.writes(6, 0x4000, 0x10);
        f.writes(6, 0x5000, 0);
        assert_eq!(f.latest_writer(0x1000, 4), Some(3));
        assert_eq!(
            f.latest_writer(0x10F0, 0x20),
            Some(4),
            "both overlap: the newer"
        );
        assert_eq!(f.latest_writer(0x1100, 0x80), Some(4));
        assert_eq!(
            f.latest_writer(0x1180, 0x10),
            None,
            "adjacent, not overlapping"
        );
        assert_eq!(f.latest_writer(0x0FF0, 0x10), None);
        assert_eq!(
            f.latest_writer(0x5000, 0x10),
            None,
            "nothing written by an empty range"
        );
        assert_eq!(f.latest_writer(0x3FF8, 0x10), Some(6));
        f.finished(3);
        assert_eq!(f.latest_writer(0x1000, 4), None, "copied back");
        assert_eq!(f.latest_writer(0x10F0, 0x20), Some(4));
        f.finished(6);
        assert!(f.is_empty());
    }
}

/// [`Retire`] against a real D3D12 queue on WARP: eight submissions in
/// flight with no CPU wait between them, each copying from its own upload
/// buffer recorded with its own command allocator; what is kept with each
/// is released only once the fence passes it, every copy reads its own
/// data, and the released allocators are reset and reused.
#[cfg(all(test, windows))]
mod d3d12 {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    use super::{InFlight, Retire};

    fn buffer(
        d: &ID3D12Device,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        bytes: u64,
    ) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation with descriptions of ours.
        unsafe {
            d.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut r,
            )
        }
        .expect("a buffer");
        r.expect("a buffer")
    }

    /// What one submission uses: its upload buffer, allocator and list.
    struct Used {
        upload: ID3D12Resource,
        alloc: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
    }

    #[test]
    fn d3d12_submissions_in_flight_keep_what_they_use_until_the_fence_passes() {
        const N: u64 = 8;
        // SAFETY: plain D3D12 calls on objects of ours; each upload buffer
        // is written before its submission and not again while in flight,
        // and an allocator is reset only once the fence passed its work.
        unsafe {
            let Some(d) = crate::warp::device() else {
                return;
            };
            let queue: ID3D12CommandQueue = d
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("a queue");
            let fence: ID3D12Fence = d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
            let back = buffer(
                &d,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
                4 * 2 * N,
            );
            let mut retire = Retire::default();
            let submit = |slot: u64, value: u32, alloc: ID3D12CommandAllocator| -> Used {
                let upload = buffer(
                    &d,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    256,
                );
                let mut p = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut p)).expect("map");
                *p.cast::<u32>() = value;
                upload.Unmap(0, None);
                let list: ID3D12GraphicsCommandList = d
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                    .expect("a list");
                list.CopyBufferRegion(&back, 4 * slot, &upload, 0, 4);
                list.Close().expect("close");
                queue.ExecuteCommandLists(&[Some(list.cast().expect("a list"))]);
                Used {
                    upload,
                    alloc,
                    list,
                }
            };
            // Eight in flight, nothing waited for.
            for k in 0..N {
                let alloc = d
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .expect("an allocator");
                let used = submit(k, 100 + k as u32, alloc);
                queue.Signal(&fence, k + 1).expect("signal");
                retire.keep(k + 1, used);
            }
            // Only what the GPU is past is free, whenever this is asked.
            let completed = fence.GetCompletedValue();
            let early = retire.release(completed);
            assert_eq!(early.len() as u64, completed.min(N));
            assert_eq!(retire.len() as u64, N - completed.min(N));
            let event = CreateEventW(None, false, false, None).expect("an event");
            fence.SetEventOnCompletion(N, event).expect("an event wait");
            WaitForSingleObject(event, INFINITE);
            let freed: Vec<Used> = early.into_iter().chain(retire.release(N)).collect();
            assert_eq!(freed.len() as u64, N);
            assert_eq!(retire.len(), 0);
            // The freed allocators are reset and record the next round.
            for (k, used) in freed.into_iter().enumerate() {
                used.alloc.Reset().expect("an allocator free to reset");
                drop((used.upload, used.list));
                let used = submit(N + k as u64, 200 + k as u32, used.alloc);
                queue.Signal(&fence, N + 1 + k as u64).expect("signal");
                retire.keep(N + 1 + k as u64, used);
            }
            fence
                .SetEventOnCompletion(2 * N, event)
                .expect("an event wait");
            WaitForSingleObject(event, INFINITE);
            assert_eq!(retire.release(2 * N).len() as u64, N);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("map");
            let words = std::slice::from_raw_parts(p.cast::<u32>(), 2 * N as usize);
            let want: Vec<u32> = (0..N as u32)
                .map(|k| 100 + k)
                .chain((0..N as u32).map(|k| 200 + k))
                .collect();
            assert_eq!(words, &want[..], "each copy read its own upload buffer");
            back.Unmap(0, None);
        }
    }

    /// [`InFlight`] against a real D3D12 queue on WARP: reading a range
    /// needs the newest submission writing it to have finished, and that
    /// alone: two copies run, a third one writing the first one's range is
    /// held behind a queue wait, and the second one's range is read while
    /// it is held (its data, the fence short of the third), then the first
    /// range once the third has run (the third's data).
    #[test]
    fn d3d12_a_read_waits_for_exactly_the_newest_writer_of_its_range() {
        // SAFETY: plain D3D12 calls on objects of ours; each upload buffer
        // is written before its copy is submitted, and the readback buffer
        // is read only where the fence says the copies writing it are done.
        unsafe {
            let Some(d) = crate::warp::device() else {
                return;
            };
            let queue: ID3D12CommandQueue = d
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("a queue");
            let (fence, gate): (ID3D12Fence, ID3D12Fence) = (
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
            );
            let back = buffer(
                &d,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
                256,
            );
            let alloc: ID3D12CommandAllocator = d
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("an allocator");
            let mut kept = Vec::new();
            let mut flight = InFlight::default();
            // Submission `value` copies `word` to bytes `at..at + 4`.
            let mut copy = |value: u64, at: u64, word: u32| {
                let upload = buffer(
                    &d,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    256,
                );
                let mut p = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut p)).expect("map");
                *p.cast::<u32>() = word;
                upload.Unmap(0, None);
                let list: ID3D12GraphicsCommandList = d
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                    .expect("a list");
                list.CopyBufferRegion(&back, at, &upload, 0, 4);
                list.Close().expect("close");
                queue.ExecuteCommandLists(&[Some(list.cast().expect("a list"))]);
                queue.Signal(&fence, value).expect("signal");
                flight.writes(value, at as usize, 4);
                kept.push((upload, list));
            };
            copy(1, 0, 0x11);
            copy(2, 4, 0x22);
            queue.Wait(&gate, 1).expect("wait");
            copy(3, 0, 0x33);
            let event = CreateEventW(None, false, false, None).expect("an event");
            let wait_for = |value: u64| {
                fence
                    .SetEventOnCompletion(value, event)
                    .expect("an event wait");
                WaitForSingleObject(event, INFINITE);
            };
            let word = |at: usize| {
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).expect("map");
                let w = *p.cast::<u8>().add(at).cast::<u32>();
                back.Unmap(0, None);
                w
            };
            // The second range: its newest writer is 2, and that is enough.
            let writer = flight.latest_writer(4, 4).expect("written in flight");
            assert_eq!(writer, 2);
            wait_for(writer);
            assert_eq!(word(4), 0x22);
            assert!(
                fence.GetCompletedValue() < 3,
                "the third copy is still held"
            );
            // The first range: its newest writer is 3, held until the gate.
            let writer = flight.latest_writer(0, 4).expect("written in flight");
            assert_eq!(writer, 3);
            gate.Signal(1).expect("a CPU signal");
            wait_for(writer);
            flight.finished(fence.GetCompletedValue());
            assert!(flight.is_empty());
            assert_eq!(word(0), 0x33, "the newest writer's data");
        }
    }
}
