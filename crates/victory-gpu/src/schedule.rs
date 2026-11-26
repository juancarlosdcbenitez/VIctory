//! Queue order: what each command queue may run now and what waits.
//!
//! A D3D12 queue runs what is submitted to it in order, and its
//! `Wait(fence, value)` holds back everything submitted after the wait
//! until the fence reaches the value (ID3D12CommandQueue::Wait: "the GPU
//! waits until the specified fence reaches or exceeds the specified
//! value"). Our queues run work when it is submitted (decision 0007), which
//! is right only while no wait holds the queue back. [`Schedule`] decides,
//! for each operation submitted to a queue, whether it may run now or must
//! wait behind a wait not yet reached, and which held operations a fence's
//! move releases, in order.
//!
//! The rules (Microsoft's "Multi-engine synchronization"): a wait is
//! satisfied once the fence reaches its value, even if a lower value
//! replaces it before the queue runs on ("If a race does occur, between a
//! value which will satisfy an outstanding wait, and a lower value which
//! will not, the wait will be satisfied regardless"); a wait submitted
//! after the fence was set lower compares against the lower value.
//!
//! It knows nothing of D3D12: operations are values of `T` and queues and
//! fences are numbers, so it is tested on its own (below).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// What a queue holds in order: waits, and operations behind them.
#[derive(Debug)]
enum Item<T> {
    Wait {
        fence: usize,
        value: u64,
        /// A move of the fence reached the value.
        reached: bool,
    },
    Op(T),
}

/// Per queue (by number), what is held back.
#[derive(Debug)]
pub(crate) struct Schedule<T> {
    queues: BTreeMap<usize, VecDeque<Item<T>>>,
    /// Queues whose released operations are still running (given out by
    /// [`Schedule::runnable`], not yet [`Schedule::finished`]): what is
    /// submitted to them meanwhile waits its turn too.
    busy: BTreeSet<usize>,
}

impl<T> Default for Schedule<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Schedule<T> {
    pub(crate) const fn new() -> Self {
        Schedule {
            queues: BTreeMap::new(),
            busy: BTreeSet::new(),
        }
    }

    /// Whether `queue` holds anything back, or runs what it released: then
    /// what is submitted to it waits too, to keep the queue's order.
    pub(crate) fn holds(&self, queue: usize) -> bool {
        self.busy.contains(&queue) || self.queues.get(&queue).is_some_and(|q| !q.is_empty())
    }

    /// Claims `queue` for work its submitter runs at once: true if nothing
    /// holds it back. Until [`Schedule::finished`], the queue is busy, so
    /// waits and work submitted meanwhile (from other threads) are held
    /// behind it in order instead of running ahead of it.
    pub(crate) fn direct(&mut self, queue: usize) -> bool {
        if self.holds(queue) {
            return false;
        }
        self.busy.insert(queue);
        true
    }

    /// The operations [`Schedule::runnable`] gave out for `queue`, or the
    /// work run at once under [`Schedule::direct`], have run.
    pub(crate) fn finished(&mut self, queue: usize) {
        self.busy.remove(&queue);
    }

    /// `queue` waits for `fence` (now at `current`) to reach `value`. A wait
    /// already reached holds nothing back, unless something is held before
    /// it (then it keeps its place, reached).
    pub(crate) fn wait(&mut self, queue: usize, fence: usize, value: u64, current: u64) {
        let reached = current >= value;
        if reached && !self.holds(queue) {
            return;
        }
        self.queues.entry(queue).or_default().push_back(Item::Wait {
            fence,
            value,
            reached,
        });
    }

    /// An operation submitted to `queue`: given back to run now if nothing
    /// holds the queue back, else kept, in order, for [`Schedule::runnable`].
    pub(crate) fn submit(&mut self, queue: usize, op: T) -> Option<T> {
        if !self.holds(queue) {
            return Some(op);
        }
        self.queues
            .entry(queue)
            .or_default()
            .push_back(Item::Op(op));
        None
    }

    /// An operation kept for [`Schedule::runnable`] even if nothing holds
    /// `queue` back: for queues whose work all runs on the worker.
    pub(crate) fn defer(&mut self, queue: usize, op: T) {
        self.queues
            .entry(queue)
            .or_default()
            .push_back(Item::Op(op));
    }

    /// `fence` moved to `value` (a signal, or a write to its memory): the
    /// held waits it reaches are satisfied from now on.
    pub(crate) fn fence_moved(&mut self, fence: usize, value: u64) {
        for q in self.queues.values_mut() {
            for item in q.iter_mut() {
                if let Item::Wait {
                    fence: f,
                    value: v,
                    reached,
                } = item
                    && *f == fence
                    && *v <= value
                {
                    *reached = true;
                }
            }
        }
    }

    /// The held operations nothing holds back any more, queue by queue in
    /// each queue's order: from each queue's front, satisfied waits are
    /// dropped and operations taken until a wait not yet reached. A queue
    /// that gives some out is busy until [`Schedule::finished`]; a busy
    /// queue gives none (its earlier ones are still running).
    pub(crate) fn runnable(&mut self) -> Vec<(usize, T)> {
        let mut out = Vec::new();
        for (&queue, q) in self.queues.iter_mut() {
            if self.busy.contains(&queue) {
                continue;
            }
            let before = out.len();
            while let Some(front) = q.front() {
                match front {
                    Item::Wait { reached: false, .. } => break,
                    Item::Wait { .. } => {
                        q.pop_front();
                    }
                    Item::Op(_) => {
                        let Some(Item::Op(op)) = q.pop_front() else {
                            unreachable!()
                        };
                        out.push((queue, op));
                    }
                }
            }
            if out.len() > before {
                self.busy.insert(queue);
            }
        }
        self.queues.retain(|_, q| !q.is_empty());
        out
    }

    /// The waits that hold queues back: `(queue, fence, value)`, the first
    /// of each queue.
    pub(crate) fn blocked(&self) -> Vec<(usize, usize, u64)> {
        self.queues
            .iter()
            .filter_map(|(&queue, q)| {
                q.iter().find_map(|item| match *item {
                    Item::Wait {
                        fence,
                        value,
                        reached: false,
                    } => Some((queue, fence, value)),
                    _ => None,
                })
            })
            .collect()
    }

    /// The blocked waits that only held-back operations could satisfy:
    /// every operation that signals their fence to their value or more
    /// (`signals` says what an operation signals) is itself held. Unless a
    /// CPU thread signals the fence, those queues never run on: a deadlock
    /// in the title's use of its queues, or work our queues hold that
    /// D3D12 would run.
    pub(crate) fn stuck(
        &self,
        signals: impl Fn(&T) -> Option<(usize, u64)>,
    ) -> Vec<(usize, usize, u64)> {
        let held_signals: Vec<(usize, u64)> = self
            .queues
            .values()
            .flatten()
            .filter_map(|item| match item {
                Item::Op(op) => signals(op),
                Item::Wait { .. } => None,
            })
            .collect();
        self.blocked()
            .into_iter()
            .filter(|&(_, fence, value)| {
                held_signals.iter().any(|&(f, v)| f == fence && v >= value)
            })
            .collect()
    }

    /// How many operations are held, over all queues.
    pub(crate) fn held(&self) -> usize {
        self.queues
            .values()
            .flatten()
            .filter(|item| matches!(item, Item::Op(_)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test operation: a name, and what it signals when it runs.
    #[derive(Debug, Clone, PartialEq)]
    struct Op(&'static str, Option<(usize, u64)>);

    const A: usize = 1;
    const B: usize = 2;
    const F: usize = 10;
    const G: usize = 11;

    fn names(ops: Vec<(usize, Op)>) -> Vec<(usize, &'static str)> {
        ops.into_iter().map(|(q, op)| (q, op.0)).collect()
    }

    /// Work its submitter runs at once claims the queue: a wait and work
    /// another thread submits meanwhile are held behind it, in order, and
    /// run only once it has finished.
    #[test]
    fn work_run_at_once_holds_what_is_submitted_meanwhile() {
        let mut s: Schedule<u32> = Schedule::new();
        assert!(s.direct(A), "nothing held: run at once");
        assert!(s.holds(A));
        // Meanwhile, another thread: a wait already reached, then work.
        s.wait(A, F, 1, 1);
        assert_eq!(s.submit(A, 7), None, "held behind the running work");
        assert!(s.runnable().is_empty(), "busy");
        s.finished(A);
        assert_eq!(s.runnable(), vec![(A, 7)]);
        assert!(!s.direct(A), "busy with the released work");
        s.finished(A);
        assert!(s.direct(A));
        s.finished(A);
        assert!(!s.holds(A));
    }

    /// Deferred work (queues whose work all runs on the worker) is kept
    /// even on a free queue, released in order with what follows it, and a
    /// wait not reached holds back what is deferred after it.
    #[test]
    fn deferred_work_runs_on_the_worker_in_order() {
        let mut s: Schedule<u32> = Schedule::new();
        s.defer(A, 1);
        s.defer(A, 2);
        assert!(s.holds(A), "kept for the worker");
        assert!(!s.direct(A));
        assert_eq!(s.runnable(), vec![(A, 1), (A, 2)]);
        s.finished(A);
        s.wait(A, F, 4, 0);
        s.defer(A, 3);
        assert!(s.runnable().is_empty(), "behind the wait");
        s.fence_moved(F, 4);
        assert_eq!(s.runnable(), vec![(A, 3)]);
        s.finished(A);
        assert!(!s.holds(A));
    }

    #[test]
    fn without_waits_everything_runs_at_once() {
        let mut s = Schedule::default();
        assert_eq!(s.submit(A, Op("draw", None)), Some(Op("draw", None)));
        s.wait(A, F, 3, 5);
        assert_eq!(
            s.submit(A, Op("after a reached wait", None)).map(|o| o.0),
            Some("after a reached wait")
        );
        assert!(!s.holds(A) && s.runnable().is_empty());
    }

    /// A wait not reached holds back everything submitted to its queue
    /// after it, in order, until the fence reaches the value; a later wait
    /// not reached holds back what follows it.
    #[test]
    fn a_wait_holds_its_queue_until_the_fence_reaches_it() {
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert!(s.holds(A));
        assert_eq!(s.submit(A, Op("first", None)), None);
        assert_eq!(s.submit(A, Op("second", None)), None);
        s.wait(A, G, 4, 0);
        assert_eq!(s.submit(A, Op("third", None)), None);
        // Other queues are not held.
        assert_eq!(s.submit(B, Op("other", None)).map(|o| o.0), Some("other"));
        assert!(s.runnable().is_empty(), "nothing reached yet");
        s.fence_moved(F, 1);
        assert_eq!(names(s.runnable()), [(A, "first"), (A, "second")]);
        assert!(s.holds(A), "busy running them");
        s.finished(A);
        assert_eq!(s.blocked(), [(A, G, 4)]);
        s.fence_moved(G, 3);
        assert!(s.runnable().is_empty(), "3 is below 4");
        s.fence_moved(G, 9);
        assert_eq!(names(s.runnable()), [(A, "third")]);
        s.finished(A);
        assert!(!s.holds(A));
        assert_eq!(s.held(), 0);
    }

    /// A wait reached by a value that a lower one then replaces is
    /// satisfied all the same; a wait submitted after the lower value
    /// compares against it.
    #[test]
    fn a_rewound_fence_still_satisfies_the_wait_it_reached() {
        let mut s = Schedule::default();
        s.wait(A, F, 5, 0);
        assert_eq!(s.submit(A, Op("held", None)), None);
        s.fence_moved(F, 7);
        s.fence_moved(F, 3);
        assert_eq!(names(s.runnable()), [(A, "held")]);
        s.finished(A);
        // A new wait for 5, the fence now at 3.
        s.wait(A, F, 5, 3);
        assert_eq!(s.submit(A, Op("later", None)), None);
        assert!(s.runnable().is_empty());
    }

    /// One fence's move releases every queue waiting for it; a queue's
    /// held work that signals another queue's fence releases that queue
    /// once it runs (the caller reports the signal).
    #[test]
    fn moves_release_every_waiting_queue_and_chains_follow() {
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        s.wait(B, F, 1, 0);
        assert_eq!(s.submit(A, Op("a", Some((G, 1)))), None);
        s.wait(B, G, 1, 0);
        assert_eq!(s.submit(B, Op("b", None)), None);
        s.fence_moved(F, 1);
        let ran = s.runnable();
        assert_eq!(names(ran.clone()), [(A, "a")], "b also waits for G");
        // "a" ran and signalled G.
        for (queue, op) in ran {
            if let Some((fence, value)) = op.1 {
                s.fence_moved(fence, value);
            }
            s.finished(queue);
        }
        assert_eq!(names(s.runnable()), [(B, "b")]);
    }

    /// While a queue's released operations run, what is submitted to it
    /// waits behind them, and comes out next.
    #[test]
    fn work_submitted_while_released_work_runs_waits_its_turn() {
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert_eq!(s.submit(A, Op("held", None)), None);
        s.fence_moved(F, 1);
        assert_eq!(names(s.runnable()), [(A, "held")]);
        // Still running "held": a new submission does not overtake it.
        assert_eq!(s.submit(A, Op("next", None)), None);
        assert!(s.runnable().is_empty(), "busy");
        s.finished(A);
        assert_eq!(names(s.runnable()), [(A, "next")]);
        s.finished(A);
        assert_eq!(s.submit(A, Op("free", None)).map(|o| o.0), Some("free"));
    }

    /// A wait that only held-back work signals is stuck: on its own queue
    /// (it signals behind its own wait) or across two queues that each wait
    /// for the other's held signal. A wait a CPU thread may still signal is
    /// not.
    #[test]
    fn waits_only_held_work_could_satisfy_are_stuck() {
        let signals = |op: &Op| op.1;
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert_eq!(s.submit(A, Op("signals F", Some((F, 1)))), None);
        assert_eq!(s.stuck(signals), [(A, F, 1)]);
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert_eq!(s.submit(A, Op("signals G", Some((G, 2)))), None);
        s.wait(B, G, 2, 0);
        assert_eq!(s.submit(B, Op("signals F", Some((F, 1)))), None);
        assert_eq!(s.stuck(signals), [(A, F, 1), (B, G, 2)]);
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert_eq!(s.submit(A, Op("draw", None)), None);
        assert!(s.stuck(signals).is_empty(), "a CPU signal may still come");
    }

    /// A wait already reached behind held work keeps its place (it does
    /// not let what follows it run early).
    #[test]
    fn a_reached_wait_behind_held_work_keeps_the_order() {
        let mut s = Schedule::default();
        s.wait(A, F, 1, 0);
        assert_eq!(s.submit(A, Op("first", None)), None);
        s.wait(A, G, 1, 1);
        assert_eq!(s.submit(A, Op("second", None)), None);
        assert!(s.runnable().is_empty());
        s.fence_moved(F, 1);
        assert_eq!(names(s.runnable()), [(A, "first"), (A, "second")]);
    }
}

/// The rules the model follows, on a real D3D12 queue (WARP: these run
/// on Windows CI, which has no GPU).
#[cfg(all(test, windows))]
mod d3d12 {
    use std::time::Duration;

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    fn buffer(
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        d: &ID3D12Device,
    ) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
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

    /// Waits up to `limit` for `fence` to reach `value`.
    fn reaches(fence: &ID3D12Fence, value: u64, limit: Duration) -> bool {
        // SAFETY: an event of ours, waited on and left to the process.
        unsafe {
            let event = CreateEventW(None, false, false, None).expect("an event");
            fence
                .SetEventOnCompletion(value, event)
                .expect("an event wait");
            WaitForSingleObject(event, limit.as_millis() as u32);
            fence.GetCompletedValue() >= value
        }
    }

    /// A queue told to wait for a fence holds back the copy submitted after
    /// the wait until a CPU signal reaches the value: the copy reads data
    /// written after it was submitted. And a wait reached by a value that
    /// a lower one replaces at once is satisfied all the same.
    #[test]
    fn d3d12_queue_waits_hold_work_until_the_fence_reaches_them() {
        // SAFETY: plain D3D12 calls on objects of ours; the CPU writes the
        // upload buffer only while no GPU work reads it, as the waits say.
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
            let src = buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                &d,
            );
            let back = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &d);
            let mut up = std::ptr::null_mut();
            src.Map(0, None, Some(&mut up)).expect("map");
            let up = up.cast::<u32>();
            let alloc: ID3D12CommandAllocator = d
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("an allocator");
            let list: ID3D12GraphicsCommandList = d
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("a list");
            list.CopyBufferRegion(&back, 0, &src, 0, 4);
            list.Close().expect("close");
            let lists = [Some(
                windows::core::Interface::cast::<ID3D12CommandList>(&list).expect("a list"),
            )];
            let read = || {
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).expect("map");
                let v = *p.cast::<u32>();
                back.Unmap(0, None);
                v
            };

            // Held until the CPU signals; then it copies what is there then.
            let (f, g): (ID3D12Fence, ID3D12Fence) = (
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
            );
            *up = 1;
            queue.Wait(&f, 1).expect("wait");
            queue.ExecuteCommandLists(&lists);
            queue.Signal(&g, 1).expect("signal");
            assert!(
                !reaches(&g, 1, Duration::from_millis(200)),
                "the copy ran before the wait was reached"
            );
            *up = 2;
            f.Signal(1).expect("a CPU signal");
            assert!(
                reaches(&g, 1, Duration::from_secs(10)),
                "the copy did not run once the wait was reached"
            );
            assert_eq!(
                read(),
                2,
                "the copy read the data written after it was submitted"
            );

            // A wait for 5; the fence goes to 7 and straight back to 3.
            let (f, g): (ID3D12Fence, ID3D12Fence) = (
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
                d.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence"),
            );
            *up = 3;
            queue.Wait(&f, 5).expect("wait");
            queue.ExecuteCommandLists(&lists);
            queue.Signal(&g, 1).expect("signal");
            f.Signal(7).expect("a CPU signal");
            f.Signal(3).expect("a CPU signal");
            assert!(
                reaches(&g, 1, Duration::from_secs(10)),
                "a wait reached then rewound was not satisfied"
            );
            assert_eq!(read(), 3);
        }
    }
}
