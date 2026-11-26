//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once, while nothing holds the queue back.
//!
//! A queue's `Wait(fence, value)` holds back what is submitted to the
//! queue after it until the fence reaches the value, as on D3D12: those
//! submissions and signals are kept in order ([`crate::schedule`]) and run
//! by our queue worker thread once the fence gets there (by a signal, a
//! bottom-of-pipe write, or a store we only see by polling). The submitting
//! thread never waits: a title may signal the fence itself after
//! submitting, which D3D12 allows since only the queue waits. A held
//! submission keeps the host lists it was recorded into (a title list reset
//! meanwhile records into a fresh one, `list::reset`).
//!
//! shortcut: presents and page mappings on a held queue run at once (they
//! read title memory at the call); they are counted.
//!
//! Experimental (diagnostics builds, `VICTORY_QUEUE_ASYNC=1`): every
//! submission, signal, present and page mapping is held and run by the
//! queue worker in each queue's order, so title threads never wait for the
//! GPU in a submission; presents and page mappings take what they read at
//! the call. Off by default: not run on a GPU yet.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::schedule::Schedule;

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

/// What a held queue runs later, in order.
enum Held {
    /// A submission: the host lists it was recorded into.
    #[cfg(windows)]
    Execute(Vec<std::sync::Arc<crate::host::HostList>>),
    /// A signal of the fence whose value is at `fence`.
    Signal { fence: usize, value: u64 },
    /// A present (asynchronous queues): frame `frame` of the title texture
    /// at `resource` (for the log), whose host copy is `shown`.
    Present {
        frame: u64,
        resource: usize,
        #[cfg(windows)]
        shown: Option<HostCopy>,
    },
    /// Page mappings (asynchronous queues), from the pool of `pool_pages`
    /// pages at `pool`.
    Map {
        batches: Batches,
        pool: u64,
        pool_pages: u32,
    },
}

/// Page mapping batches as read at the call: each batch's destination and
/// its ranges as (kind, start, count) (see [`PageRange`]).
type Batches = Vec<(u64, Vec<(u32, u32, u32)>)>;

/// A title texture's host copy, for a present run later.
#[cfg(windows)]
struct HostCopy(std::sync::Arc<crate::resource::HostTexture>);

// SAFETY: D3D12 resources are free-threaded, and the state is atomic; the
// held present only reads the copy and moves its state, as a present on
// the title's thread does.
#[cfg(windows)]
unsafe impl Send for HostCopy {}

/// The queues' held work, by queue object.
static QUEUES: Mutex<Schedule<Held>> = Mutex::new(Schedule::new());
/// Woken (under `QUEUES`) when a fence moves or work is held.
static MOVED: Condvar = Condvar::new();

fn queues() -> MutexGuard<'static, Schedule<Held>> {
    QUEUES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Submissions and signals held back since the frame-rate line last
/// reported them, presents or page mappings that ran ahead of held work
/// (the shortcut above), and submissions that waited for the asynchronous
/// worker's backlog.
static HELD: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Since the last call (the frame-rate line): submissions and signals
/// held back by waits, presents and page mappings that ran ahead, and
/// waits for the asynchronous worker's backlog.
#[cfg(windows)]
pub(crate) fn take_held() -> [u64; 3] {
    [0, 1, 2].map(|i| HELD[i].swap(0, Ordering::Relaxed))
}

/// Whether every queue holds what is submitted to it for the worker
/// (experimental, see the module): diagnostics builds with
/// `VICTORY_QUEUE_ASYNC=1`, while queue waits hold work back.
fn asynchronous() -> bool {
    #[cfg(test)]
    if ASYNC_FOR_TESTS.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = cfg!(feature = "diagnostics")
            && std::env::var("VICTORY_QUEUE_ASYNC").is_ok_and(|v| v.trim() == "1");
        if asked && !holding() {
            tracing::warn!("VICTORY_QUEUE_ASYNC=1 needs queue waits to hold work: not asynchronous");
            return false;
        }
        if asked {
            tracing::warn!(
                "VICTORY_QUEUE_ASYNC=1: submissions, signals, presents and page mappings run on the \
                 queue worker (experimental)"
            );
        }
        asked
    })
}

#[cfg(test)]
thread_local! {
    /// Asynchronous queues for the calls of one test's thread.
    static ASYNC_FOR_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Asynchronous queues for the calling test's thread (tests of other
/// modules).
#[cfg(test)]
pub(crate) fn asynchronous_for_tests(on: bool) {
    ASYNC_FOR_TESTS.set(on);
}

/// A queue's signal (tests of other modules).
#[cfg(test)]
pub(crate) fn signal_for_tests(queue: This, fence: This, value: u64) -> HRESULT {
    signal(queue, fence, value)
}

/// Whether queue waits hold work back (the default). With the
/// `diagnostics` feature, `VICTORY_QUEUE_WAITS=0` makes queues run what is
/// submitted at once, reached or not, as before, to compare.
fn holding() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let off = cfg!(feature = "diagnostics")
            && std::env::var("VICTORY_QUEUE_WAITS").is_ok_and(|v| v.trim() == "0");
        if off {
            tracing::warn!("VICTORY_QUEUE_WAITS=0: queue waits do not hold work back");
        }
        !off
    })
}

/// A fence moved to `value` (its value lives at `at`): held waits it
/// reaches are satisfied, and the worker runs what they held.
pub(crate) fn fence_moved(at: usize, value: u64) {
    let mut q = queues();
    if q.blocked().is_empty() {
        return;
    }
    q.fence_moved(at, value);
    MOVED.notify_all();
}

/// Holds `held` back on `queue` if anything holds the queue; else gives it
/// back to run now.
/// What a submission to a queue does.
enum Submit {
    /// Held back behind the queue's waits.
    Held,
    /// Run now by the submitting thread, the queue claimed until done.
    Now(Claim),
}

/// A queue claimed for work its submitter runs at once: until this is
/// dropped, waits and work other threads submit to the queue are held
/// behind it, so they cannot run ahead of it on the worker.
struct Claim(Option<usize>);

impl Drop for Claim {
    fn drop(&mut self) {
        if let Some(queue) = self.0 {
            queues().finished(queue);
            MOVED.notify_all();
        }
    }
}

/// Holds `held` back if the queue has waits not reached (or work before
/// it), else claims the queue for the caller to run it now.
fn hold(queue: usize, held: impl FnOnce() -> Held) -> Submit {
    if !holding() && !asynchronous() {
        return Submit::Now(Claim(None));
    }
    if asynchronous() {
        to_worker(queue, held());
        return Submit::Held;
    }
    let mut q = queues();
    if q.direct(queue) {
        return Submit::Now(Claim(Some(queue)));
    }
    // The queue holds something, so this is kept.
    let _ = q.submit(queue, held());
    drop(q);
    HELD[0].fetch_add(1, Ordering::Relaxed);
    start_worker();
    MOVED.notify_all();
    Submit::Held
}

/// The most operations the worker may hold that it can run on its own (not
/// behind a wait not reached) before a submitting thread waits for it:
/// several frames' submissions and signals, so that a title running ahead
/// of the GPU cannot grow the backlog, and the memory its host lists hold,
/// without bound.
const BACKLOG: usize = 512;

/// Asynchronous queues: keeps `item` for the worker, in the queue's order;
/// then waits while the worker has more than [`BACKLOG`] operations it can
/// run without any fence moving. Held work behind a wait not reached never
/// makes a thread wait (the thread may be what releases it), and the
/// worker runs the rest on its own, so the wait ends.
fn to_worker(queue: usize, item: Held) {
    let mut q = queues();
    q.defer(queue, item);
    HELD[0].fetch_add(1, Ordering::Relaxed);
    start_worker();
    MOVED.notify_all();
    if q.ready_len() > BACKLOG {
        HELD[2].fetch_add(1, Ordering::Relaxed);
        while q.ready_len() > BACKLOG {
            q = MOVED.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn start_worker() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("victory queue worker".into())
            .spawn(worker);
        if let Err(e) = spawned {
            tracing::error!(error = %e, "no queue worker: held queue work will not run");
        }
    });
}

/// When a held wait that has lasted is logged: at 5 s, then each time it
/// has doubled.
fn warning_due(warned: u32) -> Duration {
    Duration::from_secs(5).saturating_mul(1 << warned.min(20))
}

/// The queue worker: runs held work once nothing holds it back, in each
/// queue's order; polls held fences for stores nothing reports; logs waits
/// that last, with what could still satisfy them.
fn worker() {
    let mut since: Option<Instant> = None;
    let mut warned = 0;
    loop {
        let (ready, blocked) = {
            let mut q = queues();
            let ready = q.runnable();
            let blocked = if ready.is_empty() {
                q.blocked()
            } else {
                Vec::new()
            };
            if ready.is_empty() && blocked.is_empty() {
                since = None;
                warned = 0;
                drop(MOVED.wait(q).unwrap_or_else(|e| e.into_inner()));
                continue;
            }
            (ready, blocked)
        };
        if !ready.is_empty() {
            run(ready);
            continue;
        }
        for &(_, at, value) in &blocked {
            if let Some(now) = crate::fence::current(at)
                && now >= value
            {
                queues().fence_moved(at, now);
            }
        }
        let waited = since.get_or_insert_with(Instant::now).elapsed();
        if waited >= warning_due(warned) {
            warned += 1;
            let q = queues();
            let stuck = q.stuck(|h| match *h {
                Held::Signal { fence, value } => Some((fence, value)),
                #[cfg(windows)]
                Held::Execute(_) => None,
                Held::Present { .. } | Held::Map { .. } => None,
            });
            tracing::warn!(
                seconds = waited.as_secs(),
                held = q.held(),
                blocked = ?q.blocked(),
                only_held_work_signals = ?stuck,
                "queues are held back by waits not reached: (queue, fence, value); a wait only \
                 held work signals never ends unless a CPU thread signals its fence"
            );
        }
        let q = queues();
        let poll = Duration::from_millis(1);
        let woken = MOVED.wait_timeout(q, poll);
        drop(woken.unwrap_or_else(|e| e.into_inner()));
    }
}

/// Runs released work, then lets each queue it came from run on.
fn run(ready: Vec<(usize, Held)>) {
    let mut done: Vec<usize> = Vec::new();
    for (queue, held) in ready {
        match held {
            #[cfg(windows)]
            Held::Execute(lists) => {
                if let Some(host) = crate::host::host() {
                    let lists: Vec<&crate::host::HostList> = lists.iter().map(|l| &**l).collect();
                    host.execute(&lists);
                }
            }
            Held::Signal { fence, value } => {
                if !crate::fence::signal_at(fence, value) {
                    tracing::warn!(
                        fence = format_args!("{fence:#x}"),
                        value,
                        "a held signal of a fence the title released"
                    );
                }
            }
            Held::Present {
                frame,
                resource,
                #[cfg(windows)]
                shown,
            } => {
                #[cfg(windows)]
                show(frame, resource, shown.as_ref().map(|c| &*c.0));
                #[cfg(not(windows))]
                let _ = (frame, resource);
                #[cfg(test)]
                HELD_PRESENTS_RUN.fetch_add(1, Ordering::SeqCst);
            }
            Held::Map {
                batches,
                pool,
                pool_pages,
            } => map_pages(&batches, pool, pool_pages),
        }
        if !done.contains(&queue) {
            done.push(queue);
        }
    }
    let mut q = queues();
    for queue in done {
        q.finished(queue);
    }
    MOVED.notify_all();
}

/// Counts a present or page mapping that runs while its queue holds work
/// back (the shortcut above).
fn ran_ahead(queue: usize, what: &str) {
    if !queues().holds(queue) {
        return;
    }
    HELD[1].fetch_add(1, Ordering::Relaxed);
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
        tracing::warn!("{what} runs ahead of work its queue holds back for a wait");
    }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Runs the lists' host lists and waits for them (decision 0007), or holds
/// them back behind the queue's waits.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let shared = || {
            Held::Execute(
                lists
                    .iter()
                    .filter_map(|&l| crate::list::host_list_shared(l))
                    .collect(),
            )
        };
        let Submit::Now(claim) = hold(this as usize, shared) else {
            return;
        };
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        drop(claim);
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    let at = crate::fence::value_address(fence);
    let Submit::Now(claim) = hold(this as usize, || Held::Signal { fence: at, value }) else {
        return S_OK;
    };
    let signalled = crate::fence::signal(fence, value);
    drop(claim);
    signalled
}
/// Holds back what is submitted to the queue after it until the fence
/// reaches `value` (see the module).
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    if !holding() {
        return S_OK;
    }
    let at = crate::fence::value_address(fence);
    let now = crate::fence::current(at).unwrap_or(0);
    let mut q = queues();
    q.wait(this as usize, at, value, now);
    if q.holds(this as usize) {
        start_worker();
        MOVED.notify_all();
    }
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
        // A fence's value, written through its GPU address, wakes what
        // waits for it.
        crate::fence::written(at as usize, bytes);
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    if !asynchronous() {
        ran_ahead(this as usize, "a present");
    }
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    // The first plane's first resource, read at the call.
    let mut resource: This = std::ptr::null();
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
    {
        // SAFETY: the plane lists `count` resources.
        resource = unsafe { *plane.resources };
    }
    #[cfg(windows)]
    let shown = crate::resource::host_copy(resource).map(HostCopy);
    if asynchronous() {
        // In order with the queue's work: the worker presents.
        let held = Held::Present {
            frame,
            resource: resource as usize,
            #[cfg(windows)]
            shown,
        };
        to_worker(this as usize, held);
        return S_OK;
    }
    #[cfg(windows)]
    show(frame, resource as usize, shown.as_ref().map(|c| &*c.0));
    S_OK
}

/// The host part of a present: the frames listed for capture saved, and the
/// title texture's host copy (if it has one) shown in our window.
#[cfg(windows)]
fn show(frame: u64, resource: usize, copy: Option<&crate::resource::HostTexture>) {
    #[cfg(feature = "diagnostics")]
    if crate::capture::listed(frame) {
        crate::capture::dump_targets(frame);
    }
    let (Some(copy), Some(host)) = (copy, crate::host::host()) else {
        return;
    };
    const COPY_SOURCE: u32 = 0x800;
    // The state the copy is in, tracked as the copy source while the
    // present copies it, then back.
    let state = copy.state.swap(COPY_SOURCE, Ordering::Relaxed);
    let shown = host.present(&copy.texture, state, frame);
    copy.state.store(state, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    blinks(frame, resource as This, state, shown);
    #[cfg(not(feature = "diagnostics"))]
    let _ = (shown, resource);
}

/// Held presents the worker has run (tests).
#[cfg(test)]
static HELD_PRESENTS_RUN: AtomicU64 = AtomicU64::new(0);
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    if !asynchronous() {
        ran_ahead(this as usize, "a page mapping");
    }
    if crate::resource::page_mapper().is_none() {
        return E_INVALIDARG;
    }
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    // What the call names, read now: the worker may map it later.
    let batches: Batches = batches
        .into_iter()
        .flatten()
        .filter(|b| !b.ranges.is_null())
        .map(|b| {
            // SAFETY: each batch holds `count` ranges.
            let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
            (
                b.destination,
                ranges.iter().map(|r| (r.kind, r.start, r.count)).collect(),
            )
        })
        .collect();
    if asynchronous() {
        // In order with the queue's work: the worker maps.
        let held = Held::Map {
            batches,
            pool,
            pool_pages,
        };
        to_worker(this as usize, held);
        return S_OK;
    }
    map_pages(&batches, pool, pool_pages);
    S_OK
}

/// Maps the batches' ranges (see [`copy_page_mappings`]).
fn map_pages(batches: &Batches, pool: u64, pool_pages: u32) {
    let Some(map) = crate::resource::page_mapper() else {
        return;
    };
    let mut ok = true;
    for (destination, ranges) in batches {
        let mut at = *destination;
        for &(kind, start, count) in ranges {
            let len = u64::from(count) * PAGE;
            let inside = |pages: u32| start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(start) * PAGE;
            ok &= match kind {
                0 if inside(count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    fn new_queue() -> This {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 2,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        q
    }

    fn new_fence(initial: u64) -> This {
        let (null, mut f) = (std::ptr::null(), std::ptr::null());
        let iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(crate::fence::create(null, initial, 0, iid, &mut f), S_OK);
        f
    }

    fn value(f: This) -> u64 {
        slot::<extern "system" fn(This) -> u64>(f, 0x40)(f)
    }

    /// Waits up to 5 s for `f` to reach `v` (the queue worker runs held
    /// work on its own thread).
    fn reaches(f: This, v: u64) -> bool {
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(5) {
            if value(f) >= v {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// A queue's signals behind a wait not reached are held, in order,
    /// until a CPU signal reaches it; the submitting thread does not wait.
    /// A wait already reached holds nothing; a null fence is refused.
    #[test]
    fn signals_behind_a_wait_not_reached_wait_for_it() {
        let q = new_queue();
        let (f, g) = (new_fence(3), new_fence(0));
        assert_eq!(wait(q, f, 3), S_OK, "reached already");
        assert_eq!(signal(q, g, 1), S_OK);
        assert_eq!(value(g), 1, "nothing held");
        assert_eq!(wait(q, f, 5), S_OK);
        assert_eq!(signal(q, g, 2), S_OK);
        assert_eq!(signal(q, g, 3), S_OK);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(value(g), 1, "held behind the wait for 5");
        assert_eq!(crate::fence::signal(f, 5), S_OK);
        assert!(reaches(g, 3), "released");
        assert_eq!(value(g), 3, "in order: 2, then 3");
        // Nothing holds the queue any more: signals run at once again.
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(signal(q, g, 4), S_OK);
        assert!(reaches(g, 4));
        assert_eq!(wait(q, std::ptr::null(), 1), E_INVALIDARG);
    }

    /// Queues chain through fences: the second waits for what the first,
    /// held, signals once released.
    #[test]
    fn held_queues_chain_through_fences() {
        let (q1, q2) = (new_queue(), new_queue());
        let (f, g, h) = (new_fence(0), new_fence(0), new_fence(0));
        assert_eq!(wait(q1, f, 1), S_OK);
        assert_eq!(signal(q1, g, 1), S_OK);
        assert_eq!(wait(q2, g, 1), S_OK);
        assert_eq!(signal(q2, h, 1), S_OK);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!((value(g), value(h)), (0, 0));
        assert_eq!(crate::fence::signal(f, 1), S_OK);
        assert!(reaches(h, 1), "the chain ran");
        assert_eq!(value(g), 1);
    }

    /// A store into a fence's memory that nothing reports (neither a signal
    /// nor a bottom-of-pipe write) still releases the queue: the worker
    /// polls held fences.
    #[test]
    fn a_store_nothing_reports_releases_a_held_queue() {
        let q = new_queue();
        let (f, g) = (new_fence(0), new_fence(0));
        assert_eq!(wait(q, f, 2), S_OK);
        assert_eq!(signal(q, g, 1), S_OK);
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as *mut u64;
        // SAFETY: the fence's value, as titles store it.
        unsafe { at.write_volatile(2) };
        assert!(reaches(g, 1));
    }

    /// Asynchronous queues, the backlog bound: thousands of signals on a
    /// queue nothing holds all run, in order, whether or not a submitting
    /// thread had to wait for the worker; behind a wait not reached they
    /// are held without the submitting thread ever waiting (it is what
    /// signals the fence the wait needs), then run once it does.
    #[test]
    fn asynchronous_backlogs_are_bounded_without_waiting_on_held_work() {
        ASYNC_FOR_TESTS.set(true);
        let q = new_queue();
        let (f, g) = (new_fence(0), new_fence(0));
        let n = 3 * BACKLOG as u64;
        for v in 1..=n {
            assert_eq!(signal(q, g, v), S_OK);
        }
        assert!(reaches(g, n), "all ran");
        assert_eq!(wait(q, f, 1), S_OK);
        for v in n + 1..=2 * n {
            assert_eq!(signal(q, g, v), S_OK);
        }
        assert_eq!(value(g), n, "held behind the wait, the thread never stopped");
        assert_eq!(crate::fence::signal(f, 1), S_OK);
        assert!(reaches(g, 2 * n), "released in order");
        ASYNC_FOR_TESTS.set(false);
    }

    /// Asynchronous queues (experimental): the worker runs every signal,
    /// even on a queue nothing holds, and a present is held in order with
    /// the queue's work: behind a wait not reached, neither the present nor
    /// the signal after it runs; once the wait is reached, both do, the
    /// present first.
    #[test]
    fn asynchronous_queues_run_signals_and_presents_on_the_worker_in_order() {
        ASYNC_FOR_TESTS.set(true);
        let q = new_queue();
        let (f, g) = (new_fence(0), new_fence(0));
        assert_eq!(signal(q, g, 1), S_OK);
        assert!(reaches(g, 1), "the worker signals");
        assert_eq!(wait(q, f, 1), S_OK);
        let presents = HELD_PRESENTS_RUN.load(Ordering::SeqCst);
        assert_eq!(present(q, 0, std::ptr::null(), std::ptr::null()), S_OK);
        assert_eq!(signal(q, g, 2), S_OK);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(value(g), 1, "held behind the wait");
        assert_eq!(
            HELD_PRESENTS_RUN.load(Ordering::SeqCst),
            presents,
            "the present is held too"
        );
        assert_eq!(crate::fence::signal(f, 1), S_OK);
        assert!(reaches(g, 2), "released");
        assert!(
            HELD_PRESENTS_RUN.load(Ordering::SeqCst) > presents,
            "the present ran before the signal behind it"
        );
        ASYNC_FOR_TESTS.set(false);
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
