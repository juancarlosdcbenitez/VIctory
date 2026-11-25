//! Fences: a 64-bit value and what waits for it. The value moves by
//! signals (the title's, and our queues', which signal once a submission
//! has run) and by bottom-of-pipe writes from title command lists to the
//! fence's GPU address ([`written`]).
//!
//! Waits follow D3D12's rules (ID3D12Fence::SetEventOnCompletion and
//! "Multi-engine synchronization"): a wait without an event does not return
//! until the value is reached, however long that takes; a signal that
//! reaches a waited-for value satisfies the wait even if a lower value
//! replaces it before the waiting thread runs; a wait that starts after a
//! fence was set lower compares against the lower value.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    waiters: Mutex<Waiters>,
    /// Woken (under `waiters`) when the value moves, for threads blocked in
    /// a wait without an event.
    moved: Condvar,
}

/// What waits for a fence's value.
#[derive(Default)]
struct Waiters {
    /// Events to set once the value reaches theirs (handles as integers).
    events: Vec<(u64, usize)>,
    /// Threads blocked in a wait without an event: the value each waits
    /// for, and a flag set once a signal or write reached it. The flag
    /// keeps a wait satisfied by a value that was replaced by a lower one
    /// before the thread looked.
    blocked: Vec<(u64, Arc<AtomicBool>)>,
}

impl Waiters {
    /// Sets the events and releases the threads that `value` satisfies.
    fn reach(&mut self, value: u64) {
        self.events.retain(|&(wanted, event)| {
            let reached = wanted <= value;
            if reached {
                set_event(event);
            }
            !reached
        });
        self.blocked.retain(|(wanted, done)| {
            let reached = *wanted <= value;
            if reached {
                done.store(true, Ordering::Release);
            }
            !reached
        });
    }
}

/// Live fences by the address of their value (their GPU address), for
/// [`written`].
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

impl Drop for Fence {
    fn drop(&mut self) {
        live().remove(&((&raw const self.value) as usize));
    }
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

impl Fence {
    fn waiters(&self) -> MutexGuard<'_, Waiters> {
        self.waiters.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Where a fence's value is (its GPU address), by which [`current`] finds
/// it while it lives.
pub(crate) fn value_address(this: This) -> usize {
    (&raw const fence(this).value) as usize
}

/// The value of the live fence whose value is at `at`; `None` once that
/// fence has been released.
pub(crate) fn current(at: usize) -> Option<u64> {
    let live = live();
    let &object = live.get(&at)?;
    // SAFETY: as in `written`: a fence in `LIVE` is alive.
    let fence = unsafe { &*(object as *const Fence) };
    Some(fence.value.load(Ordering::Acquire))
}

/// After a command list's bottom-of-pipe write stored `bytes` at `at`: a
/// fence whose value it touched has moved, as if signalled, so its events
/// are set and its blocked waits rechecked. Titles signal fences this way
/// through the fence's GPU address, and only `signal` used to wake waiters.
pub(crate) fn written(at: usize, bytes: usize) {
    const VALUE: usize = size_of::<u64>();
    let mut moved = Vec::new();
    {
        let live = live();
        // Fences whose value overlaps at..at + bytes.
        for (&value_at, &object) in
            live.range(at.saturating_sub(VALUE - 1)..at.saturating_add(bytes))
        {
            // SAFETY: a fence in `LIVE` is alive: it leaves `LIVE` (under
            // the lock held here) before it is freed.
            let fence = unsafe { &*(object as *const Fence) };
            let mut waiters = fence.waiters();
            let value = fence.value.load(Ordering::Acquire);
            waiters.reach(value);
            fence.moved.notify_all();
            moved.push((value_at, value));
        }
    }
    // Queues held back by waits on them, after the fences' locks.
    for (value_at, value) in moved {
        crate::queue::fence_moved(value_at, value);
    }
}

/// Signals the live fence whose value is at `at` (a signal a held queue
/// runs later); false if the title released the fence meanwhile.
pub(crate) fn signal_at(at: usize, value: u64) -> bool {
    {
        let live = live();
        let Some(&object) = live.get(&at) else {
            return false;
        };
        // SAFETY: a fence in `LIVE` is alive while the lock is held.
        set(unsafe { &*(object as *const Fence) }, value);
    }
    crate::queue::fence_moved(at, value);
    true
}

/// Sets `fence` to `value` and wakes what its value releases.
fn set(fence: &Fence, value: u64) {
    let mut waiters = fence.waiters();
    fence.value.store(value, Ordering::Release);
    waiters.reach(value);
    fence.moved.notify_all();
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Waiters::default()),
        moved: Condvar::new(),
    });
    // Unregistered by `Drop`, also when `hand_out` drops it.
    live().insert(
        (&raw const fence.value) as usize,
        (&raw const *fence) as usize,
    );
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters();
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // No event: D3D12 does not return until the value is reached, and
        // neither do we. A value nothing reaches is a hang, logged as one.
        block_until(fence, waiters, value, None);
        return S_OK;
    }
    waiters.events.push((value, event as usize));
    S_OK
}

/// When a wait without an event that has not ended is logged: at 5 s, then
/// each time the wait has doubled.
fn warning_due(warned: u32) -> Duration {
    Duration::from_secs(5).saturating_mul(1 << warned.min(20))
}

/// Blocks (with `waiters` locked on entry) until `fence` reaches `value`;
/// false only if `give_up` (tests only) passes first. Woken by signals and
/// bottom-of-pipe writes, and polling too, for writes to the value that
/// reach neither (a title's own stores to it, say). A wait that lasts is
/// logged at [`warning_due`].
fn block_until(
    fence: &Fence,
    mut waiters: MutexGuard<'_, Waiters>,
    value: u64,
    give_up: Option<Duration>,
) -> bool {
    const POLL: Duration = Duration::from_millis(1);
    let done = Arc::new(AtomicBool::new(false));
    waiters.blocked.push((value, done.clone()));
    let started = Instant::now();
    let mut warned = 0;
    let ended = loop {
        if done.load(Ordering::Acquire) || fence.value.load(Ordering::Acquire) >= value {
            break true;
        }
        let waited = started.elapsed();
        if give_up.is_some_and(|limit| waited >= limit) {
            break false;
        }
        if waited >= warning_due(warned) {
            warned += 1;
            tracing::warn!(
                fence = format_args!("{:#x}", (&raw const fence.value) as usize),
                waiting_for = value,
                value = fence.value.load(Ordering::Acquire),
                seconds = waited.as_secs(),
                "a blocking fence wait has not been reached: a fence nothing signals, or \
                 the title's threads wait on each other; still waiting"
            );
        }
        waiters = fence
            .moved
            .wait_timeout(waiters, POLL)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    };
    // Reached by polling or given up: no longer waiting.
    waiters.blocked.retain(|(_, d)| !Arc::ptr_eq(d, &done));
    if warned > 0 && ended {
        tracing::warn!(
            waiting_for = value,
            seconds = started.elapsed().as_secs(),
            "a long blocking fence wait ended"
        );
    }
    ended
}

pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    set(fence, value);
    // Queues held back by waits on it, after the fence's lock.
    crate::queue::fence_moved((&raw const fence.value) as usize, value);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // A bottom-of-pipe write to that address sets the events it
        // reaches, as a signal does.
        assert_eq!(wait(f, 8, event), S_OK);
        crate::queue::Written::Value(8).write(at as *mut u8, 8);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    /// A wait without an event blocks until another thread's signal
    /// reaches its value (D3D12's contract); it used to fail at once.
    #[test]
    fn a_wait_without_an_event_blocks_until_a_signal_reaches_it() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        let address = f as usize;
        let signaller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            assert_eq!(signal(address as This, 3), S_OK);
            std::thread::sleep(Duration::from_millis(30));
            assert_eq!(signal(address as This, 7), S_OK);
        });
        let started = Instant::now();
        assert_eq!(wait(f, 7, std::ptr::null_mut()), S_OK);
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "returned early"
        );
        assert_eq!(fence(f).value.load(Ordering::Acquire), 7);
        signaller.join().unwrap();
        // Reached already: no wait.
        assert_eq!(wait(f, 7, std::ptr::null_mut()), S_OK);
    }

    /// A value stored straight into the fence's memory, by nothing we see
    /// (neither a signal nor a bottom-of-pipe write), still ends the wait,
    /// by polling. (The limit is the test's own, so a regression fails it
    /// rather than hanging it; waits themselves have none.)
    #[test]
    fn blocking_waits_see_stores_nothing_reports() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let address = f as usize;
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            fence(address as This).value.store(9, Ordering::Release);
        });
        let fence = fence(f);
        assert!(block_until(
            fence,
            fence.waiters(),
            9,
            Some(Duration::from_secs(5))
        ));
        writer.join().unwrap();
        assert!(fence.waiters().blocked.is_empty(), "the ended wait left");
        let started = Instant::now();
        assert!(!block_until(
            fence,
            fence.waiters(),
            10,
            Some(Duration::from_millis(40))
        ));
        assert!(started.elapsed() >= Duration::from_millis(40));
        assert!(
            fence.waiters().blocked.is_empty(),
            "the abandoned wait left"
        );
    }

    /// Two signals race, the second lower: a thread waiting for the first's
    /// value is released all the same ("the wait will be satisfied
    /// regardless of which value remains afterwards"). It used to look at
    /// the value only, find the lower one and wait on.
    #[test]
    fn a_wait_reached_then_rewound_is_satisfied() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let address = f as usize;
        let waiter = std::thread::spawn(move || {
            let fence = fence(address as This);
            block_until(fence, fence.waiters(), 10, Some(Duration::from_secs(5)))
        });
        let fence = fence(f);
        while fence.waiters().blocked.is_empty() {
            std::thread::yield_now();
        }
        // Both signals land before the waiting thread can look.
        {
            let mut waiters = fence.waiters();
            fence.value.store(10, Ordering::Release);
            waiters.reach(10);
            fence.value.store(4, Ordering::Release);
            waiters.reach(4);
            fence.moved.notify_all();
        }
        assert!(waiter.join().unwrap(), "the reached wait was not released");
        assert_eq!(fence.value.load(Ordering::Acquire), 4);
        // A wait that starts now compares against the lower value.
        assert!(!block_until(
            fence,
            fence.waiters(),
            10,
            Some(Duration::from_millis(20))
        ));
    }

    /// A command list's bottom-of-pipe write to a fence's GPU address moves
    /// it as a signal does: events it reaches are set (taken off the list)
    /// and blocked threads released; writes elsewhere, or of lower values,
    /// leave them waiting.
    #[test]
    fn bottom_of_pipe_writes_to_a_fence_reach_its_waiters() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as usize;
        // Off Windows events are not set, but they leave the list as here.
        #[cfg(windows)]
        // SAFETY: plain manual-reset event, never closed (a test).
        let event = unsafe {
            windows_sys::Win32::System::Threading::CreateEventW(
                std::ptr::null(),
                1,
                0,
                std::ptr::null(),
            )
        };
        #[cfg(not(windows))]
        let event = std::ptr::dangling_mut::<c_void>();
        assert_eq!(wait(f, 6, event), S_OK);
        let fence = fence(f);
        let mut beside = [0u64; 2];
        crate::queue::Written::Value(6).write(beside.as_mut_ptr().cast(), 8);
        crate::queue::Written::Value(5).write(at as *mut u8, 8);
        assert_eq!(fence.waiters().events.len(), 1, "set too early");
        let address = f as usize;
        let waiter = std::thread::spawn(move || {
            let fence = super::fence(address as This);
            block_until(fence, fence.waiters(), 7, Some(Duration::from_secs(5)))
        });
        while fence.waiters().blocked.is_empty() {
            std::thread::yield_now();
        }
        // The low half of the value, as a 32-bit write would set it.
        crate::queue::Written::Value(7).write(at as *mut u8, 4);
        assert!(waiter.join().unwrap());
        assert!(fence.waiters().events.is_empty(), "the event was not set");
        assert_eq!(fence.value.load(Ordering::Acquire), 7);
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::WaitForSingleObject;
            // SAFETY: a valid event.
            let state = unsafe { WaitForSingleObject(event, 0) };
            assert_eq!(state, windows_sys::Win32::Foundation::WAIT_OBJECT_0);
        }
    }

    /// A released fence leaves the registry, so later writes to the memory
    /// it had do not reach it.
    #[test]
    fn released_fences_leave_the_registry() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as usize;
        assert!(live().contains_key(&at));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
        assert!(!live().contains_key(&at));
        // A creation that is refused (an unknown interface) frees the fence
        // at once, and unregisters it too.
        let before = live().len();
        let unknown = guid(1, 2, 3, [4; 8]);
        let mut g = std::ptr::null();
        assert_ne!(create(std::ptr::null(), 0, 0, &unknown, &mut g), S_OK);
        assert!(live().len() <= before);
    }

    /// Waits that last are logged at 5, 10, 20, 40 s and so on, never
    /// ended.
    #[test]
    fn long_waits_are_logged_as_they_double() {
        let due: Vec<u64> = (0..5).map(|n| warning_due(n).as_secs()).collect();
        assert_eq!(due, [5, 10, 20, 40, 80]);
        assert!(warning_due(u32::MAX) >= Duration::from_secs(5 << 20));
    }
}
