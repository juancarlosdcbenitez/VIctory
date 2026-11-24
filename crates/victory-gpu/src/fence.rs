//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
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
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    /// Woken (under `waiters`) when a signal moves the value, for threads
    /// blocked in a wait without an event.
    moved: Condvar,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
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
        waiters: Mutex::new(Vec::new()),
        moved: Condvar::new(),
    });
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
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // No event: block until the value is reached (another thread's
        // submission or signal moves it).
        // shortcut: a value nothing reaches within BLOCKING_WAIT fails the
        // wait, as before blocking waits were implemented, rather than
        // hang the thread forever.
        if block_until(fence, waiters, value, BLOCKING_WAIT) {
            return S_OK;
        }
        tracing::warn!(value, "a blocking fence wait was not reached in time");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}

/// How long a wait without an event blocks for a value nothing reaches.
const BLOCKING_WAIT: Duration = Duration::from_secs(5);

/// Blocks (with `waiters` locked on entry) until `fence` reaches `value`;
/// false if `limit` passes first. Woken by signals, and polling too: titles
/// also write fence values from GPU commands, straight into memory.
fn block_until(
    fence: &Fence,
    mut waiters: MutexGuard<'_, Vec<(u64, usize)>>,
    value: u64,
    limit: Duration,
) -> bool {
    const POLL: Duration = Duration::from_millis(1);
    let started = Instant::now();
    loop {
        if fence.value.load(Ordering::Acquire) >= value {
            return true;
        }
        let left = limit.saturating_sub(started.elapsed());
        if left.is_zero() {
            return false;
        }
        waiters = fence
            .moved
            .wait_timeout(waiters, left.min(POLL))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    fence.moved.notify_all();
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
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
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

    /// A value written straight into the fence's memory (as GPU commands
    /// do), without a signal, still ends the wait; one never reached ends
    /// it at the limit, as a failure.
    #[test]
    fn blocking_waits_see_direct_writes_and_give_up_at_the_limit() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let address = f as usize;
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            fence(address as This).value.store(9, Ordering::Release);
        });
        let fence = fence(f);
        let guard = fence.waiters.lock().unwrap();
        assert!(block_until(fence, guard, 9, Duration::from_secs(5)));
        writer.join().unwrap();
        let guard = fence.waiters.lock().unwrap();
        let started = Instant::now();
        assert!(!block_until(fence, guard, 10, Duration::from_millis(40)));
        assert!(started.elapsed() >= Duration::from_millis(40));
    }
}
