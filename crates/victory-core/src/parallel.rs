//! Work spread over a few threads where the result cannot depend on how it
//! is spread: a BLAKE3 hash equal bit for bit to [`blake3::hash`], and
//! independent items each handled once.
//!
//! The threads are a pool started once and parked between calls, so a call
//! costs a wake-up, not thread creation (which took 58 to 217 microseconds a
//! call for 2 to 8 threads on the machine measured, PERF-5). One call runs
//! on the pool at a time; a call made while another runs (from another
//! thread, or nested inside the work) works on its caller's thread alone.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use blake3::hazmat::{
    ChainingValue, HasherExt, Mode, left_subtree_len, merge_subtrees_non_root, merge_subtrees_root,
};

/// Inputs shorter than this are hashed on the calling thread.
pub const PARALLEL_HASH_MIN: usize = 2 << 20;

/// The smallest subtree a thread hashes: enough chunks for BLAKE3's own
/// SIMD parallelism inside it.
const LEAF_MIN: usize = 1 << 20;

/// The most threads parallel work uses.
const MAX_THREADS: usize = 8;

/// How many threads parallel work uses: the machine's, at most 8. In
/// diagnostics builds `VICTORY_TEXTURE_THREADS=<n>` sets it (1 to 8; with 1
/// everything runs on the calling thread and no pool is started), so a run
/// can compare frame times with the title's own threads less contended
/// (PERF-5): texture checks are the only parallel work.
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let machine = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let asked = if cfg!(feature = "diagnostics") {
            std::env::var("VICTORY_TEXTURE_THREADS").ok()
        } else {
            None
        };
        let n = chosen_threads(machine, asked.as_deref());
        if let Some(asked) = asked {
            tracing::warn!(
                "VICTORY_TEXTURE_THREADS={asked}: texture checks use up to {n} thread(s) \
                 (the machine has {machine})"
            );
        }
        n
    })
}

/// The thread count on a machine with `machine` threads: `asked` if it is
/// a number (kept within 1 to [`MAX_THREADS`]), else the machine's, at most
/// [`MAX_THREADS`].
fn chosen_threads(machine: usize, asked: Option<&str>) -> usize {
    match asked.and_then(|a| a.trim().parse::<usize>().ok()) {
        Some(n) => n.clamp(1, MAX_THREADS),
        None => machine.clamp(1, MAX_THREADS),
    }
}

/// The BLAKE3 hash of `bytes`, equal to `blake3::hash(bytes)`, computed on
/// up to `threads` threads for inputs of [`PARALLEL_HASH_MIN`] bytes and
/// more. The input is cut where BLAKE3's own tree cuts it (a subtree's left
/// part takes the largest power-of-two number of chunks below the whole)
/// into subtrees of about `len / threads`, which are hashed side by side
/// and merged in the tree's order, so the result cannot depend on how many
/// threads ran.
pub fn blake3(bytes: &[u8], threads: usize) -> blake3::Hash {
    if threads < 2 || bytes.len() < PARALLEL_HASH_MIN.max(2 * LEAF_MIN) {
        return blake3::hash(bytes);
    }
    let target = (bytes.len() / threads).max(LEAF_MIN);
    let mut leaves = Vec::new();
    let split = left_subtree_len(bytes.len() as u64) as usize;
    subtrees(0, split, target, &mut leaves);
    subtrees(split, bytes.len() - split, target, &mut leaves);
    let cvs: Vec<OnceLock<ChainingValue>> = leaves.iter().map(|_| OnceLock::new()).collect();
    let indices: Vec<usize> = (0..leaves.len()).collect();
    for_each(&indices, threads, |&i| {
        let (at, len) = leaves[i];
        let cv = blake3::Hasher::new()
            .set_input_offset(at as u64)
            .update(&bytes[at..at + len])
            .finalize_non_root();
        let _ = cvs[i].set(cv);
    });
    let mut next = cvs
        .iter()
        .map(|cv| *cv.get().expect("every subtree hashed"));
    let left = merged(split, target, &mut next);
    let right = merged(bytes.len() - split, target, &mut next);
    merge_subtrees_root(&left, &right, Mode::Hash)
}

/// The subtrees the subtree of `len` bytes at `at` is hashed as: itself if
/// it is no bigger than `target` (or a single chunk), else those of its two
/// halves, in input order.
fn subtrees(at: usize, len: usize, target: usize, out: &mut Vec<(usize, usize)>) {
    if len <= target || len <= blake3::CHUNK_LEN {
        out.push((at, len));
        return;
    }
    let split = left_subtree_len(len as u64) as usize;
    subtrees(at, split, target, out);
    subtrees(at + split, len - split, target, out);
}

/// The chaining value of a subtree of `len` bytes, from the chaining values
/// of the subtrees [`subtrees`] cut it into, taken from `next` in order.
fn merged(
    len: usize,
    target: usize,
    next: &mut impl Iterator<Item = ChainingValue>,
) -> ChainingValue {
    if len <= target || len <= blake3::CHUNK_LEN {
        return next.next().expect("a chaining value per subtree");
    }
    let split = left_subtree_len(len as u64) as usize;
    let left = merged(split, target, next);
    let right = merged(len - split, target, next);
    merge_subtrees_non_root(&left, &right, Mode::Hash)
}

/// Calls `f` once for each of `items`, on up to `threads` threads (the
/// caller's among them), each taking the next item not yet taken. The order
/// the items are handled in is not defined, so `f` must not depend on it. A
/// panic in `f` is raised again here, once every thread has stopped.
pub fn for_each<T: Sync>(items: &[T], threads: usize, f: impl Fn(&T) + Sync) {
    let workers = threads.min(items.len()).min(pool_workers() + 1);
    if workers < 2 {
        items.iter().for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || {
        while let Some(item) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
            f(item);
        }
    };
    run(workers - 1, &work);
}

/// The pool's workers: idle ones wait for a job, a caller posts one, works
/// on it too, then withdraws it and waits until no worker still runs it.
struct Pool {
    state: Mutex<State>,
    /// Workers wait here for a job.
    posted: Condvar,
    /// A caller waits here for its job's workers to finish.
    finished: Condvar,
}

#[derive(Default)]
struct State {
    /// The job posted, if one is.
    job: Option<Job>,
    /// How many more workers may join it.
    seats: usize,
    /// How many workers are running it.
    running: usize,
    /// Whether one of them panicked.
    panicked: bool,
}

/// A caller's work, its lifetime erased: [`run`] withdraws it and waits for
/// every worker that took it before the borrow ends.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn() + Sync + 'static));

// SAFETY: the work is `Sync`, and `run` keeps it alive while any worker can
// reach it.
unsafe impl Send for Job {}

/// How many workers the pool has: the threads parallel work uses, less the
/// caller's own.
fn pool_workers() -> usize {
    threads() - 1
}

/// The pool, its workers started on first use.
fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State::default()),
            posted: Condvar::new(),
            finished: Condvar::new(),
        }));
        for k in 0..pool_workers() {
            let spawned = std::thread::Builder::new()
                .name(format!("victory parallel {k}"))
                .spawn(move || worker(pool));
            if let Err(e) = spawned {
                tracing::warn!("parallel worker {k} not started: {e}");
            }
        }
        pool
    })
}

fn worker(pool: &'static Pool) {
    let lock = |m: &'static Mutex<State>| m.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let mut state = lock(&pool.state);
        let job = loop {
            match state.job {
                Some(job) if state.seats > 0 => {
                    state.seats -= 1;
                    state.running += 1;
                    break job;
                }
                _ => state = pool.posted.wait(state).unwrap_or_else(|e| e.into_inner()),
            }
        };
        drop(state);
        // SAFETY: the caller withdraws the job and waits until `running` is
        // back to zero before the work it points to goes away.
        let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.0)() }));
        let mut state = lock(&pool.state);
        state.panicked |= outcome.is_err();
        state.running -= 1;
        if state.running == 0 {
            pool.finished.notify_all();
        }
    }
}

/// Runs `work` on the caller's thread and on up to `helpers` pool workers,
/// returning once all have finished with it. If the pool is busy with
/// another call, the caller runs it alone.
fn run(helpers: usize, work: &(dyn Fn() + Sync)) {
    let pool = pool();
    let lock = || pool.state.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut state = lock();
        if state.job.is_some() || state.running > 0 {
            drop(state);
            work();
            return;
        }
        // SAFETY: only the lifetime is erased; see `Job`.
        let job: *const (dyn Fn() + Sync + 'static) = unsafe { std::mem::transmute(work) };
        state.job = Some(Job(job));
        state.seats = helpers;
        state.panicked = false;
    }
    pool.posted.notify_all();
    let mine = catch_unwind(AssertUnwindSafe(work));
    let mut state = lock();
    state.job = None;
    state.seats = 0;
    while state.running > 0 {
        state = pool.finished.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    let panicked = std::mem::take(&mut state.panicked);
    drop(state);
    if let Err(payload) = mine {
        resume_unwind(payload);
    }
    assert!(!panicked, "a parallel worker panicked");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not all zeros: a split at the wrong place would show.
    fn input(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B1).to_le_bytes()[2])
            .collect()
    }

    #[test]
    fn split_hashes_equal_blake3() {
        const CHUNK: usize = blake3::CHUNK_LEN;
        let lens = [
            0,
            1,
            CHUNK,
            PARALLEL_HASH_MIN - 1,
            PARALLEL_HASH_MIN,
            PARALLEL_HASH_MIN + 1,
            PARALLEL_HASH_MIN + CHUNK,
            3 * LEAF_MIN - 1,
            4 * LEAF_MIN,
            4 * LEAF_MIN + 1,
            5 * LEAF_MIN + 3 * CHUNK + 7,
            8 * LEAF_MIN - CHUNK,
            11 * LEAF_MIN + 12_345,
        ];
        let max = input(*lens.iter().max().unwrap());
        for len in lens {
            let bytes = &max[..len];
            let want = blake3::hash(bytes);
            for threads in [1, 2, 3, 4, 5, 8] {
                assert_eq!(
                    blake3(bytes, threads),
                    want,
                    "{len} bytes, {threads} threads"
                );
            }
        }
    }

    #[test]
    fn each_item_is_handled_once() {
        for (count, threads) in [(0, 4), (1, 4), (3, 1), (7, 3), (100, 8), (5, 64)] {
            let items: Vec<usize> = (0..count).collect();
            let seen: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            for_each(&items, threads, |&i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                seen.iter().all(|n| n.load(Ordering::Relaxed) == 1),
                "{count} items on {threads} threads"
            );
        }
    }

    #[test]
    fn threads_are_bounded() {
        assert!((1..=MAX_THREADS).contains(&threads()));
    }

    #[test]
    fn a_thread_count_asked_for_is_kept_within_bounds() {
        assert_eq!(chosen_threads(16, None), 8);
        assert_eq!(chosen_threads(4, None), 4);
        assert_eq!(chosen_threads(1, None), 1);
        assert_eq!(chosen_threads(16, Some("1")), 1);
        assert_eq!(chosen_threads(16, Some(" 3 ")), 3);
        assert_eq!(chosen_threads(4, Some("6")), 6);
        assert_eq!(chosen_threads(16, Some("0")), 1);
        assert_eq!(chosen_threads(16, Some("64")), 8);
        assert_eq!(chosen_threads(6, Some("many")), 6);
        assert_eq!(chosen_threads(6, Some("")), 6);
    }

    /// Callers on several threads at once: one has the pool, the others
    /// work alone; each handles each of its items once.
    #[test]
    fn concurrent_callers_each_handle_their_items_once() {
        std::thread::scope(|s| {
            for caller in 0..6 {
                s.spawn(move || {
                    for round in 0..20 {
                        let items: Vec<usize> = (0..37 + caller + round).collect();
                        let seen: Vec<AtomicUsize> =
                            items.iter().map(|_| AtomicUsize::new(0)).collect();
                        for_each(&items, 8, |&i| {
                            seen[i].fetch_add(1, Ordering::Relaxed);
                        });
                        assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) == 1));
                    }
                });
            }
        });
    }

    /// A call inside the work runs on its caller's thread and finishes.
    #[test]
    fn nested_calls_finish() {
        let total = AtomicUsize::new(0);
        let outer: Vec<usize> = (0..8).collect();
        for_each(&outer, 8, |_| {
            let inner: Vec<usize> = (0..5).collect();
            for_each(&inner, 8, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 40);
    }

    /// A panic in the work reaches the caller once every thread stopped,
    /// and the pool still works afterwards.
    #[test]
    fn a_panic_reaches_the_caller_and_the_pool_survives() {
        let items: Vec<usize> = (0..64).collect();
        let caught = std::panic::catch_unwind(|| {
            for_each(&items, 8, |&i| {
                assert_ne!(i, 41, "item 41");
            });
        });
        assert!(caught.is_err());
        let seen = AtomicUsize::new(0);
        for_each(&items, 8, |_| {
            seen.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(seen.load(Ordering::Relaxed), 64);
    }

    /// CPU throughput of hashing, one thread against this module (not a
    /// frame rate: what a submission's texture checks could save on this
    /// machine). Run with `cargo test --release -p victory-core --
    /// --ignored --nocapture hashing_throughput`.
    #[test]
    #[ignore = "benchmark"]
    fn hashing_throughput() {
        // Seconds a call of `f` takes: repeated for at least 50 ms (three
        // times at least), after a call to warm up.
        let time = |f: &dyn Fn()| {
            f();
            let (mut reps, started) = (0u32, std::time::Instant::now());
            while reps < 3 || started.elapsed() < std::time::Duration::from_millis(50) {
                f();
                reps += 1;
            }
            started.elapsed().as_secs_f64() / f64::from(reps)
        };
        let threads = threads();
        println!("hashing_throughput: {threads} threads");
        // What a call costs when the work is nothing: the pool's wake-up.
        let items: Vec<u64> = (0..8).collect();
        for helpers in [2, 4, 8] {
            let reps = 2000;
            for_each(&items, helpers, |x| {
                std::hint::black_box(x);
            });
            let started = std::time::Instant::now();
            for _ in 0..reps {
                for_each(&items, helpers, |x| {
                    std::hint::black_box(x);
                });
            }
            let us = started.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
            println!("a call over 8 empty items on up to {helpers} threads: {us:.1} us");
        }
        let big = input(64 << 20);
        // The thread counts to compare, up to the machine's.
        let counts: Vec<usize> = [2, 4, 8].into_iter().filter(|&n| n <= threads).collect();
        for mib in [1, 4, 16, 64] {
            let bytes = &big[..mib << 20];
            let one = time(&|| {
                std::hint::black_box(blake3::hash(std::hint::black_box(bytes)));
            });
            let mut line = format!(
                "one input of {mib} MiB: {:.2} GB/s on one thread",
                bytes.len() as f64 / one / 1e9
            );
            for &n in &counts {
                let split = time(&|| {
                    std::hint::black_box(super::blake3(std::hint::black_box(bytes), n));
                });
                line += &format!(
                    ", {:.2} split on {n} ({:.1}x)",
                    bytes.len() as f64 / split / 1e9,
                    one / split
                );
            }
            println!("{line}");
        }
        // A submission's group: 48 textures of 1 MiB, then 12 of 4 MiB.
        for (count, mib) in [(48, 1), (12, 4)] {
            let textures: Vec<&[u8]> = (0..count)
                .map(|t| &big[(t * (mib << 20)) % (32 << 20)..][..mib << 20])
                .collect();
            let one = time(&|| {
                for t in &textures {
                    std::hint::black_box(blake3::hash(t));
                }
            });
            let total = (count * (mib << 20)) as f64;
            let mut line = format!(
                "{count} inputs of {mib} MiB: {:.2} GB/s one by one",
                total / one / 1e9
            );
            for &n in &counts {
                let side_by_side = time(&|| {
                    for_each(&textures, n, |t| {
                        std::hint::black_box(blake3::hash(t));
                    });
                });
                line += &format!(
                    ", {:.2} side by side on {n} ({:.1}x)",
                    total / side_by_side / 1e9,
                    one / side_by_side
                );
            }
            println!("{line}");
        }
    }
}
