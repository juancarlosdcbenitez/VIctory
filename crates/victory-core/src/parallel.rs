//! Work spread over a few threads where the result cannot depend on how it
//! is spread: a BLAKE3 hash equal bit for bit to [`blake3::hash`], and
//! independent items each handled once.
//!
//! Threads are scoped and started per call (tens of microseconds each), so
//! callers keep small inputs on their own thread: [`PARALLEL_HASH_MIN`] is
//! where a split hash starts to pay on the machines measured (see
//! `docs/engineering-status.md`, PERF-5).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use blake3::hazmat::{
    ChainingValue, HasherExt, Mode, left_subtree_len, merge_subtrees_non_root, merge_subtrees_root,
};

/// Inputs shorter than this are hashed on the calling thread.
pub const PARALLEL_HASH_MIN: usize = 2 << 20;

/// The smallest subtree a thread hashes: enough chunks for BLAKE3's own
/// SIMD parallelism inside it.
const LEAF_MIN: usize = 1 << 20;

/// How many threads parallel work uses: the machine's, at most 8.
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(8)
    })
}

/// The BLAKE3 hash of `bytes`, equal to `blake3::hash(bytes)`, computed on
/// up to `threads` threads for inputs of [`PARALLEL_HASH_MIN`] bytes and
/// more. The input is split where BLAKE3's own tree splits it (the left
/// subtree takes the largest power-of-two number of chunks below the whole),
/// so each thread hashes whole subtrees and the result does not depend on
/// the number of threads.
pub fn blake3(bytes: &[u8], threads: usize) -> blake3::Hash {
    if threads < 2 || bytes.len() < PARALLEL_HASH_MIN.max(2 * LEAF_MIN) {
        return blake3::hash(bytes);
    }
    let (left, right) = halves(bytes, 0, threads);
    merge_subtrees_root(&left, &right, Mode::Hash)
}

/// The chaining values of the two subtrees of `bytes`, which begins at
/// `offset` of the input and is longer than one chunk, hashed on up to
/// `threads` threads.
fn halves(bytes: &[u8], offset: u64, threads: usize) -> (ChainingValue, ChainingValue) {
    let split = left_subtree_len(bytes.len() as u64) as usize;
    let (left, right) = bytes.split_at(split);
    let right_threads = threads / 2;
    std::thread::scope(|s| {
        let right = s.spawn(move || subtree(right, offset + split as u64, right_threads));
        let left = subtree(left, offset, threads - right_threads);
        (
            left,
            right
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e)),
        )
    })
}

/// The chaining value of the subtree `bytes`, which begins at `offset`.
fn subtree(bytes: &[u8], offset: u64, threads: usize) -> ChainingValue {
    if threads < 2 || bytes.len() < 2 * LEAF_MIN {
        return blake3::Hasher::new()
            .set_input_offset(offset)
            .update(bytes)
            .finalize_non_root();
    }
    let (left, right) = halves(bytes, offset, threads);
    merge_subtrees_non_root(&left, &right, Mode::Hash)
}

/// Calls `f` once for each of `items`, on up to `threads` threads (the
/// caller's among them), each taking the next item not yet taken. The order
/// the items are handled in is not defined, so `f` must not depend on it.
pub fn for_each<T: Sync>(items: &[T], threads: usize, f: impl Fn(&T) + Sync) {
    let workers = threads.min(items.len());
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
    std::thread::scope(|s| {
        for _ in 1..workers {
            s.spawn(work);
        }
        work();
    });
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
        assert!((1..=8).contains(&threads()));
    }

    /// CPU throughput of hashing, one thread against this module (not a
    /// frame rate: what a submission's texture checks could save on this
    /// machine). Run with `cargo test --release -p victory-core --
    /// --ignored --nocapture hashing_throughput`.
    #[test]
    #[ignore = "benchmark"]
    fn hashing_throughput() {
        let time = |f: &dyn Fn()| {
            f(); // warm up
            let reps = 5;
            let started = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            started.elapsed().as_secs_f64() / f64::from(reps)
        };
        let threads = threads();
        println!("hashing_throughput: {threads} threads");
        let big = input(64 << 20);
        for mib in [1, 4, 16, 64] {
            let bytes = &big[..mib << 20];
            let one = time(&|| {
                std::hint::black_box(blake3::hash(std::hint::black_box(bytes)));
            });
            let split = time(&|| {
                std::hint::black_box(super::blake3(std::hint::black_box(bytes), threads));
            });
            println!(
                "one input of {mib} MiB: {:.2} GB/s on one thread, {:.2} GB/s split ({:.1}x)",
                bytes.len() as f64 / one / 1e9,
                bytes.len() as f64 / split / 1e9,
                one / split
            );
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
            let side_by_side = time(&|| {
                for_each(&textures, threads, |t| {
                    std::hint::black_box(blake3::hash(t));
                });
            });
            let total = (count * (mib << 20)) as f64;
            println!(
                "{count} inputs of {mib} MiB: {:.2} GB/s one by one, {:.2} GB/s side by side ({:.1}x)",
                total / one / 1e9,
                total / side_by_side / 1e9,
                one / side_by_side
            );
        }
    }
}
