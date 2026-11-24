//! What a submission's shaders wrote, merged into title memory
//! ([`crate::shadow`]'s copy-back), apart so it can be tested and timed on
//! any host.

/// Merges one piece: where the host copy `host` differs from its bytes
/// before the submission (`old`), the GPU wrote, and title memory (`title`,
/// as just read) takes the copy's byte; then the copy takes title memory's
/// bytes everywhere (the title may have written the rest meanwhile).
/// Returns whether `title` changed, which is whether it must be written
/// back.
///
/// Byte for byte the same as
/// `if h != o && t != h { t = h; changed = true } h = t`, written without a
/// branch so the compiler vectorises it (about six times the branching
/// loop's speed on baseline x86-64, `merge_speed`).
pub(crate) fn merge(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    assert!(title.len() == host.len() && host.len() == old.len());
    let mut changed = 0u8;
    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
        let new = if *h != *o { *h } else { *t };
        changed |= new ^ *t;
        *t = new;
        *h = new;
    }
    changed != 0
}

#[cfg(test)]
mod tests {
    use super::merge;

    /// The byte loop `copy_back` ran before (the reference).
    fn merge_bytes(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
        let mut changed = false;
        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
            if *h != *o && *t != *h {
                *t = *h;
                changed = true;
            }
            *h = *t;
        }
        changed
    }

    /// A small deterministic generator (xorshift).
    fn bytes(seed: &mut u64, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                (*seed >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn merging_matches_the_byte_loop() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        // Lengths around the word size, and pieces where the GPU wrote
        // nothing, some bytes, or every byte; title memory equal to the
        // GPU's bytes in places, so "changed" is both true and false.
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 4096 + 3] {
            for density in [0u32, 1, 8, 128, 255, 256] {
                let old = bytes(&mut seed, len);
                let noise = bytes(&mut seed, len);
                let mut host = old.clone();
                let mut title = bytes(&mut seed, len);
                for i in 0..len {
                    if u32::from(noise[i]) < density {
                        host[i] = noise[i] ^ 0x5A;
                        if i % 3 == 0 {
                            title[i] = host[i];
                        }
                    }
                }
                let (mut t1, mut h1) = (title.clone(), host.clone());
                let (mut t2, mut h2) = (title, host);
                let a = merge_bytes(&mut t1, &mut h1, &old);
                let b = merge(&mut t2, &mut h2, &old);
                assert_eq!((a, &t1, &h1), (b, &t2, &h2), "len {len}, density {density}");
            }
        }
    }

    /// Throughput of the byte loop and of `merge` over a 64 KiB piece the
    /// GPU partly wrote. `cargo test --release -p victory-gpu merge_speed --
    /// --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn merge_speed() {
        const PIECE: usize = 64 << 10;
        let mut seed = 1u64;
        let old = bytes(&mut seed, PIECE);
        let mut host = old.clone();
        for (i, b) in host.iter_mut().enumerate() {
            if i % 3 != 0 {
                *b ^= 0x81;
            }
        }
        let title0 = bytes(&mut seed, PIECE);
        let rounds = 4000;
        let time = |f: fn(&mut [u8], &mut [u8], &[u8]) -> bool| {
            let (mut t, mut h) = (title0.clone(), host.clone());
            let start = std::time::Instant::now();
            let mut any = false;
            for _ in 0..rounds {
                t.copy_from_slice(&title0);
                h.copy_from_slice(&host);
                any |= f(
                    std::hint::black_box(&mut t),
                    std::hint::black_box(&mut h),
                    std::hint::black_box(&old),
                );
            }
            assert!(any);
            let s = start.elapsed().as_secs_f64();
            (rounds * PIECE) as f64 / s / 1e9
        };
        let base = time(|_, _, _| true);
        let bytewise = time(merge_bytes);
        let merged = time(merge);
        eprintln!(
            "64 KiB pieces, GPU wrote 2/3 of the bytes: byte loop {bytewise:.2} GB/s, \
             merge {merged:.2} GB/s (the two piece copies alone: {base:.2} GB/s)"
        );
    }
}
