//! Where one list of a submission depends on an earlier one through title
//! memory (invariant C5, architecture.md).
//!
//! A submission's work at submission time runs on the CPU before any of its
//! GPU work: title buffer copies, fills of upload memory, rectangles
//! expanded into triangles, and the refreshes that bring host copies of
//! title memory up to date for the GPU. D3D12 orders all of a list's
//! commands after everything the earlier lists of the submission do. So the
//! submission is cut before a list whose work at submission reads or writes
//! title memory that GPU work of an earlier list in its part writes, or
//! writes title memory such GPU work reads; and before a list whose GPU
//! work reads or writes, through another copy of the same title memory
//! (aliased or nested buffers, a texture over a buffer's memory), what
//! earlier GPU work writes: each copy is brought up to date from title
//! memory alone. Each part runs, and what its GPU work wrote reaches title
//! memory, before the next part's work at submission, as if the title had
//! submitted the parts one after another. Lists that touch nothing of each
//! other's that way stay in one part.
//!
//! Inside one list nothing can be cut: a copy whose source shaders recorded
//! before it write runs on the GPU instead (`copy`), and the rest is counted
//! (`host::STALE_COPIES`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Copies by address: hashed by a multiply (std's hasher cost more than
/// the rest of a check, at thousands of reads per list).
type Map<V> = HashMap<usize, V, BuildHasherDefault<Mix>>;

#[derive(Default)]
struct Mix(u64);

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// Title memory `[start, end)`.
pub(crate) type Range = (usize, usize);

/// The copy that bottom-of-pipe writes go to: title memory itself, written
/// once the list has run (after what shaders wrote is copied back).
pub(crate) const TITLE: usize = 0;

/// What one list does with title memory when its submission runs.
#[derive(Debug, Default, Clone)]
pub(crate) struct Access {
    /// What its GPU work writes, with the copy of title memory it writes (a
    /// buffer's host copy, by address; [`TITLE`] for bottom-of-pipe
    /// writes).
    pub(crate) gpu_writes: Vec<(usize, Range)>,
    /// What its GPU work reads, with the copy it reads (a buffer's host copy
    /// or a texture's staging buffer, by address), brought up to date when
    /// the submission starts.
    pub(crate) gpu_reads: Vec<(usize, Range)>,
    /// What its work at submission reads: copy and fill sources, vertices
    /// expanded on the CPU.
    pub(crate) cpu_reads: Vec<Range>,
    /// What its work at submission writes: copy destinations.
    pub(crate) cpu_writes: Vec<Range>,
    /// The title memory each copy it reads or writes through holds (its
    /// buffer's, its texture's), once per copy: only copies holding some of
    /// the same memory can disagree about it.
    pub(crate) copies: Vec<(usize, Range)>,
}

/// Why a submission is cut before a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Hazard {
    /// Its work at submission reads what earlier GPU work writes.
    ReadAfterWrite,
    /// Its work at submission writes what earlier GPU work writes.
    WriteAfterWrite,
    /// Its work at submission writes what earlier GPU work reads.
    WriteAfterRead,
    /// Its GPU work reads, through another copy, what earlier GPU work
    /// writes.
    AliasedRead,
    /// Its GPU work writes, through another copy, what earlier GPU work
    /// writes.
    AliasedWrite,
}

impl Hazard {
    /// For logs.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) fn text(self) -> &'static str {
        match self {
            Hazard::ReadAfterWrite => {
                "its work at submission reads what an earlier list's GPU work writes"
            }
            Hazard::WriteAfterWrite => {
                "its work at submission writes what an earlier list's GPU work writes"
            }
            Hazard::WriteAfterRead => {
                "its work at submission writes what an earlier list's GPU work reads"
            }
            Hazard::AliasedRead => {
                "its GPU work reads, through another copy, what an earlier list's GPU work writes"
            }
            Hazard::AliasedWrite => {
                "its GPU work writes, through another copy, what an earlier list's GPU work writes"
            }
        }
    }
}

/// Where a submission of `lists` (in order) is cut: each list that starts a
/// new part, with the first hazard found. Empty when the lists can run as
/// one submission.
pub(crate) fn cuts(lists: &[Access]) -> Vec<(usize, Hazard)> {
    let shared = Shared::of(lists);
    let mut cuts = Vec::new();
    let mut part = Part::default();
    for (i, list) in lists.iter().enumerate() {
        if let Some(hazard) = part.hazard(list) {
            cuts.push((i, hazard));
            part = Part::default();
        }
        part.add(list, &shared);
    }
    cuts
}

/// What the checks need from the whole submission. A world submission
/// reads thousands of ranges per list and copies or aliases little, so only
/// what can matter is kept: reads that some work at submission writes, and
/// writes through copies another copy shares memory with.
struct Shared {
    /// Copies holding memory another copy holds too, each with those others
    /// ([`TITLE`] among them where a bottom-of-pipe write lands in a copy).
    aliases: Map<Vec<usize>>,
    /// The memory each copy holds (all it was seen holding).
    spans: Map<Range>,
    /// What the submission's work at submission writes.
    cpu_written: Ranges,
}

impl Shared {
    fn of(lists: &[Access]) -> Shared {
        let mut spans: Vec<(Range, usize)> = Vec::new();
        for list in lists {
            spans.extend(list.copies.iter().map(|&(copy, r)| (r, copy)));
            let title = list.gpu_writes.iter().filter(|&&(copy, _)| copy == TITLE);
            spans.extend(title.map(|&(copy, r)| (r, copy)));
        }
        spans.retain(|&((start, end), _)| start < end);
        spans.sort_unstable();
        spans.dedup();
        let mut held: Map<Range> = Map::default();
        for &((start, end), copy) in &spans {
            let span = held.entry(copy).or_insert((start, end));
            *span = (span.0.min(start), span.1.max(end));
        }
        let mut aliases: Map<Vec<usize>> = Map::default();
        for (i, &((_, end), a)) in spans.iter().enumerate() {
            // Sorted by start: the spans after this one starting before its
            // end overlap it, and only those.
            for &((start, _), b) in &spans[i + 1..] {
                if start >= end {
                    break;
                }
                for (x, y) in [(a, b), (b, a)] {
                    let others = aliases.entry(x).or_default();
                    if x != y && !others.contains(&y) {
                        others.push(y);
                    }
                }
            }
        }
        aliases.retain(|_, others| !others.is_empty());
        let mut cpu_written = Ranges::default();
        cpu_written.extend(lists.iter().flat_map(|l| l.cpu_writes.iter().copied()));
        Shared {
            aliases,
            spans: held,
            cpu_written,
        }
    }
}

/// What the GPU work of a part's lists so far does.
#[derive(Default)]
struct Part {
    /// Written, through any copy.
    written: Ranges,
    /// For each copy sharing memory with others, what those others wrote
    /// in its memory, by copy (few copies: a sorted list).
    foreign: Vec<(usize, Ranges)>,
    /// Read, where the submission's work at submission writes.
    read: Ranges,
}

impl Part {
    fn hazard(&self, list: &Access) -> Option<Hazard> {
        // Any range, through its copy, over what other copies of the same
        // memory wrote (a list's ranges come mostly in runs of one copy).
        let elsewhere = |ranges: &[(usize, Range)]| {
            let mut last: Option<(usize, Option<&Ranges>)> = None;
            ranges.iter().any(|&(copy, r)| {
                let foreign = match last {
                    Some((c, foreign)) if c == copy => foreign,
                    _ => {
                        let found = self.foreign.binary_search_by_key(&copy, |&(c, _)| c);
                        let foreign = found.ok().map(|i| &self.foreign[i].1);
                        last = Some((copy, foreign));
                        foreign
                    }
                };
                foreign.is_some_and(|w| w.overlaps(r))
            })
        };
        if list.cpu_reads.iter().any(|&r| self.written.overlaps(r)) {
            Some(Hazard::ReadAfterWrite)
        } else if list.cpu_writes.iter().any(|&r| self.written.overlaps(r)) {
            Some(Hazard::WriteAfterWrite)
        } else if list.cpu_writes.iter().any(|&r| self.read.overlaps(r)) {
            Some(Hazard::WriteAfterRead)
        } else if self.foreign.is_empty() {
            None
        } else if elsewhere(&list.gpu_reads) {
            Some(Hazard::AliasedRead)
        } else if elsewhere(&list.gpu_writes) {
            Some(Hazard::AliasedWrite)
        } else {
            None
        }
    }

    fn add(&mut self, list: &Access, shared: &Shared) {
        self.written.extend(list.gpu_writes.iter().map(|&(_, r)| r));
        let mut shared_writes: Vec<(usize, Range)> = list
            .gpu_writes
            .iter()
            .copied()
            .filter(|(copy, _)| shared.aliases.contains_key(copy))
            .collect();
        shared_writes.sort_unstable_by_key(|&(copy, _)| copy);
        for writes in shared_writes.chunk_by(|a, b| a.0 == b.0) {
            for &other in &shared.aliases[&writes[0].0] {
                // Only what lands in the other copy's memory can matter.
                let (start, end) = shared.spans[&other];
                let inside = writes
                    .iter()
                    .map(|&(_, (s, e))| (s.max(start), e.min(end)))
                    .filter(|&(s, e)| s < e);
                let inside: Vec<Range> = inside.collect();
                if inside.is_empty() {
                    continue;
                }
                let at = match self.foreign.binary_search_by_key(&other, |&(c, _)| c) {
                    Ok(at) => at,
                    Err(at) => {
                        self.foreign.insert(at, (other, Ranges::default()));
                        at
                    }
                };
                self.foreign[at].1.extend(inside);
            }
        }
        if !shared.cpu_written.0.is_empty() {
            let reads = list.gpu_reads.iter().map(|&(_, r)| r);
            self.read
                .extend(reads.filter(|&r| shared.cpu_written.overlaps(r)));
        }
    }
}

/// Ranges, sorted and disjoint (touching ones merged).
#[derive(Default, Debug)]
struct Ranges(Vec<Range>);

impl Ranges {
    /// Adds `more` (empty ranges left out): sorted, then merged with what is
    /// there in one pass.
    fn extend(&mut self, more: impl IntoIterator<Item = Range>) {
        let mut more: Vec<Range> = more.into_iter().filter(|&(s, e)| s < e).collect();
        if more.is_empty() {
            return;
        }
        more.sort_unstable();
        let old = std::mem::take(&mut self.0);
        let mut merged: Vec<Range> = Vec::with_capacity(old.len() + more.len());
        let (mut a, mut b) = (old.into_iter().peekable(), more.into_iter().peekable());
        loop {
            let next = match (a.peek(), b.peek()) {
                (Some(x), Some(y)) if x <= y => a.next(),
                (Some(_), Some(_)) | (None, Some(_)) => b.next(),
                (Some(_), None) => a.next(),
                (None, None) => break,
            };
            let Some((start, end)) = next else { break };
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        self.0 = merged;
    }

    /// Whether any byte of `[start, end)` is in a range.
    fn overlaps(&self, (start, end): Range) -> bool {
        let (Some(first), Some(last)) = (self.0.first(), self.0.last()) else {
            return false;
        };
        if start >= end || end <= first.0 || start >= last.1 {
            return false;
        }
        let i = self.0.partition_point(|&(_, e)| e <= start);
        self.0.get(i).is_some_and(|&(s, _)| s < end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: usize = 0x1000;
    const B: usize = 0x2000;
    /// Host copies (by address).
    const COPY: usize = 0xc0;
    const OTHER: usize = 0xd0;

    /// The memory every test copy holds: all of the tests' addresses, so
    /// that any two copies share it.
    const SPAN: Range = (0, 1 << 20);

    fn gpu_write(copy: usize, at: usize, len: usize) -> Access {
        Access {
            gpu_writes: vec![(copy, (at, at + len))],
            copies: vec![(copy, SPAN)],
            ..Access::default()
        }
    }

    fn copy(dst: usize, src: usize, len: usize) -> Access {
        Access {
            cpu_reads: vec![(src, src + len)],
            cpu_writes: vec![(dst, dst + len)],
            ..Access::default()
        }
    }

    fn gpu_read(copy: usize, at: usize, len: usize) -> Access {
        Access {
            gpu_reads: vec![(copy, (at, at + len))],
            copies: vec![(copy, SPAN)],
            ..Access::default()
        }
    }

    #[test]
    fn ranges_answer_overlaps_byte_for_byte() {
        let mut r = Ranges::default();
        r.extend([(30, 40), (10, 20), (50, 50)]);
        r.extend([(100, 200), (20, 25)]);
        assert_eq!(r.0, vec![(10, 25), (30, 40), (100, 200)]);
        assert!(r.overlaps((24, 26)) && r.overlaps((39, 41)) && r.overlaps((0, 11)));
        assert!(!r.overlaps((25, 30)), "between two ranges");
        assert!(!r.overlaps((40, 100)) && !r.overlaps((200, 300)) && !r.overlaps((0, 10)));
        assert!(!r.overlaps((150, 150)), "an empty range touches nothing");
        assert!(r.overlaps((0, usize::MAX)));
    }

    #[test]
    fn a_copy_of_what_an_earlier_list_writes_on_the_gpu_starts_a_part() {
        let lists = [gpu_write(COPY, A, 1024), copy(B, A, 1024)];
        assert_eq!(cuts(&lists), [(1, Hazard::ReadAfterWrite)]);
        // The same list twice: its second run copies what its first wrote.
        let both = Access {
            gpu_writes: vec![(COPY, (A, A + 1024))],
            ..copy(B, A, 1024)
        };
        assert_eq!(cuts(&[both.clone(), both]), [(1, Hazard::ReadAfterWrite)]);
    }

    #[test]
    fn lists_sharing_no_byte_stay_one_submission() {
        // The copy reads the half the shaders do not write, into memory no
        // GPU work of the submission uses; others share only reads.
        let lists = [
            gpu_write(COPY, A, 512),
            copy(B, A + 512, 512),
            gpu_read(COPY, A + 512, 512),
            gpu_read(OTHER, B + 4096, 16),
        ];
        assert!(cuts(&lists).is_empty());
        // A copy's write followed by GPU work reading it is in order already
        // (the copy runs first), and so is GPU work of one list after another's.
        let lists = [
            copy(B, A, 64),
            gpu_read(COPY, B, 64),
            gpu_write(COPY, B, 64),
        ];
        assert!(cuts(&lists).is_empty());
        let lists = [
            gpu_write(COPY, A, 64),
            gpu_read(COPY, A, 64),
            gpu_write(COPY, A, 8),
        ];
        assert!(cuts(&lists).is_empty(), "the same copy: the GPU orders it");
    }

    #[test]
    fn partial_overlaps_count_and_edges_do_not() {
        let at_edge = [gpu_write(COPY, A, 512), copy(B, A + 512, 256)];
        assert!(cuts(&at_edge).is_empty());
        let one_byte = [gpu_write(COPY, A, 512), copy(B, A + 511, 256)];
        assert_eq!(cuts(&one_byte), [(1, Hazard::ReadAfterWrite)]);
        let covering = [gpu_write(COPY, A + 100, 8), copy(B, A, 512)];
        assert_eq!(cuts(&covering), [(1, Hazard::ReadAfterWrite)]);
    }

    #[test]
    fn writes_over_gpu_work_start_a_part() {
        // The copy's bytes must land after the shaders' (copied back first).
        let lists = [gpu_write(COPY, A, 64), copy(A + 32, B, 64)];
        assert_eq!(cuts(&lists), [(1, Hazard::WriteAfterWrite)]);
        // An earlier list's GPU work must read what was there before the copy.
        let lists = [gpu_read(COPY, A, 64), copy(A, B, 64), gpu_read(COPY, A, 64)];
        assert_eq!(cuts(&lists), [(1, Hazard::WriteAfterRead)]);
    }

    #[test]
    fn copies_of_the_same_memory_are_kept_apart() {
        // A nested or aliased buffer, or a texture over the memory: its copy
        // is brought up to date from title memory only.
        let lists = [gpu_write(COPY, A, 256), gpu_read(OTHER, A + 128, 16)];
        assert_eq!(cuts(&lists), [(1, Hazard::AliasedRead)]);
        let lists = [gpu_write(COPY, A, 256), gpu_write(OTHER, A + 128, 16)];
        assert_eq!(cuts(&lists), [(1, Hazard::AliasedWrite)]);
        // Bottom-of-pipe writes reach title memory after the list ran.
        let lists = [gpu_write(TITLE, A, 8), copy(B, A, 8)];
        assert_eq!(cuts(&lists), [(1, Hazard::ReadAfterWrite)]);
        let lists = [gpu_write(TITLE, A, 8), gpu_read(COPY, A, 8)];
        assert_eq!(cuts(&lists), [(1, Hazard::AliasedRead)]);
    }

    #[test]
    fn each_dependent_list_starts_its_own_part() {
        // Shaders write A; a copy takes A to B; shaders (a GPU copy) read B and
        // write C; a copy reads C. A new part starts empty: a later copy of A
        // alone would not cut again (A is back in title memory by then).
        let lists = [
            gpu_write(COPY, A, 64),
            copy(B, A, 64),
            Access {
                gpu_reads: vec![(OTHER, (B, B + 64))],
                gpu_writes: vec![(OTHER, (0x3000, 0x3040))],
                copies: vec![(OTHER, SPAN)],
                ..Access::default()
            },
            copy(0x4000, 0x3000, 64),
            copy(0x5000, A, 64),
        ];
        assert_eq!(
            cuts(&lists),
            [(1, Hazard::ReadAfterWrite), (3, Hazard::ReadAfterWrite)]
        );
    }

    /// What deciding the cuts costs a submission the size of a world one
    /// (48 lists, each with 2,000 ranges read through 8 host copies, 50
    /// written through 4, 200 fills and 5 copies), none depending on another
    /// (every list checked against everything before it): CPU time per
    /// submission, not a frame rate. `cargo test --release -p victory-gpu
    /// cut_decisions -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn cut_decisions_per_submission() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: usize| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as usize % n
        };
        // Each host copy's ranges in its own buffer, as distinct buffers are.
        let mut range = |base: usize, span: usize, len: usize| {
            let at = base + next(span - len);
            (at, at + len)
        };
        let lists: Vec<Access> = (0..48)
            .map(|_| Access {
                gpu_reads: (0..2000)
                    .map(|i| (i % 8, range((i % 8) << 25, 1 << 25, 4096)))
                    .collect(),
                gpu_writes: (0..50)
                    .map(|i| (16 + i % 4, range((8 + i % 4) << 26, 1 << 26, 1 << 20)))
                    .collect(),
                cpu_reads: (0..205).map(|_| range(0x3000_0000, 1 << 28, 256)).collect(),
                cpu_writes: (0..5).map(|_| range(0x4000_0000, 1 << 28, 4096)).collect(),
                copies: (0..8)
                    .map(|c| (c, (c << 25, (c + 1) << 25)))
                    .chain((0..4).map(|c| (16 + c, ((8 + c) << 26, (9 + c) << 26))))
                    .collect(),
            })
            .collect();
        let time = |lists: &[Access], what: &str| {
            assert!(cuts(lists).is_empty());
            let rounds = 50;
            let started = std::time::Instant::now();
            for _ in 0..rounds {
                assert!(cuts(std::hint::black_box(lists)).is_empty());
            }
            let us = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
            println!("cut decisions, {what}: {us:.0} us per submission of 48 lists");
        };
        time(&lists, "separate buffers");
        // The same inside one big buffer whose copy shaders write too (a
        // heap's buffer and buffers placed in it): every read through the
        // inner copies is checked against the big one's writes.
        let nested: Vec<Access> = lists
            .iter()
            .map(|list| {
                let mut list = list.clone();
                list.copies.push((99, (0, 1 << 29)));
                list.gpu_writes
                    .extend((0..10).map(|_| (99, range(1 << 28, 1 << 28, 1 << 16))));
                list
            })
            .collect();
        time(&nested, "nested in one written buffer");
    }
}
