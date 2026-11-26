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
    let mut cuts = Vec::new();
    let mut part = Part::default();
    for (i, list) in lists.iter().enumerate() {
        if let Some(hazard) = part.hazard(list) {
            cuts.push((i, hazard));
            part = Part::default();
        }
        part.add(list);
    }
    cuts
}

/// What the GPU work of a part's lists so far does.
#[derive(Default)]
struct Part {
    /// Written, by copy (a part writes through few copies).
    writes: Vec<(usize, Ranges)>,
    /// Read, through any copy.
    reads: Ranges,
}

impl Part {
    fn hazard(&self, list: &Access) -> Option<Hazard> {
        let written = |r: Range| self.writes.iter().any(|(_, w)| w.overlaps(r));
        let elsewhere =
            |copy: usize, r: Range| self.writes.iter().any(|(c, w)| *c != copy && w.overlaps(r));
        if list.cpu_reads.iter().any(|&r| written(r)) {
            Some(Hazard::ReadAfterWrite)
        } else if list.cpu_writes.iter().any(|&r| written(r)) {
            Some(Hazard::WriteAfterWrite)
        } else if list.cpu_writes.iter().any(|&r| self.reads.overlaps(r)) {
            Some(Hazard::WriteAfterRead)
        } else if list.gpu_reads.iter().any(|&(c, r)| elsewhere(c, r)) {
            Some(Hazard::AliasedRead)
        } else if list.gpu_writes.iter().any(|&(c, r)| elsewhere(c, r)) {
            Some(Hazard::AliasedWrite)
        } else {
            None
        }
    }

    fn add(&mut self, list: &Access) {
        for &(copy, r) in &list.gpu_writes {
            match self.writes.iter_mut().find(|(c, _)| *c == copy) {
                Some((_, w)) => w.add(r),
                None => {
                    let mut w = Ranges::default();
                    w.add(r);
                    self.writes.push((copy, w));
                }
            }
        }
        for (_, w) in &mut self.writes {
            w.normalize();
        }
        for &(_, r) in &list.gpu_reads {
            self.reads.add(r);
        }
        self.reads.normalize();
    }
}

/// Ranges, sorted and disjoint once normalized.
#[derive(Default, Debug)]
struct Ranges(Vec<Range>);

impl Ranges {
    /// Adds `r` (empty ranges are left out); [`Ranges::normalize`] before
    /// asking.
    fn add(&mut self, (start, end): Range) {
        if start < end {
            self.0.push((start, end));
        }
    }

    /// Sorts and merges the ranges (touching ones too).
    fn normalize(&mut self) {
        self.0.sort_unstable();
        let mut merged: Vec<Range> = Vec::with_capacity(self.0.len());
        for (start, end) in self.0.drain(..) {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        self.0 = merged;
    }

    /// Whether any byte of `[start, end)` is in a range.
    fn overlaps(&self, (start, end): Range) -> bool {
        if start >= end {
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

    fn gpu_write(copy: usize, at: usize, len: usize) -> Access {
        Access {
            gpu_writes: vec![(copy, (at, at + len))],
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
            ..Access::default()
        }
    }

    #[test]
    fn ranges_answer_overlaps_byte_for_byte() {
        let mut r = Ranges::default();
        for range in [(30, 40), (10, 20), (20, 25), (50, 50), (100, 200)] {
            r.add(range);
        }
        r.normalize();
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
}
