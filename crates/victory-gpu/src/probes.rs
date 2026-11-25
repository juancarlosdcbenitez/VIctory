//! The probe buffer translated shaders write when they carry probes
//! (`victory_shader::emit::probes`: `VICTORY_SHADER_PROBES=1` in a
//! diagnostics build; GPU-1 and PERF-3 in `docs/engineering-status.md`).
//!
//! Every draw and dispatch of a pipeline with probes gets the bound heaps'
//! slot counts (root constants) and this buffer (a root UAV). A shader
//! counts and records each heap access whose index is at or past its
//! heap's slot count, and each lane the loop limit stops; the accesses
//! themselves are not changed. Submissions run one at a time and are
//! waited for (decision 0007), so after each one the buffer is read here,
//! what it holds is logged with the shader it names, and it is cleared.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_shader::emit::probe;
use windows::Win32::Graphics::Direct3D12::*;

/// The buffer: the resource, its mapped words and its GPU address.
struct Buffer {
    resource: ID3D12Resource,
    words: *mut u32,
    gpu: u64,
}

// SAFETY: the mapping is read and written only between submissions, which
// run one at a time (`host::Host::execute` holds its lock), or for a
// failure report, when the run ends.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

static BUFFER: OnceLock<Option<Buffer>> = OnceLock::new();

/// The probe buffer, made (cleared) on first use; `None` without a host.
fn buffer() -> Option<&'static Buffer> {
    BUFFER
        .get_or_init(|| {
            let host = crate::host::host()?;
            let bytes = 4 * probe::DWORDS as u64;
            let (resource, data) = host.shared_buffer(bytes)?;
            let words = data.cast::<u32>();
            // SAFETY: the mapping holds `DWORDS` words.
            unsafe { std::ptr::write_bytes(words, 0, probe::DWORDS) };
            // SAFETY: a live buffer.
            let gpu = unsafe { resource.GetGPUVirtualAddress() };
            crate::host::name(&resource, || "probe buffer".into());
            tracing::warn!("shader probes on: heap indices and loop-limit stops are recorded");
            Some(Buffer {
                resource,
                words,
                gpu,
            })
        })
        .as_ref()
}

/// Which shader each probe id names, as translation logged it.
static NAMES: Mutex<Option<HashMap<u32, String>>> = Mutex::new(None);

/// A shader translated with probes, under `id`.
pub(crate) fn name(id: u32, shader: impl FnOnce() -> String) {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names
        .get_or_insert_with(HashMap::new)
        .entry(id)
        .or_insert_with(shader);
}

fn name_of(id: u32) -> String {
    let names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names
        .as_ref()
        .and_then(|n| n.get(&id).cloned())
        .unwrap_or_else(|| format!("shader {id:#010x}"))
}

/// Binds the probe constants (`param`) and buffer (`param + 1`) for a draw
/// or dispatch: `slots` are the bound view and sampler heaps' slot counts.
pub(crate) fn bind(list: &ID3D12GraphicsCommandList, param: u32, compute: bool, slots: [u32; 2]) {
    let Some(b) = buffer() else { return };
    // SAFETY: an open list; the root signature has both parameters, and the
    // buffer lives for the run.
    unsafe {
        if compute {
            list.SetComputeRoot32BitConstants(param, 2, slots.as_ptr().cast(), 0);
            list.SetComputeRootUnorderedAccessView(param + 1, b.gpu);
        } else {
            list.SetGraphicsRoot32BitConstants(param, 2, slots.as_ptr().cast(), 0);
            list.SetGraphicsRootUnorderedAccessView(param + 1, b.gpu);
        }
    }
    let _ = &b.resource;
}

/// A record of the buffer, decoded for a log line.
fn describe(record: [u32; 4]) -> String {
    let [id, site, index, slots] = record;
    let shader = name_of(id);
    if site == probe::LOOP_SITE {
        return format!("{shader}: a lane stopped by the loop limit");
    }
    let heap = if site & probe::SAMPLER != 0 {
        "sampler"
    } else {
        "view"
    };
    let lanes = if site & probe::PER_LANE != 0 {
        "per-lane"
    } else {
        "uniform"
    };
    format!(
        "{shader}: heap access {} ({lanes}) at {heap} index {index:#x}, {} past the bound heap's {slots} slots",
        site & probe::SITE,
        index.wrapping_sub(slots)
    )
}

/// What the buffer holds: the counts and the records kept.
fn read(b: &Buffer) -> ([u32; 3], Vec<[u32; 4]>) {
    // SAFETY: the mapping holds `DWORDS` words; no submission is running.
    let words = unsafe { std::slice::from_raw_parts(b.words, probe::DWORDS) };
    let counts = [
        words[probe::PAST_HEAP],
        words[probe::LOOP_STOPS],
        words[probe::RECORDS],
    ];
    let kept = counts[2].min(probe::RING) as usize;
    let records = (0..kept)
        .map(|k| {
            let at = probe::RING_START + 4 * k;
            [words[at], words[at + 1], words[at + 2], words[at + 3]]
        })
        .collect();
    (counts, records)
}

/// Accesses past a heap and loop-limit stops since the run began.
static TOTALS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// How many records are logged one by one over the run.
const LOGGED: u64 = 64;

/// After a submission ran: logs what its shaders recorded, then clears the
/// buffer for the next one.
pub(crate) fn after_submission(pipelines: impl FnOnce() -> String) {
    let Some(b) = BUFFER.get().and_then(Option::as_ref) else {
        return;
    };
    let ([past, stops, records], kept) = read(b);
    if past == 0 && stops == 0 {
        return;
    }
    TOTALS[0].fetch_add(u64::from(past), Ordering::Relaxed);
    TOTALS[1].fetch_add(u64::from(stops), Ordering::Relaxed);
    tracing::warn!(
        past_heap = past,
        loop_stops = stops,
        records,
        pipelines = pipelines(),
        "probes: a submission's shaders indexed past a heap or hit the loop limit"
    );
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    for record in kept {
        if SHOWN.fetch_add(1, Ordering::Relaxed) < LOGGED {
            tracing::warn!("probe: {}", describe(record));
        }
    }
    // SAFETY: as in `read`.
    unsafe { std::ptr::write_bytes(b.words, 0, probe::DWORDS) };
}

/// For a GPU failure report: what the buffer holds now (the failed
/// submission's records, as far as they reached memory) and the run's
/// totals. Empty without probes.
pub(crate) fn summary() -> String {
    let Some(b) = BUFFER.get().and_then(Option::as_ref) else {
        return String::new();
    };
    let ([past, stops, records], kept) = read(b);
    let mut text = format!(
        "; probes: {past} heap accesses past a heap and {stops} loop-limit stops in the last \
         submission ({records} records), {} and {} before",
        TOTALS[0].load(Ordering::Relaxed),
        TOTALS[1].load(Ordering::Relaxed)
    );
    for record in kept.into_iter().take(8) {
        text.push_str(&format!("; probe: {}", describe(record)));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_read_as_what_they_name() {
        name(0x1234, || "pixel shader 0xabc".into());
        let past = describe([0x1234, probe::PER_LANE | 3, 0x10_0005, 0x10_0000]);
        assert_eq!(
            past,
            "pixel shader 0xabc: heap access 3 (per-lane) at view index 0x100005, 5 past the \
             bound heap's 1048576 slots"
        );
        let sampler = describe([0x99, probe::SAMPLER, 2048, 2048]);
        assert!(sampler.starts_with("shader 0x00000099: heap access 0 (uniform) at sampler"));
        let stop = describe([0x1234, probe::LOOP_SITE, 0, 0]);
        assert_eq!(stop, "pixel shader 0xabc: a lane stopped by the loop limit");
    }
}
