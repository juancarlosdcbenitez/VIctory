# Architecture

```
 user's files (outside repo)             VIctory
 ───────────────────────────             ─────────────────────────────────────────
 game assets ──────────────────────────► victory-pipeline + victory-game
                                         (victory-formats, victory-shader)
                                                    │
                                                    ▼
                                         cache_dir: converted data
                                                    │
 game executable ─► victory-loader ────► title code (native x86-64)
                    prepare, map, bind      │ imports
                                            ▼
                                         victory-hle ── host Windows (shared APIs)
                                            │ D3D12.X
                                            ▼
                                         victory-gpu ─► host D3D12
                                            │ shaders at pipeline creation
                                            ▼
                                         victory-shader ◄─► cache_dir
```

Shaders are translated when the title creates a pipeline, from the bytes it
passes, wherever it loaded them from (executable, archive, memory).
Converting them ahead of time from files is only a warm-up.

## Crates

| Crate | Role |
|-------|------|
| `victory-core` | config, asset-policy guard, logging |
| `victory-pe` | read-only PE32+ and import-library analysis |
| `victory-loader` | prepare, map, bind and start the title image |
| `victory-hle` | console API implementations, host pass-through policy |
| `victory-gpu` | D3D12.X objects, command lists and GPU memory → host D3D12 |
| `victory-shader` | Xbox shader containers → host shaders |
| `victory-formats` | the GPU's texture layouts (sizes, tiling, detiling), audio codecs; planned: BCPack |
| `victory-pipeline` | converter trait, content-addressed cache |
| `victory-game` | Grand Theft Auto VI-specific formats, build identities, hooks |
| `apps/victory` | command-line front end |
| `xtask` | asset-policy check, git hooks |
| `tests/fixtures/hle-probe` | a stand-in "title" (our own code) for loader tests |

Everything except `victory-game` is platform-level and can be tested with
any Xbox Series title.

## Import resolution

For every `dll!function` the title imports, in order:

1. our implementation in `victory-hle::modules` (always wins);
2. the host's own function, if `HostPolicy` allows that DLL;
3. an export of a DLL the title ships beside its executable, mapped like
   the executable (only DLLs neither of the above serves);
4. a generated stub that names the import when called (`--unimplemented
   trap` breaks into the debugger and aborts; `zero` logs once and returns 0).

`LoadLibrary*` of a console-only DLL we implement (e.g. `xgameruntime.dll`)
returns a pseudo handle; `GetProcAddress` on it finds our functions.
`LoadLibrary*` of a title DLL returns its mapped image, mapping it first if
nothing imports it. By name, as by import, a title gets only host DLLs
`HostPolicy` passes through, and `GetProcAddress` on one returns our
override where import binding would.
`QueryApiImpl` hands out our runtime objects by provider and interface id
(`victory-hle::modules::xgameruntime` lists the areas); online areas answer
as an offline console. Game saves are files under `VICTORY_DATA_DIR`.

Title storage is a drive on the console, and titles build paths on that.
`victory-hle::modules::files` binds every path-taking title import
(kernel32 and the C/C++ runtime) and rewrites paths on a virtual drive to
the host folder behind it; persistent local storage is one such drive over
a folder under `VICTORY_DATA_DIR`. Other paths reach the host unchanged.

An exe's code reaches its static TLS as slot 0; mapped as a DLL it gets
another slot, so the loader redirects those loads (`victory-loader::tls`).
`victory run --execute` installs a crash reporter that names faults as
`module+rva` and unwinds the faulting thread's stack
(`victory-loader::crash`).

`victory census` shows this classification without running anything (it
does not map title DLLs; `victory run` without `--execute` does).

## Shaders

`victory_shader::ir::build` turns a native program into SSA, one
instruction per machine instruction. `victory_shader::structure` then
removes EXEC: lane masks become per-lane booleans, wave-wide mask tests
become per-lane branches, and vector instructions keep a guard only where
a lane can reach them with its EXEC bit clear. `victory_shader::resource`
traces each descriptor and scalar memory address to the root parameter it
comes from (table, heap, root descriptor or constant). A descriptor a
waterfall loop reads from the first lane (`v_readfirstlane_b32`) is traced
to the per-lane load or choice that gives each lane its own (a mask's
`v_cndmask_b32`, or an if/else whose arms each set it: traced on the paths
the lanes take, from structurisation), and accessed with a non-uniform heap
index in one pass. `victory translate`
runs every shader it finds (containers, SGD5 files) through these and
lists what they cannot handle yet.

What a lowering computes is tested without a GPU: `emit::bitcode::eval`
runs the emitted module for one thread, and `emit::eval_tests` checks
synthetic programs on it against the ISA. See [testing.md](testing.md);
the engineering record and backlog are in
[engineering-status.md](engineering-status.md).

## GPU submission and its invariants

Title command lists are recorded into host command lists as the title
records them. A submission (`ExecuteCommandLists`, or held work the queue
worker releases) runs one at a time in `host::Host::execute`, under one
lock (decision 0007):

1. preparation: the title memory the lists read is copied into host
   copies, only what changed (buffers: `shadow`; textures:
   `texture::Staging`, checked side by side on the parallel pool);
2. the lists run on one host direct queue, and the CPU waits for them
   (host fence, no timeout);
3. what shaders wrote is copied back into title memory, then the
   bottom-of-pipe writes (fence values, timestamps) are made and the probe
   buffer is read.

What this keeps true, and what any overlap of CPU and GPU work (PERF-2)
has to keep true some other way:

| | Invariant | Kept today by | Kept with overlap by |
|-|-----------|---------------|----------------------|
| C1 | A host copy has one writer at a time: preparation (CPU) or one submission's GPU work | the lock and the CPU wait | a version of the copy per submission in flight, or copy-on-write while in flight |
| C2 | GPU-written ranges reach title memory before the submission's bottom-of-pipe writes and fence signals tell the title it finished | step order in `execute` | completion (copy-back, bottom-of-pipe writes, signals) run by the queue worker after the host fence |
| C3 | Preparation reads title memory only after every earlier submission's copy-back of the ranges it reads | the CPU wait | `inflight::InFlight`: wait for exactly the submission that writes a range about to be read |
| C4 | Work on one title queue runs in submission order, behind the waits it was submitted after | `schedule::Schedule`; work run at once claims its queue | unchanged |
| C5 | Work runs after the GPU work recorded before it (D3D12's order on a queue) | **across lists**: a submission is cut before a list whose work at submission (copies, fills of upload memory, rectangle expansion, all on the CPU before any GPU work) reads or writes title memory that GPU work of an earlier list in its part writes, or writes title memory such work reads, and before a list whose GPU work reads or writes, through another copy of the same memory, what earlier GPU work writes (`order`); each part runs, and what it wrote reaches title memory, before the next part's work at submission. **Inside a list** (nothing to cut): a title buffer or memory copy whose source shaders recorded earlier in the list write runs on the GPU, in order (`copy`: a compute pass between the host copies, the destination copied back afterwards); fills and expansions reading such memory still run first, counted and logged (`host::STALE_COPIES`), and so do copies over memory earlier GPU work in the list reads or writes (not counted) | the same; each cut would then cost a wait, where ordering the dependent work on the GPU would not |
| O1 | A host list belongs to the submission running it; a title list reset or released meanwhile gets a fresh host list | `Arc` shared with held work | unchanged |
| O2 | Host resources, allocators and descriptor slots a submission uses live until it finishes | released after `execute` returns, after the GPU finished; the dynamic descriptor slots of fresh tables and the list-owned upload memory are the host list's, so held work keeps them | `inflight::Retire`, released by host fence value |
| O3 | The probe buffer is read and cleared between submissions | the lock | a probe buffer per submission in flight |

What C5's cuts see (`order::Access`, from each list's steps): GPU reads
are the ranges brought up to date for it (buffer views, root views,
tables' buffers, textures it samples), each with the copy it reads
(buffers' host copies differ per title buffer, so nested or aliased
buffers, and textures over a buffer's memory, are different copies of the
same title memory, made consistent only through title memory); GPU writes
are the ranges declared written (unordered access, GPU copies) and
bottom-of-pipe writes; CPU reads and writes are the copies', fills' and
expansions' ranges. Not seen: buffers read through directly indexed heaps,
which are not brought up to date per use (shadow.rs), and what shaders
write into host textures (render targets, texture UAVs), which never
reaches title memory. Inside a list, a copy over memory earlier GPU work
reads or writes is not moved to the GPU: a fill after it in the list (a
constant buffer not 256-byte aligned) would then read title memory from
before it.

`inflight::Retire` and `inflight::InFlight` exist and are tested (WARP:
eight submissions in flight with their own upload buffers and allocators,
retired by fence value) but nothing uses them yet: every submission still
waits, so neither has anything to do.

Held work is already asynchronous for the title: a submission behind a
wait not reached runs later on the queue worker, with the host lists it
was recorded into. Until the sixth session two things it reads belonged to
the title list and went with its reset: the dynamic descriptor slots its
fresh tables were made in (owned by the draw state) and the list-owned
upload memory (`allocate_upload`). Both are now the host list's (O2).

**Asynchronous queues** (experimental, diagnostics builds,
`VICTORY_QUEUE_ASYNC=1`): every submission, signal, present and page
mapping is held and run by the queue worker in each queue's order, so a
title thread returns from `ExecuteCommandLists` at once and its CPU work
overlaps the GPU's. The worker runs `execute` as above, one submission at
a time and waiting for the GPU, so C1, C3 and O3 hold as they do today; a
queue's signals and presents run after its earlier submissions' copy-back
(C2); the schedule keeps C4; O1 and O2 hold through the host lists' own
ownership. Presents and page mappings take what they read at the call (the
texture's host copy, the mapping batches) and run in order, where they run
ahead of held work otherwise. What does not overlap: the worker's own
preparation and copy-back with the GPU. Off by default until a GPU run
(hardware-validation.md, stage 9) shows it safe and worth it.
