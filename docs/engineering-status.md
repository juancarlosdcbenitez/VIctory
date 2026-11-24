# Engineering status

What is known, how it is known, and what comes next. The user-facing
status is the README's; this file is the engineering record behind it.
Classes: **verified** (a test or a run shows it), **verified off-GPU**
(checked on the shader evaluator, not yet on a host GPU), **implemented,
not run** (compiles, never executed), **hypothesis** (not yet checked).

## Session log

### 2026-11-24 (Linux container, no GPU, no game data)

Baseline: `ffe310e`. Rust 1.98.1 as pinned; x86-64, 4 cores, 15 GB.
Windows code type-checked from Linux (see [testing.md](testing.md));
nothing ran on Windows or a GPU, and no title was run.

Baseline results: CI's two clippy runs clean for the Windows target;
portable tests pass (victory-shader 96, victory-formats 23 with 4 data
tests ignored, victory-core 11, victory-game 2, victory-pipeline 4,
xtask 4); `victory-gpu` did not build outside Windows; `victory-pe` and
`victory-loader` tests fail on Linux only because they parse the ELF
test binary as PE.

Done (commits after `ffe310e`):

* **Shader evaluator** (verified): `emit::bitcode::eval` interprets the
  module the DXIL writer builds, for one compute thread;
  `emit::eval_tests` runs synthetic programs on it. It reproduces every
  hardware-verified case of `valu_tests` it covers (med3/max/min with NaN
  and signed zeros, v_cmp_lg with NaN, v_fract near 1, 16-bit
  conversions, large-turn sin/cos, bfe, v_cmp_class_f16, packed inline
  constants).
* **SCC defect fixed** (verified off-GPU): after `s_min_*`/`s_max_*` SCC
  is the comparison and after `s_lshl{1-4}_add_u32` the unsigned
  overflow; the lowering gave "result is not zero". The test fails on the
  old code.
* **59 ALU instructions translated** (verified off-GPU against ISA
  references; DXIL validity not checked here, see below): 22 SALU
  (`s_absdiff_i32`, `s_sext_*`, `s_bcnt0_*`, `s_ff0_*`, `s_pack_*`,
  `s_cmov*`, `s_brev_b64`, `s_flbit_i32_b64/i64`, `s_bitset*_b64`,
  `s_bfm_b64`, `s_bfe_*64`, `s_bitreplicate_b64_b32`) and 37 VALU
  (`v_mad_u64_u32`, `v_mad_i64_i32`, `v_ldexp_f32`, `v_frexp_*_f32`,
  24-bit `mul_hi`, `v_add/sub_nc_i32`, `v_mul_lo_i32`, `v_mad_*32_*16`,
  `v_cvt_pk_*16_*32`, `v_cvt_pknorm_*16_f32`, `v_sad_*`, `v_lerp_u8`,
  integer dot products, 16-bit `add/sub_nc_i16`, `mad`, `min3/max3/med3`).
  How many of the title's untranslated shaders this unblocks is unknown:
  `victory translate` over the title was not run.
* **ISA sweeps of existing lowerings** (verified off-GPU, no defects
  found): all VOPC predicates (f32, f16, 16/32/64-bit integers, class),
  conversions, rounding, omod/clamp, carries, 64-bit SALU, SDWA, f32 and
  f16 arithmetic and the legacy multiply, packed 16-bit integers,
  `v_perm_b32` with literal and computed selectors. The sweep found the
  constant `v_cmp_f/t_i64/u64` refused; they now translate.
* **victory-gpu builds on Linux** (verified): its 24 portable tests run.
* Evaluator programs are validated by `dxil.dll` wherever it loads
  (implemented, not run: there is no `dxil.dll` here; Windows CI has one).

Blocked here: running DXC's validator on Linux (a conda-forge build of the
open-source DXC was fetched but executing it was not permitted in this
session). The branch was pushed afterwards with the owner's authorisation.

### 2026-11-24, second session (same container)

Baseline: `d472bbd`. Each commit was pushed to the branch as it landed.

* **Copy-back merge without a branch per byte** (verified off-GPU):
  `Shadow::copy_back` merged GPU-written bytes into title memory with a
  branch per byte. `merge::merge` is the same rule in a form the compiler
  vectorises; it is tested byte for byte against the old loop.
  `merge_speed` (release, 64 KiB pieces) measures 1.6-2.1 GB/s before and
  11.9 GB/s after. What this saves per frame was not measured.
* **Detiling benchmark** (`detile_speed`): runs are 16 or 32 bytes;
  4.2-7.2 GB/s, bound by scattered reads. Fixed-size run copies measured
  the same, so the detiling code is unchanged.
* **Fence waits without an event block** (verified off-GPU, not run on
  Windows): `SetEventOnCompletion(value, NULL)` failed at once
  (`E_NOTIMPL`) when the value was not yet reached. It now blocks: woken
  by signals and polling for direct writes, failing after 5 s as before.
  A log line "a blocking fence wait was not reached in time" means a
  fence we never signal.
* **Integer clamp and half dot products** (verified off-GPU): the VOP3
  clamp bit on v_add/sub(_co)/nc_u32/i32, the 16-bit adds and subtracts
  and the integer dot products now saturates instead of stopping the
  translation. v_dot2_f32_f16 and v_dot2c_f32_f16 are translated in f32
  (shortcut: two roundings).
* **Lockstep waves in the evaluator** (verified): whole waves run in
  lockstep with the wave operations evaluated across lanes, and
  `synth::eval_wave` gives each lane its own inputs. Sweeps of the
  existing cross-lane lowering (readlane/writelane/readfirstlane, mbcnt,
  all DPP16 controls with masks and bound_ctrl, DPP8, ds_swizzle) found
  no defects, nor did sweeps of packed f16 arithmetic under op_sel and of
  group-shared memory across a wave (offsets, read2/write2 with st64,
  atomics).
* **v_permlane16/x16_b32 and ds_bpermute_b32 translated** (verified
  off-GPU). ds_bpermute's wave32 index wrap is our reading of the ISA.
* **ds_swizzle rotate and FFT modes refused** (offsets from 0xC000): the
  lowering had read them as quad permutes, giving wrong lanes silently.
* **Audits**, recorded under GPU-1 and SH-1 below: descriptor-heap
  indexing, the SH-1 resource and view path, fences and submission. The
  submission path's synchronous design is now PERF-2.

### 2026-11-24, third session (same container; Windows CI from here on)

Baseline: `ba0e1dd`. Each commit was pushed to the branch as it landed.
New class used below: **verified on WARP**, run on Windows CI's D3D12
software adapters (the runners have no GPU), which is not a hardware
result.

* **Windows validation on the fork** (verified):
  `.github/workflows/windows-validation.yml` runs on pushes to `claude/**`
  (see [testing.md](testing.md)). Runs 3 to 6 (`a6699d5`, `5a7e0c0`,
  `1c87a62`, `37c0ff5`) passed, about 7 minutes each with the cache: both
  clippy configurations, the workspace tests, the shader evaluator tests
  with the SDK's `dxil.dll` (10.0.26100.0) required, and the D3D12 tests
  on WARP. Every evaluator program, the new loop tests' included, passes
  Microsoft's validator (SH-7). Runs 1 and 2 were cancelled by later
  pushes, which no longer cancel a run in progress.
* **"Hardware" GPU-test runs were software** (fixed, verified on
  Windows): the test host took the system's default adapter, and on a
  machine without a GPU that is the Basic Render Driver. Skipping adapters
  flagged `DXGI_ADAPTER_FLAG_SOFTWARE` was not enough: the runner's report
  shows the primary adapter as `Microsoft Basic Render Driver (vendor
  0x1414, device 0x008c, flags 0x0)`, which DXGI's guide describes (the
  primary adapter of a machine whose display driver is not working has the
  Basic Render Driver's name and no software flag).
  `emit::renders_in_software` decides by flag, ids or name; the test host,
  the capability report and the GPU layer use it. Since run 6 the
  "hardware" half says `skipped: no hardware adapter at SM 6.6` and every
  D3D12 run is labelled software.
* **Validation that cannot pass silently** (verified on Windows): tests
  skipped DXIL validation without a word when `dxil.dll` did not load.
  `Validator::for_tests` fails them instead under
  `VICTORY_DXIL_REQUIRED=1`, which the workflow sets once it has found the
  SDK's DLL.
* **D3D12 capability report** (verified on WARP): every adapter and WARP
  with shader model, wave sizes, mesh shaders, 64-bit atomics, format
  casting and enhanced barriers. WARP on the runner: SM 6.8, waves 4,
  mesh shader tier 10, 64-bit atomics on typed resources, heap resources
  and group memory, relaxed format casting and enhanced barriers. So
  SH-1's shader side can now be validated (dxil.dll) and run (WARP) here.
* **The GPU layer names its adapter** (verified on WARP): the device log
  line and GPU failure reports name the adapter, and a device that renders
  in software without being asked to (`VICTORY_D3D12_WARP`) is a warning:
  a user whose GPU driver is broken gets the Basic Render Driver.
* **Fence waits follow D3D12** (verified off-GPU; the Windows-only event
  test passes on Windows), checked against Microsoft's reference sources
  (MicrosoftDocs/sdk-api: ID3D12Fence::SetEventOnCompletion, GetCompletedValue,
  ID3D12Device1::SetEventOnMultipleFenceCompletion; "Multi-engine
  synchronization"): a wait without an event blocks until reached, with no
  timeout (ours failed after 5 s), logged as a hang at 5, 10, 20, 40 s and
  on; a signal that reaches a waiting thread's value releases it even if a
  lower value replaces it first ("the wait will be satisfied regardless");
  and command lists' bottom-of-pipe writes to a fence's GPU address now set
  its events and release its waiters (only `Signal` did). Both fixes fail
  their tests when reverted.
* **Queue waits not carried out are counted** (verified off-GPU, no title
  evidence yet): `ID3D12CommandQueue::Wait` was a no-op, so work submitted
  after a wait for a value not yet reached runs ahead of it. Blocking the
  submitting thread could deadlock a title that signals from that thread
  later (legal: only the queue waits), so each such run-ahead is counted,
  the first 8 logged, and the frame-rate report gives the count (SYNC-1).
* **The loop limit's contract holds at every back edge** (verified
  off-GPU; DXIL-valid on Windows): a lane reaching the limit at a
  conditional back edge (`s_cbranch_scc1 loop`, the common form) took the
  loop's normal exit and ran on with partial results, stores included;
  only unconditional back edges left the program as documented. The exit
  of such loops now sends the lanes the limit stopped to the abort block.
  Found by the first loop tests (exact counts, the boundary at the limit,
  per-entry counting, both branch senses, skippable loops, a wrapped count
  at the real 2^20); all fail on the old lowering. Why the limit exists:
  a count read from memory the host has not mirrored (zeros) can make a
  loop run 2^32 times and hang the GPU. Whether it hides translation bugs
  in the title is not known: a run that counted the lanes it stops would
  say (PERF-3).
* **Submission and memory map** (audit): one global submission lock; per
  `ExecuteCommandLists`, title memory is snapshotted into host copies,
  the lists run on one host direct queue, the CPU waits `INFINITE` on a
  host fence, GPU-written ranges are copied back and the bottom-of-pipe
  writes made; title `Signal`s are then immediate. The CPU wait is
  load-bearing (copy-back, and host copies reused by the next submission),
  not redundant; overlap needs PERF-2's design. Texture hashing (PERF-5):
  textures in the texture pool are written through other views (xmem's
  `write_physical`, DirectStorage destinations), which page guards on the
  texture's own view cannot see, so a whole-content hash is the only exact
  check there today; within a submission each texture is already hashed
  once (`Batch::texture_due`). blake3's `hazmat` API would allow a
  bit-identical parallel hash; not done, since its benefit against the
  title's own threads needs a world measurement.
* **ds_swizzle and ds_bpermute** (reviewed, unchanged): LLVM's AMDGPU
  assembler (SIDefines.h, AMDGPUAsmParser.cpp) confirms GFX9-and-later
  encodings: rotate from 0xC000 (direction bit 10, size bits 9:5), FFT
  from 0xE000 (bits 4:0), which we refuse rather than misread. Their lane
  semantics, and ds_bpermute's index in wave32, need AMD's RDNA2 ISA
  guide, which this environment cannot reach (amd.com and gpuopen.com are
  refused); LLVM's ds_bpermute folding avoids indices at or above the wave
  size, so it settles nothing.

## Backlog

Priority order: P0 GPU stability, P1 shader correctness, P2 performance,
P3 runtime compatibility, P4 usability.

| Id | P | State | Item |
|----|---|-------|------|
| GPU-1 | P0 | hypothesis, narrowed | **Out-of-range heap indices as a hang cause.** Audited (second session): table bases cannot leave the heaps (`draw::State::user_value` gives 0 for a table outside the bound heaps), and tables reaching past a smaller host heap get fresh slots (`draw::State::tables`). What remains unbounded is an index the shader computes per lane (bindless indexing, waterfall loops: `emit::lower::lane_heap_index`, `fetched_index`), read through `ResourceDescriptorHeap[]` with no clamp. Check: a world run with `VICTORY_D3D12_DRED=1`; the report's `near_heaps` part says whether the faulting address lies past a heap. If it does, pass each heap's slot count as user data and clamp the per-lane indices (or send them to a null slot) in the lowering. |
| GPU-2 | P0 | verified off-GPU | Fence waits follow D3D12's rules (third session): no timeout, rewinds, bottom-of-pipe writes wake waiters. Next: a world run; a "blocking fence wait has not been reached" warning names a fence nothing signals, or a title deadlock. |
| SYNC-1 | P0 | hypothesis, instrumented | Queue waits are not carried out (`queue::wait`): work submitted after a wait for a value not yet reached runs ahead of it, which on a title's async compute would read data not yet written. Counted since the third session. Check: the frame-rate report's "ran ahead of a queue wait" count in a world run. If not zero, queues need to defer work behind unreached waits (a per-queue worker that blocks on the fence), which also starts PERF-2. |
| PERF-2 | P2 | open (design) | Every submission runs synchronously: `host::Host::execute` holds a lock, executes, signals and waits `INFINITE` before returning (decision 0007). This makes lifetimes safe by construction, but CPU and GPU never overlap, and one submission past about 2 s is a TDR. Making submission asynchronous needs per-submission dependency tracking for copy-back and fences. It is a design change to plan after measurements (`VICTORY_GPU_TIMING`, the `SUBMISSION_US` counters). |
| PERF-3 | P2 | hypothesis | Loops stopped only at `LOOP_LIMIT` (2^20 per lane) can cost about a second in a full-screen pass, the order of the world's frame times. Since the third session a lane the limit stops leaves the program at every back edge (it used to run on at conditional ones). Check: a diagnostics build in the world with `VICTORY_SHADER_LOOP_LIMIT=4096`; if frames get much faster, find the loops (`VICTORY_GPU_TIMING=1` names slow lists and their commands). Next instrument: count the lanes the limit stops (a diagnostics UAV the host reads after each submission), which also tells whether it hides translation bugs. |
| PERF-4 | P2 | hypothesis | Host copies of title buffers are custom-heap, write-back L0 (system memory; `host::Host::shared_buffer`), so a discrete GPU reads every vertex, index and constant there over PCIe. Check by timing a heavy pass with its inputs moved to a default heap. |
| PERF-5 | P2 | hypothesis | Textures in memory that cannot be write-watched are hashed whole (BLAKE3) at every submission that samples them (`texture::Staging::refresh_now`), once per submission. Exact elimination across submissions needs to see writes through every view of the texture pool's pages (third session); a bit-identical parallel hash (blake3 `hazmat`) is the other option. Check first: the frame-rate line's "textures ms, MB" and the `TexHashed` counter (diagnostics, copystat) per world frame. |
| SH-1 | P1 | not implemented, investigated | **64-bit image atomics** (README). WARP on Windows CI supports what they need (third session's capability report: 64-bit atomics on typed and heap resources, relaxed format casting), so both sides can be validated there before hardware. Shader side: `dx.op.atomicBinOp.i64` / `atomicCompareExchange.i64` on a typed UAV annotated U64 (component type 7: props `(kind \| 0x1000, 0x107)`), with the two data dwords joined and the old value split; shader flags Int64Ops (bit 20), AtomicInt64OnTypedResource (bit 27) and AtomicInt64OnHeapResource (bit 32), SFI0 0x8000, 0x400000 and 0x10000000 (from memory of DXC's headers: confirm). Host side, found in the second session: host UAVs take the title's view description, which is R32G32 for these atomics (`descriptor::host::make`), and host textures are created once with the title's format (`resource::host_texture`). A shader-only change would bind a 64-bit typed UAV declaration to an R32G32 view, which is undefined behaviour where today the draw is only dropped, so the two sides land together. That needs R64_UINT views of those textures (a castable-format resource via `CreateCommittedResource3`, which needs OPTIONS12 RelaxedFormatCastingSupported), the OPTIONS9 AtomicInt64OnTypedResourceSupported and OPTIONS11 AtomicInt64OnDescriptorHeapResourceSupported checks (refuse such pipelines without them), and a record per pipeline of which table entries need R64 views. Validate on Windows: a valu_tests-style dispatch doing image_atomic_umax with dmask 0x3 on an R64_UINT UAV, read back, clean under `VICTORY_D3D12_DEBUG=1` and `VICTORY_D3D12_GBV=1`; then `victory translate` counts. |
| SH-2 | P1 | not implemented | `tbuffer_store_format_*`: typed-format stores are refused (`buffer()`); the inverse of `format_load` (pack, round and saturate per numeric format). |
| SH-3 | P1 | not implemented | Double precision (`v_*_f64`, `v_cvt_*_f64`, f64 compares): the DXIL writer has no `double` type. |
| SH-4 | P1 | not implemented | `v_div_scale/fmas/fixup_f32` (the precise division sequence). |
| SH-5 | P1 | not implemented | `ds_permute_b32` (forward permute), ds_swizzle's rotate and FFT modes (refused; encodings confirmed from LLVM, semantics need AMD's ISA guide), `v_msad_u8`, `v_qsad/mqsad`, `image_msaa_load`, `image_*_pck*`, flat and scratch memory, `s_getreg_b32`, scalar stores and atomics, clamped integer mads. (v_permlane16/x16, v_dot2(c)_f32_f16, ds_bpermute and integer clamp on adds, subtracts and dots were done in the second session.) |
| SH-6 | P1 | not implemented | Merged geometry stages; 64-lane cross-lane code on 32-lane hosts (README). |
| SH-7 | P1 | DXIL verified | The evaluator programs, the 59 lowerings of the first session among them, pass `dxil.dll` on Windows CI (third session). Not run on a GPU: `valu_tests` cases for a few (`v_mad_u64_u32`, `v_ldexp_f32`, `v_frexp_*`) would cover WARP, and hardware where there is one. |
| SH-8 | P1 | not verified | ds_bpermute's index in wave32 (we wrap at 32) against AMD's RDNA2 ISA guide (DS_BPERMUTE_B32), which this environment cannot reach. |
| CI-1 | P1 | verified | Windows validation workflow (third session). It never claims a hardware result: hosted runners have no GPU. A hardware result needs a self-hosted runner with a GPU or a run on the owner's machine. |
| PERF-1 | P2 | open | World frames take 0.5-1.8 s (README, one machine). No measurement possible here. The existing switches (`VICTORY_GPU_TIMING`, `VICTORY_SHADER_LOOP_LIMIT`) are where to start. |

## Next session

Remote branch head at the end of the third session: see the last commit
of `claude/sharp-thompson-o8cl4c` (the docs commit that adds this
paragraph); Windows validation run 6 (`37c0ff5`) is the last code run.

1. Check the latest Windows validation run of the branch (Actions tab or
   the job summary): it must stay green, and its summary must say no
   hardware GPU was present.
2. On a machine with the game (diagnostics build): `victory translate
   <game dir> --validate --list` for the counts and most frequent
   reasons; then a world run with `VICTORY_D3D12_DRED=1`, reading the log
   for the device-removal report (GPU-1), "ran ahead of a queue wait"
   counts (SYNC-1), fence-wait warnings (GPU-2), and the frame-rate
   line's texture time (PERF-5); then one with
   `VICTORY_SHADER_LOOP_LIMIT=4096` (PERF-3).
3. The exact next implementation task: **SH-1's shader side, validated on
   WARP**: lower 64-bit `image_atomic_*` (dmask 0x3; cmpswap 0xF) to
   `atomicBinOp.i64` / `atomicCompareExchange.i64` on a U64 typed UAV,
   behind a switch the GPU layer leaves off; confirm the shader flags and
   SFI0 bits from DXC's sources (raw.githubusercontent.com is reachable:
   `include/dxc/DXIL/DxilShaderFlags.h`); evaluator tests from the ISA; a
   D3D12 test on an R64_UINT UAV in the shader test host, run on WARP by
   CI. Then the host side (castable R64_UINT views, capability checks),
   and only then the switch on.
4. If step 2 shows SYNC-1 run-aheads, deferring queue work behind
   unreached waits comes first.
