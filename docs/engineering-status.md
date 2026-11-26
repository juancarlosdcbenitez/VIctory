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
  (see [testing.md](testing.md)). Runs 3 to 10 (`a6699d5`, `5a7e0c0`,
  `1c87a62`, `37c0ff5`, `71a1129`, `54ef1ba`, `f931612`, `adfdd3f`) all
  passed, 3 to 7 minutes each with the cache: both clippy configurations,
  the workspace tests, the shader evaluator tests with the SDK's
  `dxil.dll` (10.0.26100.0) required, and the D3D12 tests on WARP. Every
  evaluator program, the loop and image atomic tests' included, passes
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
* **Image atomics went through a float4 handle** (fixed; verified
  off-GPU, DXIL-valid on Windows, run 7): every descriptor load makes a
  size-query handle annotated as the view's float4 form, and heap handles
  were cached on the first property word only, so an image atomic in the
  same block as its descriptor load (the usual case) did a 32-bit integer
  atomic through a handle annotated F32 x4. Found by the first image
  atomic tests (the evaluator now models typed texture UAVs).
* **SH-1's shader side** (verified on WARP, run 8): 64-bit image atomics
  are lowered to `atomicBinOp.i64` / `atomicCompareExchange.i64` on a U64
  heap handle, with the shader flags and SFI0 bits checked in DXC's
  sources, behind `emit::image_atomics_64`, off by default. Evaluator
  tests cover every operation against ISA references; on Windows CI they
  pass `dxil.dll`, and six of them run on WARP on an R32G32_UINT texture
  UAV and match the evaluator. Microsoft's SM 6.6 atomics spec settles the
  view format: "Typed resources used in 64-bit integer operations must be
  declared with HLSL type int64_t or uint64_t and have format
  R32G32_UINT", so the second session's plan of R64 views through
  castable resources was wrong; the title's two-dword view is the format
  D3D12 wants. The GPU layer now refuses these pipelines on devices
  without OPTIONS9/OPTIONS11 support, and a diagnostics build turns them
  on with `VICTORY_SHADER_IMAGE_ATOMICS_64=1` (implemented, not run in the
  world).
* **The first session's integer lowerings on WARP** (verified on WARP,
  run 10): 24-bit mul_hi, add/sub_nc_i32, mul_lo_i32, the 16-bit mads,
  cvt_pk_*16_*32, the sad family, lerp_u8, v_mad_u64_u32 and
  v_mad_i64_i32, each for 16 input sets: all four stored dwords match the
  evaluator (`valu_tests::integer_lowerings_agree_with_the_evaluator_on_the_host`).
* **ds_swizzle and ds_bpermute** (reviewed, unchanged): LLVM's AMDGPU
  assembler (SIDefines.h, AMDGPUAsmParser.cpp) confirms GFX9-and-later
  encodings: rotate from 0xC000 (direction bit 10, size bits 9:5), FFT
  from 0xE000 (bits 4:0), which we refuse rather than misread. Their lane
  semantics, and ds_bpermute's index in wave32, need AMD's RDNA2 ISA
  guide, which this environment cannot reach (amd.com and gpuopen.com are
  refused); LLVM's ds_bpermute folding avoids indices at or above the wave
  size, so it settles nothing.

### 2026-11-25, fourth session (same container; Windows CI)

Baseline: `c594b90`, Windows runs 3 to 10 green. Each commit was pushed to
the branch as it landed and ran on Windows CI: runs 11 (`6b2eab2`), 12
(`b540592`), 13 (`b90281f`), 14 (`b4d753f`) and 15 (`fe69655`) passed;
16 (`c8b6701`) and 17 (`b4b6a7b`) failed in one test, the new float
differential (SH-7 below), the same 18 disagreements both times, and their
diagnostic builds were skipped as designed. The last green run's package
is `victory-diag-0.1.0-fe69655159b3-windows-x64` (run 15, 10.7 MB).
Nothing here ran on a hardware GPU or with the game.

* **GPU-1 and PERF-3 probes** (implemented; verified on WARP, run 11; not
  run in the world): with the diagnostics feature and
  `VICTORY_SHADER_PROBES=1`, translated shaders compare every heap index
  they use with the bound heap's slot count (root constants) and count
  and record, in a root UAV the GPU layer reads after each submission,
  each index at or past it and each lane the loop limit stops (shader id,
  site, index, slot count; a 64-record ring). Nothing is clamped or
  substituted. The checks are added after the shader is lowered, by
  moving each noted block's terminator behind them
  (`Module::move_terminator`), so the lowering is unchanged without
  probes. Evaluator tests cover in-range, boundary, past and wrapped
  (address arithmetic below the table) indices and the ring; on WARP a
  store through heap index 2 with the constants saying 3 or 2 slots
  leaves the probe buffer word for word as the evaluator does, and lands
  at index 2 both times. GPU failure reports add what the buffer holds.
  The shader cache keys on the switch.
* **Queue waits hold work back** (SYNC-1; implemented; the model verified
  off-GPU, its rules checked against a real D3D12 queue on WARP, run 12;
  no title evidence): `ID3D12CommandQueue::Wait` was a no-op. Now work
  and signals submitted behind a wait not reached are held, in order,
  without blocking the submitting thread, and a queue worker thread runs
  them once fence moves reach their waits (`schedule::Schedule`, a model
  tested on its own: rewinds, chains across queues, work submitted while
  released work runs, waits only held work could satisfy). On WARP a
  D3D12 queue's Wait holds a copy until a CPU signal, and a wait reached
  then rewound (7 then 3, for 5) is satisfied, which is the model's rule.
  Presents and page mappings on a held queue still run at once (they read
  title memory at the call) and are counted; a title list reset while a
  held submission holds it gets a fresh host list. Waits lasting 5 s or
  more are logged with the waits only held work could satisfy. With the
  diagnostics feature, `VICTORY_QUEUE_WAITS=0` restores the old
  behaviour to compare.
* **Diagnostic build download** (verified: run 13 built and packaged it;
  no game run with it): the Windows workflow's second job builds
  `victory.exe` as the README's Debugging section does, once the
  validation job has passed, and keeps it 14 days as
  `victory-diag-<version>-<commit>-windows-x64` with symbols, checksums,
  build information, the license, the notices of the 83 crates compiled
  in (`cargo xtask notices`) and two guides. It runs the asset policy
  check, checks that `victory --version` names the build
  (`<version>+diag.<commit>`, also the run summary's first line), and
  fails on any file not named. Run 13's package:
  `victory-diag-0.1.0-b90281f9eace-windows-x64`, 10.6 MB.
  [diagnostic-build.md](diagnostic-build.md) has the exact commands to
  collect logs; [hardware-validation.md](hardware-validation.md) is the
  first hardware procedure (a baseline, a translation census and twelve
  stages, one diagnostic variable per
  run, which log line settles which backlog item). Neither assumes anyone
  has a suitable machine or the game.
* **Texture checks side by side** (PERF-5; implemented; equivalence
  tested; frame effect not measured): the textures a submission checks
  between two steps that write title memory are independent, so they are
  checked on up to 8 threads (from 2 MiB of texture memory in all), and a
  hash of 2 MiB or more is split along BLAKE3's own tree
  (`victory_core::parallel::blake3`, equal to `blake3::hash` bit for bit:
  tested for 13 lengths on 1 to 8 threads). Write-watch answers do not
  depend on the order textures ask, so nothing else changes.
  `hashing_throughput` (CPU throughput, not a frame rate), 4 threads:

  | Machine | 4 MiB split | 16 MiB split | 64 MiB split | 48 x 1 MiB side by side | 12 x 4 MiB side by side |
  |---------|-------------|--------------|--------------|-------------------------|-------------------------|
  | this Linux container | 1.7x | 2.8x | 1.9x | 2.9x | 3.5x |
  | Windows runner, run 14 | 1.2x | 2.0x | 2.3x | 2.4x | 2.3x |

  Inputs below 2 MiB stay on one thread (1.0x on both). Whether texture
  checks matter in a world frame is still unknown: the frame line's
  "textures ms" in a world run says.
* **SH-1 under contention** (verified on the evaluator and on WARP, run
  15):
  32 lanes doing one 64-bit image atomic each on one texel: adds of
  0xFFFF_FFFF carrying into the high dword (each partial sum handed out
  exactly once), a cmpswap one lane wins, umax. The checks hold in any
  lane order, and refuse a dropped carry, a lost update, two winners.
  Still off by default.
* **SH-7: the float lowerings on WARP** (failing, runs 16 and 17; being
  fixed, see the next item): 43 lowerings (arithmetic,
  fma and mad, min/max/med3, ldexp, frexp, rounding, the integer, f16 and
  packed conversions, the cube-map ops, and the approximate ones) for 48
  input sets of special and ordinary values, every stored dword against
  the evaluator, with each difference D3D leaves to devices named
  (`Leeway`: NaN payloads, denormals flushed, zero signs from min/max,
  FMad fused or not, approximate results within tolerance) and any other
  failing the test.
* **CPU/GPU overlap** (PERF-2; analysed, not changed): what keeps each
  submission synchronous, from the code: (1) host copies of title memory
  are single-buffered, so preparing the next submission would overwrite
  data the GPU may still read; (2) preparation copies whole dirty pages,
  which would overwrite GPU-written ranges not yet copied back (their
  pages are dirty when the title wrote elsewhere on them); (3) copy-back
  and bottom-of-pipe writes, which tell the title the GPU is done, need
  the GPU done; (4) host resources and descriptor slots a submission uses
  are released right after it, which is safe only because it finished;
  (5) the probe buffer is read after each submission. The queue worker
  now exists and is where completion would move; the order to do it in
  is in PERF-2. None of it can be shown correct without a GPU that runs
  the title, so the synchronization stays.
* **SH-2, the part that converts nothing** (implemented; verified on the
  evaluator; DXIL-valid on Windows CI): typed stores with the format in
  the instruction were all refused. Those whose channels are all 32-bit
  UINT, SINT or FLOAT, all stored, put each register in memory unchanged,
  so they are now the plain stores they amount to; four lanes' NaN
  payloads, denormals, -0 and arbitrary words land unchanged through
  formats 20, 62, 63, 64, 74 and 77 (LLVM's `UfmtGFX10` numbers), and
  every format that converts, a partial store and d16 data stay refused
  with the reason named. How often the title uses these is not known
  here (`victory translate --list` on the game's shaders says).
* **Left alone for lack of AMD's ISA guide** (amd.com and gpuopen.com are
  still refused): `v_div_scale/fmas/fixup_f32`, `ds_permute_b32`,
  ds_swizzle's rotate and FFT modes, `v_msad_u8`, typed-format stores;
  they stay explicitly refused, as the probe of the lowering's float ops
  showed (`v_div_fixup_f32` and `v_cvt_off_f32_i4` refused, 46 others
  lowered).

### 2026-11-25, fifth session (same container; Windows CI)

Baseline: `b4b6a7b` plus a docs checkpoint; runs 16 and 17 red in the
float differential. Microsoft's D3D11.3 functional specification (v1.17)
was read from the DirectX-Specs repository on GitHub (`microsoft.github.io`
is not reachable from this environment); DXIL.rst from the
DirectXShaderCompiler repository.

* **f16 results rounded toward zero on every D3D device** (defect, fixed;
  verified on the evaluator and on WARP, run 18). D3D's `f32tof16` (DXIL
  `legacyF32ToF16`) "Follows D3D rules for floating point conversion"
  (22.13.2): round toward zero, and an overflow gives "the maximum
  representable (appropriately signed) value ..., NOT including signed
  infinity" (3.2.2). The lowering used it for every f16 result the ISA
  rounds to nearest even: `v_cvt_f16_f32`, f16 and packed f16 arithmetic,
  `v_fma_mix`, packed f16 interpolation. The evaluator had modelled the op
  as nearest-even, which hid it; runs 16 and 17 showed it (WARP: f32::MAX
  to `0x7BFF`, not infinity). Now `f16_nearest_even` converts with
  operations D3D defines exactly (integer arithmetic on the bits; for half
  denormals a power-of-two scale and `round_ne`), and the evaluator models
  the DXIL op as the specification defines it, so three existing evaluator
  tests failed against the old lowering and pass against the new. The d16
  texture and buffer returns keep D3D's conversion (a shortcut now
  documented: the texture unit's rounding is not documented here).
* **Integer-to-float conversions not exact on every device** (defect,
  fixed; verified on WARP, run 18): for integers too large to convert
  exactly, nearest-even rounding is "strongly recommended but not
  required" (22.13.5-6), and WARP rounds `0xFFFFFF6B` up to 2^32.
  `v_cvt_f32_u32` and `v_cvt_f32_i32` now build such values from their
  bits.
* **The differential accepts only what the specifications permit**: NaN
  bits (3.1.3.2), compared per half for f16; denormals flushed on input or
  output (3.1.3.2, never for halves: 3.1.5); zero signs from min/max only;
  `FMad` fused or not (DXIL.rst); approximate operations within their
  chapter 22 bounds (rcp, rsq, exp 2^-21 relative; sqrt 1 ULP; log 2^-21
  absolute on [0.5, 2] else relative; sin and cos 0.0008 absolute within
  100 pi radians). Run 18: no other difference; the three conversions and
  `v_add/sub/mul/fma_f16` agree with the evaluator to the bit.
* **Probes now check an access before it fetches its descriptor**
  (defect, fixed; verified on the evaluator; WARP: run 19). The fourth
  session's checks ran at the end of each block, after the accesses, so
  an access whose descriptor fetch faults (GPU-1's case) left no record.
  Each access is now checked just before its `createHandleFromHeap`, and
  the record is followed by a global UAV fence before the fetch is issued.
  Whether a record reaches memory before a device is removed still depends
  on the device. A test shows the record exists when the evaluator stops
  at the faulting access, and fails with the old placement.
* **Audit of the fourth session** (three more defects, fixed; the rest
  reviewed without change):
  * *Thread creation per call* (PERF-5's parallel texture checks and split
    hashes): scoped threads started on every call cost 58, 85 and 217
    microseconds for 2, 4 and 8 threads here, taken whenever a group's
    textures added up to 2 MiB, even unchanged ones. Now a pool started
    once, workers parked between calls (3 to 8 microseconds a call here;
    split and side-by-side hashing 3.1x to 3.6x on 4 threads, CPU
    throughput only). One call uses the pool at a time; calls meanwhile
    run on their caller's thread.
  * *Queue order* (SYNC-1): a submission run at once released the queue
    lock before running, so a Wait and work submitted by another thread
    in that gap could run first on the worker. Now running at once claims
    the queue until done, and what arrives meanwhile is held in order.
  * *Probe-induced drops undiagnosed* (GPU-1): probes add 4 root
    signature dwords; a pipeline already near D3D12's 64 is refused in
    probe runs only. The log now names it (`probes do not fit this
    pipeline's root signature`).
  * Reviewed: the probe buffer is read after the submission's fence wait
    from a CPU-visible heap (visible); probe counts are 32-bit and wrap
    after 2^32 counted accesses in a submission (documented); the queue
    worker drops its lock before fences or execution, re-checks after
    each run, and needs no shutdown (Windows ends other threads before
    DLLs detach); a title list reset while held work holds it gets a new
    host list; texture checks touch only their own staging buffers and
    state, and write-watch answers do not depend on order; SH-2's typed
    stores reuse the raw-store path unchanged (addressing, index stride,
    guard, alignment, bounds), which they equal for those formats.
* **Later in the session** (each pushed and run on Windows CI):
  * GPU-1: a probed pixel shader passes `dxil.dll` (run 27): the global
    UAV fence after each probe record is accepted outside compute stages,
    which every probed graphics pipeline needs.
  * PERF-2: `Retire` and `InFlight` against a real D3D12 queue on WARP
    (run 25, passed): eight submissions in flight with their allocators
    reused; a read waits for exactly the newest writer of its range.
  * SH-7: packed f16 arithmetic against the evaluator on WARP (run 25).
  * PERF-5: the hashing benchmark measures 2, 4 and 8 threads, each for at
    least 50 ms. Windows runner, run 27 (4 threads; CPU throughput, not a
    frame rate): a call on the pool costs 0.9-1.3 microseconds; split
    hashes 1.0x at 1 MiB (below the 2 MiB threshold, by design), 1.8x and
    2.6x at 4 MiB on 2 and 4 threads, 1.8x and 3.0x at 16 MiB, 2.0x and
    3.0x at 64 MiB; 48 x 1 MiB side by side 2.0x and 3.1x, 12 x 4 MiB 1.9x
    and 2.7x.
  * SYNC-1: the 5 s frame line reports the ms per frame title threads
    waited for each other's submissions (the submission lock), so a
    hardware run shows what running submissions one at a time costs
    (`4bc6e42`, measurement only).
  * CI-1: `VICTORY_WARP_REQUIRED=1` on the Windows workflow turns every
    skip of a WARP run into a failure (no WARP device; WARP without the
    64-bit typed atomics the SH-1 tests need). The GPU layer's queue and
    in-flight tests print nothing on CI (captured output), so before this
    a skip there would have passed unseen (`3d3bd7d`).
* **CI and the diagnostic package.** Runs 18-20, 22, 25 and 27 passed both
  jobs; 21, 23, 24 and 26 were replaced while waiting by newer pushes
  (cancelled, never run), and every commit they held is in a later run
  that passed. Run 27 (`de302ed`) built
  `victory-diag-0.1.0-de302ed0a8c7-windows-x64` (10,684,135 bytes; zip
  SHA-256 `c3d1452ca5c2f2023e115ebd0bb7a99790863dd04e1ad235e114f675bf2742cc`;
  `victory.exe` `58b6d7ca86ced684bc879c303df4456c4ef4286b11f2bc05cf10a7a4ff2850d2`,
  `victory.pdb` `e5c1000a029e88369b466ab18d081aa5295a446042f453ec5598ad4b4d0feb56`),
  after the job checked that the package holds exactly its nine files and
  that `victory.exe --version` names the commit; run 25's (`39b7967`) is
  `victory-diag-0.1.0-39b7967d92c8-windows-x64` (`victory.exe`
  `b14a13b87455df30c42c99b8a053121276fece807719f985325ee7342ac7df7d`). The
  runner's adapters: the Basic Render Driver twice and WARP, all software
  (SM 6.8, 64-bit typed atomics); no hardware GPU, no game run.

### 2026-11-26, sixth session (same container; Windows CI)

Baseline: `129c61a`, local and remote equal; the fifth session ended in
the middle of the `cvt_norm16` fix (uncommitted: the fix, its regression
test, the new differential and a scratch test, which was removed).

* **`v_cvt_norm_*16_f16` shifted by negative amounts** (fixed, `2d7706a`;
  run 31). The lowering shifted the scaled mantissa by 25 minus the f16
  exponent, negative from 2048 up: a shift LLVM leaves undefined. The
  saturating select discarded the value, so WARP gave the right result
  (the device test of every 251st half passed), but the IR leaned on an
  undefined value, which the evaluator refuses. Halves of 1 and more now
  shift by 24. The regression test fails on the old lowering at 2048
  (`LShr of i32 by 4294967295`) and passes now.
* **The remaining f16 and mixed-precision lowerings against WARP**
  (`89ed258`; run 31 passed, `dxil.dll` required): f16 min/max/min3/max3/
  med3, sqrt, rsq, rcp, log, exp, sin, cos, fract and the roundings,
  conversions between halves and 16-bit integers (plain and normalised),
  `v_fma_mix*`, `v_dot2_f32_f16`, `v_mad_legacy_f32`, `v_rcp_iflag_f32`.
  f16 leeways: NaN bits, zero signs of min/max (D3D11.3 3.1.3.2), and for
  approximate operations the f32 operation's chapter 22 bound plus one
  step of halves; everything else to the bit. No disagreement outside
  them. (The per-operation counts are in the run's log artifact, whose
  storage host this environment cannot reach.)
* **Held submissions read what the title list owned** (two defects, fixed,
  Windows tests; runs pending at the time of writing). A submission held
  behind a queue wait keeps its host list past the title list's reset,
  but (1) the dynamic descriptor slots its fresh constant-buffer tables
  were made in belonged to the title list's draw state and went back to
  the free pool at the reset, where other lists could make their tables
  before the held work read them (wrong constants, or a view of another
  kind read as a constant buffer: a device removal on AMD, #59), and (2)
  the list-owned upload memory (`allocate_upload`, where Grand Theft Auto
  VI puts BVH instance descriptions) was freed at the reset or release
  before the held work copied it. Now the host list holds both
  (`HostList::hold_slots`, `keep_memory`) until it is reset or dropped
  (`9ef7602`, `839c23e`, `a6557b7`). The fifth session's audit had
  missed both. The slot test was not run against the old ownership
  (reasoned: the reset's drop of the state gave the slot back).
* **Asynchronous queues, experimental** (`6aeae73`, diagnostics builds,
  `VICTORY_QUEUE_ASYNC=1`, off by default): every submission, signal,
  present and page mapping runs on the queue worker in each queue's order,
  so a title thread returns from `ExecuteCommandLists` at once and waits
  for the GPU only in its fence waits. Presents and page mappings take
  what they read at the call. The worker's `execute` is unchanged, one
  submission at a time and waiting, so the copy invariants hold as today
  (architecture.md). Tested on the schedule model and the queues
  (signals and presents in order behind a wait); not run on a GPU or with
  the game. Smallest step toward overlap that keeps every invariant:
  title CPU work overlaps the GPU; the worker's own preparation does not.

## Backlog

Priority order: P0 GPU stability, P1 shader correctness, P2 performance,
P3 runtime compatibility, P4 usability.

| Id | P | State | Item |
|----|---|-------|------|
| GPU-1 | P0 | hypothesis, instrumented | **Out-of-range heap indices as a hang cause.** Audited (second session): table bases cannot leave the heaps (`draw::State::user_value` gives 0 for a table outside the bound heaps), and tables reaching past a smaller host heap get fresh slots (`draw::State::tables`). What remains unbounded is an index the shader computes per lane (bindless indexing, waterfall loops: `emit::lower::lane_heap_index`, `fetched_index`), read through `ResourceDescriptorHeap[]` with no clamp. Since the fourth session, probes (`VICTORY_SHADER_PROBES=1`, diagnostics) record every such index at or past the bound heap's slots, with the shader, without clamping. Check: hardware-validation.md stages 4 and 5. Records name the shaders whose index computation to fix; none, with DRED placing a fault past a heap, would mean an index inside the slots naming the wrong descriptor, which probes do not see. Clamping stays out: it would hide the defect. |
| GPU-2 | P0 | verified off-GPU | Fence waits follow D3D12's rules (third session): no timeout, rewinds, bottom-of-pipe writes wake waiters. Next: a world run; a "blocking fence wait has not been reached" warning names a fence nothing signals, or a title deadlock. |
| SYNC-1 | P0 | implemented, not run in the world | Queue waits were not carried out (work behind a wait not reached ran ahead of it). Since the fourth session the work is held and a queue worker runs it once the fence gets there (`schedule`, `queue`; the rules checked on WARP). Sixth session: held work keeps the descriptor slots and list-owned upload memory it reads past the title list's reset or release (two defects fixed). Left running ahead, and counted: presents and page mappings on a held queue. Check: hardware-validation.md stage 6 (held counts after the frame lines; waits held 5 s or more; `VICTORY_QUEUE_WAITS=0` to compare). A wait that only held work could satisfy would hang: the log names it. |
| PERF-2 | P2 | open; invariants written, two components built | Every submission runs synchronously: `host::Host::execute` holds a lock, executes, signals and waits `INFINITE` before returning (decision 0007). This makes lifetimes safe by construction, but CPU and GPU never overlap, and one submission past about 2 s is a TDR. The fourth session listed what depends on it (session log): single-buffered host copies, dirty-page copies over GPU-written ranges not yet copied back, completion writes, releases right after a submission, the probe buffer. Order: (1) a registry of in-flight GPU-written ranges that preparation consults (wait for that submission only, or skip the range); (2) host copies versioned per submission, or copied on write while in flight; (3) host resources and descriptor slots retired by host fence value; (4) completion (copy-back, bottom-of-pipe writes, probe reads) moved to the queue worker; (5) only then `execute` returning before the GPU finishes. Each step needs a world run on a GPU to show it changes nothing but timing; measure first (`VICTORY_GPU_TIMING`, the frame line's CPU and GPU times). Fifth session: the invariants are in architecture.md (C1-C4, O1-O3, what keeps each today and with overlap); `inflight::Retire` (O2) and `inflight::InFlight` (C3) exist, tested on their own and on WARP (eight submissions in flight with allocators reused; a read waiting for exactly the newest writer of its range); not used yet. The frame line now reports the time title threads wait for each other's submissions. Sixth session: asynchronous queues (`VICTORY_QUEUE_ASYNC=1`, experimental, off): everything runs on the queue worker in queue order, so title CPU work overlaps the GPU; next, a GPU run (hardware-validation.md stage 9) before anything overlaps the worker's own preparation. |
| PERF-3 | P2 | hypothesis, instrumented | Loops stopped only at `LOOP_LIMIT` (2^20 per lane) can cost about a second in a full-screen pass, the order of the world's frame times. Since the third session a lane the limit stops leaves the program at every back edge; since the fourth, probes count and record such lanes with their shader. Check: hardware-validation.md stages 5 and 7 (`VICTORY_SHADER_LOOP_LIMIT=4096`); if frames get much faster, `VICTORY_GPU_TIMING=1` names the slow lists. Recorded stops in shaders whose loops should end tell whether the limit hides translation bugs. |
| PERF-4 | P2 | hypothesis | Host copies of title buffers are custom-heap, write-back L0 (system memory; `host::Host::shared_buffer`), so a discrete GPU reads every vertex, index and constant there over PCIe. Check by timing a heavy pass with its inputs moved to a default heap. |
| PERF-5 | P2 | parallel (equivalent), effect in a frame unmeasured | Textures in memory that cannot be write-watched are hashed whole (BLAKE3) at every submission that samples them, once per submission. Since the fourth session a submission's textures are checked side by side and big hashes split along BLAKE3's tree (bit-identical; on a pool of parked threads since the fifth session, which cut a call's overhead from 58-217 to 3-8 microseconds here). Skipping hashes across submissions would need to see writes through every view of the texture pool's pages (third session). Check: the frame line's "textures ms, MB" in a world run (hardware-validation.md stage 9). |
| SH-1 | P1 | verified on WARP, off by default | **64-bit image atomics** (README). Shader side done in the third session behind `emit::image_atomics_64` (evaluator tests from the ISA; `dxil.dll` on Windows; six operations on WARP on an R32G32_UINT UAV, `valu_tests::wide_image_atomics_on_the_host`). The view format is R32G32_UINT by Microsoft's SM 6.6 atomics spec (no R64 views are needed; the second session's plan was wrong), which the title's two-dword views should already be. The GPU layer refuses such pipelines on devices without OPTIONS9 AtomicInt64OnTypedResourceSupported and OPTIONS11 AtomicInt64OnDescriptorHeapResourceSupported. Fourth session: 32 lanes contending on one texel (add with carries, cmpswap, umax) pass order-independent checks on the evaluator and on WARP (run 15, 8 dispatches each). Next: hardware-validation.md stage 8 (the switch alone, then with `VICTORY_D3D12_DEBUG=1`, then GBV: the debug layer names a view whose format does not fit), then `victory translate` counts; if clean, turn it on by default. |
| SH-2 | P1 | part implemented | `tbuffer_store_format_*` (the format in the instruction): since the fourth session, formats whose channels are all 32-bit UINT, SINT or FLOAT, all stored, are lowered as the plain stores they are (evaluator tests; DXIL-valid on Windows). Every other format is refused with its reason: the inverse of `format_load` (pack, round and saturate per numeric format) and what is written for channels a store leaves out need AMD's ISA guide. |
| SH-3 | P1 | not implemented | Double precision (`v_*_f64`, `v_cvt_*_f64`, f64 compares): the DXIL writer has no `double` type. |
| SH-4 | P1 | not implemented | `v_div_scale/fmas/fixup_f32` (the precise division sequence). |
| SH-5 | P1 | not implemented | `ds_permute_b32` (forward permute), ds_swizzle's rotate and FFT modes (refused; encodings confirmed from LLVM, semantics need AMD's ISA guide), `v_msad_u8`, `v_qsad/mqsad`, `image_msaa_load`, `image_*_pck*`, flat and scratch memory, `s_getreg_b32`, scalar stores and atomics, clamped integer mads. (v_permlane16/x16, v_dot2(c)_f32_f16, ds_bpermute and integer clamp on adds, subtracts and dots were done in the second session.) |
| SH-6 | P1 | not implemented | Merged geometry stages; 64-lane cross-lane code on 32-lane hosts (README). |
| SH-7 | P1 | DXIL verified; integers, floats, f16, packed f16 and mixed precision on WARP | The evaluator programs pass `dxil.dll` on Windows CI; the integer lowerings agree with WARP (third session), and since the fifth 43 float lowerings, f16 arithmetic and packed f16 arithmetic agree to the bit but where the D3D11.3 specification leaves results to the device (NaN bits, denormal flushing, min/max zero signs, FMad fusion, chapter 22 precision). The comparison found two lowering defects, both fixed: f16 results rounded toward zero by D3D's `f32tof16`, and integer-to-float conversions not exact on every device. Left: everything on hardware; d16 texture returns keep D3D's conversion (the texture unit's rounding is not documented here). |
| SH-8 | P1 | not verified | ds_bpermute's index in wave32 (we wrap at 32) against AMD's RDNA2 ISA guide (DS_BPERMUTE_B32), which this environment cannot reach. |
| CI-1 | P1 | verified | Windows validation workflow (third session). It never claims a hardware result: hosted runners have no GPU. A hardware result needs a self-hosted runner with a GPU or a run on a machine with one. Since the fourth session it also builds the diagnostic download once the checks pass, and runs the hashing benchmark. Since the fifth, a test that cannot run its WARP half fails there instead of skipping (`VICTORY_WARP_REQUIRED=1`). |
| HW-1 | P0 | blocked (no machine) | The first hardware validation: [hardware-validation.md](hardware-validation.md): a baseline against `ffe310e`, a translation census and twelve stages on a Windows machine with a hardware D3D12 GPU and game files one may use, with the diagnostic download. Nothing in this repository's CI can stand in for it. |
| PERF-1 | P2 | open | World frames take 0.5-1.8 s (README, one machine). No measurement possible here. The existing switches (`VICTORY_GPU_TIMING`, `VICTORY_SHADER_LOOP_LIMIT`) are where to start. |

## Next session

Remote branch head at the end of the fifth session: the last commit of
`claude/sharp-thompson-o8cl4c` (the docs commit that adds this
paragraph). Windows validation runs 18-20, 22, 25 and 27 passed, each with
its diagnostic package; the latest verified one is run 27's
(`victory-diag-0.1.0-de302ed0a8c7-windows-x64`, checksums in the
fifth-session entry). Runs for `4bc6e42` and `3d3bd7d` had not finished
when this was written.

1. Check the latest Windows validation run: both jobs green, the summary
   saying no hardware GPU was present, the float, f16 and packed f16
   device tests run on WARP (not skipped), and a `victory-diag-...`
   artifact for the head commit.
2. Whoever has a Windows machine with a hardware D3D12 GPU and game files
   they may use: [hardware-validation.md](hardware-validation.md) with
   that artifact: the baseline against `ffe310e` and the translation
   census first, then the stages one variable at a time. They settle the
   open P0 items: the world hang (3, 4), GPU-1 (5), SYNC-1 (6), PERF-3 (7),
   SH-1 (8), the frame and memory baselines (9, 10), playability (12).
   Nothing in this repository's CI can stand in for it (HW-1).
3. Without such a machine, the next implementation tasks, in order:
   * PERF-2: check C3 in diagnostics builds with `inflight::InFlight`
     (every GPU-written range recorded per submission, every range
     preparation reads checked against it), so the bookkeeping is proven
     on real workloads before anything relies on it; then per-submission
     versions of host copies (C1). No step that lets submissions overlap
     before a world run measures the CPU and GPU split.
   * With the game's shaders at hand (the census), the most frequent
     refusal reasons decide the next lowering; those needing AMD's ISA
     guide (SH-2's converting formats, SH-4, SH-5) wait for it.
   * The d16 texture returns' rounding (D3D's toward zero today), once a
     hardware run can compare them with nearest-even.
