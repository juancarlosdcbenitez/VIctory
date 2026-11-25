# Testing

What runs where, and the commands. No test needs game data unless it says
so; those read paths from environment variables and skip without them.

## Layers

| Layer | Where | What it checks |
|-------|-------|----------------|
| Unit tests | any host (most crates), Windows (all) | parsers, layouts, codecs, IR, structurisation, resource tracing |
| Shader evaluator tests (`victory-shader`, `emit::eval_tests`) | any host | what a lowering computes: synthetic GFX10 programs lowered to DXIL and run by a test-only interpreter of the emitted module (`emit::bitcode::eval`), against references written from AMD's RDNA2 ISA |
| DXIL validation | Windows with `dxil.dll` (Windows SDK, `VICTORY_DXIL`; required with `VICTORY_DXIL_REQUIRED=1`) | every program the evaluator tests lowers (`synth::lower_valu`), the VALU GPU tests, and the container tests in `emit` |
| GPU tests (`emit::valu_tests`, `emit::waterfall_tests`) | Windows, hardware GPU and/or WARP with SM 6.6 | the translated shader on host D3D12; each run prints its adapter and whether it renders in software |
| Windows validation CI (`.github/workflows/windows-validation.yml`) | hosted Windows runners, on pushes to `claude/**` | all of the above that needs no game data; D3D12 on software adapters only |
| Differential tests (`emit::diff_tests`) | Windows with the console SDK's `dxc.exe` (`VICTORY_CONSOLE_DXC`) | console-compiled HLSL translated, against the same HLSL compiled for PC |
| Data tests (BCPack, XMA, loader probes) | wherever the user's files are | real streams and executables, by properties only |
| Policy | any host | `cargo xtask policy`: nothing from the game in the repository |

The evaluator makes the choices DXIL leaves to drivers one way (FMax/FMin
of two zeros give the first operand, FMad is unfused, float-to-integer
conversions saturate, fp32 denormals are kept) and models what D3D
specifies as specified: `legacyF32ToF16` rounds toward zero and stops an
overflow at the largest finite half (D3D11.3 22.13.2, 3.2.2; until the
fifth session it was modelled as nearest-even, which hid a lowering
defect) and refuses what LLVM leaves undefined (shifts by
the bit width, division by zero, branches on undefined values), so a
lowering that leans on either fails there. It runs one thread, or a whole
wave in lockstep (`Env::lanes`). Wave operations are evaluated across the
lanes with all of them active, and lanes that branch different ways are
an error, since divergence is not modelled. The first evaluator tests
replay the cases `valu_tests` checks on hardware, so its choices are known
to agree with real drivers on those.

## Commands

On Windows (what CI runs, `.github/workflows/ci.yml`):

```powershell
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
```

### Windows validation CI

`.github/workflows/windows-validation.yml` runs on every push to a
`claude/**` branch (not for documentation-only changes) and by hand. A
newer push waits for a run in progress rather than cancelling it. Steps:
both clippy configurations; the tests built; tests in crates that never
open a D3D12 device, with victory-shader's evaluator tests (`dxil.dll`
required); the translated-shader D3D12 tests; the GPU layer, runtime and
app tests, after one that names the GPU layer's adapter; the hashing
benchmark (`hashing_throughput`, release build: CPU throughput of this
runner, not a frame rate); the capability report. Together the test steps
are `cargo test --workspace --locked`. The job summary lists counts,
adapters and the benchmark's lines; logs are kept 14 days as the
`windows-validation-logs` artifact.

Once that job passes, a second job, `diagnostic-build`, builds
`victory.exe` with the `diag` profile and the `diagnostics` feature and
keeps it 14 days as the artifact `victory-diag-<version>-<commit>-windows-x64`
with its symbols, checksums, build information, the license, the
third-party notices (`cargo xtask notices`) and the two guides
([diagnostic-build.md](diagnostic-build.md),
[hardware-validation.md](hardware-validation.md)). It runs the asset
policy check first, checks that the executable names its build
(`--version`), and fails if the package holds any file not named. No game
is run with it.

Hosted runners have no GPU. Their adapters are the Basic Render Driver,
twice (once as the primary adapter, without the software flag), and
WARP; `emit::renders_in_software` classes all three as software, so the
GPU tests' "hardware" half is skipped there and the summary says that no
hardware GPU was present. Nothing CI runs is a hardware GPU result.

The adapters' D3D12 capabilities (shader model, wave sizes, mesh shaders,
64-bit atomics, format casting, enhanced barriers) on any Windows
machine:

```powershell
cargo test --locked -p victory-shader capability_report -- --ignored --nocapture
```

`VICTORY_WARP=1` makes the GPU tests' "hardware" half run on WARP too.

`VICTORY_WARP_REQUIRED=1` (set by the Windows validation workflow, whose
runners always have WARP) turns every skip of a WARP run into a failure:
no WARP device for the shader GPU tests (`valu_tests::test_host`), the
pipeline-creation test or the GPU layer's queue and in-flight tests
(`victory-gpu`'s `warp::device`), and WARP without 64-bit typed atomics
for the SH-1 tests. A test that passed on CI therefore ran on WARP; the
GPU layer's tests do not print their adapter (their output is captured).

On Linux, the portable crates' tests:

```bash
cargo test --locked -p victory-shader -p victory-formats -p victory-core -p victory-pipeline -p victory-game -p victory-gpu -p xtask
```

`victory-pe` and `victory-loader` build there, but their tests parse
their own test binary as a PE image and fail on ELF; `victory-hle` and
`apps/victory` do not build outside Windows.

The Windows code can be type-checked and linted from Linux. blake3's
build script compiles C and assembly with MSVC for that target; the
environment variable below makes it use its Rust backend for this check
only (the repository is unchanged, nothing is linked):

```bash
CARGO_FEATURE_PURE=1 cargo clippy --workspace --all-targets --locked --target x86_64-pc-windows-msvc -- -D warnings
```

```bash
CARGO_FEATURE_PURE=1 cargo clippy --workspace --all-targets --all-features --locked --target x86_64-pc-windows-msvc -- -D warnings
```

The shader evaluator tests alone:

```bash
cargo test --locked -p victory-shader eval_tests
```

## Writing a shader test

`emit::synth` builds a one-thread compute program around a body of GFX10
code: root constants arrive in `s0..s3` and are copied to `v1..v4`;
`v0`, `v5`, `v6` and `v7` are stored as the result (a 64-bit result in
`v[6:7]`); `s[4:5]` and `s[8:11]` belong to the harness. Encoders
(`vop1`, `vop2`, `vop3`, `vop3b`, `vop3p`, `sop1`, `sop2`, `sopk`,
`sopc`) and `gfx10::opcode(encoding, name)` write the body;
`eval_cases(body, inputs)` lowers and runs it per input set and panics
with the reason if either step fails. Compare with a reference written
from the ISA description, not with what the lowering does.

Branches are `sopp(op, offset)` (a dword offset from the next
instruction). For loops, `eval_tests::eval_cases_limited(body, limit,
inputs)` lowers with the loop limit at `limit` on the test's thread
(`emit::TEST_LOOP_LIMIT`), so a test reaches the limit in a few
iterations; a lane the limit stops leaves the program, so nothing is
stored (`NOT_STORED` stays).

For image atomics, `synth::image_program(body)` adds a typed texture UAV
at heap index 1, its descriptor in `s[12:19]` (`synth::mimg` encodes the
instructions), and `synth::eval_image(lowered, inputs, dims, texels)`
runs it on the evaluator, which models typed texture UAVs for atomics
(texels of 4 bytes, or 8 for a 64-bit component type). 64-bit image
atomics lower only with `emit::TEST_IMAGE_ATOMICS_64` set on the test's
thread; on Windows, `valu_tests::wide_image_atomics_on_the_host` runs
them on an R32G32_UINT texture UAV (`View::Uint2TextureUav`) on devices
with 64-bit typed atomics.

For cross-lane code, `synth::wave_program` builds one wave (32 or 64
lanes) whose lanes load their own `v1` and `v2` from two arrays and store
their own `v4`. `eval_wave(body, lanes, wave64, a, b, constants)` runs
it and returns each lane's result.

Atomics from many lanes: `synth::image_wave_program(body, lanes)` is a
wave program with the typed texture UAV; `contended_image_atomic` makes
every lane do one 64-bit atomic on the same texel and
`check_contended` checks what holds in any lane order (each partial sum
handed out once, one cmpswap winner, umax's largest value);
`eval_image_wave` runs it on the evaluator, and
`valu_tests::contended_wide_image_atomics_on_the_host` on each device
with 64-bit typed atomics.

Probes (GPU-1): `emit::TEST_PROBES` set on the test's thread lowers with
probes; `synth::eval_probed(lowered, inputs, slots, extra)` gives the
probe buffer as the evaluator leaves it for given heap slot counts. On
the host, the probe constants and buffer are root parameters
`probe_param(stages)` and the next one (`Root::Constants`,
`Root::Uav(dwords)`: a zeroed buffer bound as a root UAV and read back).

Float lowerings against a device:
`valu_tests::compare_float_lowerings` (VOP3 programs) and
`compare_lowered_bodies` (any encoding) compare each stored dword with the
evaluator's, in the result's domain (`Domain`: f32, integer bits, an f16
in the low half, two f16 halves), and name every difference Microsoft's
specifications leave to the device (`Leeway`, each with its citation: NaN
bits; denormals flushed on input or output, never for halves; zero signs
from min/max; FMad fused or not; approximate operations within the
D3D11.3 chapter 22 bounds); any other difference fails, and every one is
printed. Users: `float_lowerings_agree_with_the_evaluator_on_the_host`
(43 f32 and conversion lowerings), `f16_lowerings_...` and
`packed_f16_lowerings_...`. Conversions the ISA defines exactly are held
to it on the evaluator by dense sweeps
(`f16_conversion_rounds_to_nearest_even_on_any_device`,
`integer_to_float_conversions_round_to_nearest_even_on_any_device`).

Probes before a fault: `synth::eval_probed_fault` runs a probed program
whose access has no descriptor (the evaluator stops there, its model of a
fault) and returns the probe buffer as it stood
(`a_heap_access_is_recorded_before_it_can_fault`).
`emit::tests::a_pixel_shader_with_probes_is_valid_dxil` has `dxil.dll`
judge a probed pixel shader.

Typed stores: `synth::mtbuf_offen` encodes MTBUF with the format in the
instruction, `eval_wave_buffer` runs a wave program over a 1 KiB buffer.

Submissions in flight (victory-gpu): `inflight` holds the components
overlap needs (`Retire`, `InFlight`) with model tests, and on Windows two
WARP workloads: eight submissions in flight retired by fence value with
their allocators reused, and a read waiting for exactly the newest writer
of its range while later work is held. The parallel pool
(victory-core `parallel`) has tests for concurrent callers, nested calls
and panics.

Queues (victory-gpu, any host): `schedule::Schedule` is the model of
queue waits, tested on its own; `queue::tests` run queues end to end on
the worker thread; on Windows, `schedule::d3d12` checks the model's rules
against a real D3D12 queue on WARP.

Benchmarks (ignored tests, release builds): `merge_speed` (victory-gpu,
the copy-back merge), `detile_speed` (victory-formats), `bcpack_throughput`
(victory-formats), `bench_write` (victory-shader, the bitcode writer),
`hashing_throughput` (victory-core: one thread against split hashes and
side-by-side inputs, as texture checks use them), `texture_checks_per_submission`
(victory-gpu). They measure CPU throughput, never frame rates.
