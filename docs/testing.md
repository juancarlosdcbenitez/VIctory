# Testing

What runs where, and the commands. No test needs game data unless it says
so; those read paths from environment variables and skip without them.

## Layers

| Layer | Where | What it checks |
|-------|-------|----------------|
| Unit tests | any host (most crates), Windows (all) | parsers, layouts, codecs, IR, structurisation, resource tracing |
| Shader evaluator tests (`victory-shader`, `emit::eval_tests`) | any host | what a lowering computes: synthetic GFX10 programs lowered to DXIL and run by a test-only interpreter of the emitted module (`emit::bitcode::eval`), against references written from AMD's RDNA2 ISA |
| DXIL validation | Windows with `dxil.dll` (Windows SDK, `VICTORY_DXIL`) | every program the evaluator tests lowers (`synth::lower_valu`), the VALU GPU tests, and the container tests in `emit` |
| GPU tests (`emit::valu_tests`, `emit::waterfall_tests`) | Windows, hardware GPU and/or WARP with SM 6.6 | the translated shader on host D3D12 |
| Differential tests (`emit::diff_tests`) | Windows with the console SDK's `dxc.exe` (`VICTORY_CONSOLE_DXC`) | console-compiled HLSL translated, against the same HLSL compiled for PC |
| Data tests (BCPack, XMA, loader probes) | wherever the user's files are | real streams and executables, by properties only |
| Policy | any host | `cargo xtask policy`: nothing from the game in the repository |

The evaluator makes the choices DXIL leaves to drivers one way (FMax/FMin
of two zeros give the first operand, FMad is unfused, float-to-integer
conversions saturate) and refuses what LLVM leaves undefined (shifts by
the bit width, division by zero, branches on undefined values), so a
lowering that leans on either fails there. It runs one thread; another
lane's value (quad and wave reads) is undefined. The first evaluator
tests replay the cases `valu_tests` checks on hardware, so its choices are
known to agree with real drivers on those.

## Commands

On Windows (what CI runs, `.github/workflows/ci.yml`):

```powershell
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
```

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
