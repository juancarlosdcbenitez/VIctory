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
  conversions, rounding, omod/clamp, carries, 64-bit SALU, SDWA, f32
  arithmetic and the legacy multiply, packed 16-bit integers.
* **victory-gpu builds on Linux** (verified): its 24 portable tests run.
* Evaluator programs are validated by `dxil.dll` wherever it loads
  (implemented, not run: there is no `dxil.dll` here; Windows CI has one).

Blocked here: running DXC's validator on Linux (a conda-forge build of the
open-source DXC was fetched but executing it was not permitted in this
session); pushing the branch (not permitted without the owner's say-so).

## Backlog

Priority order: P0 GPU stability, P1 shader correctness, P2 performance,
P3 runtime compatibility, P4 usability.

| Id | P | State | Item |
|----|---|-------|------|
| GPU-1 | P0 | hypothesis | **Out-of-range heap indices as a hang cause.** Translated shaders index `ResourceDescriptorHeap[]` with `(table base + offset) >> 5` (`emit::lower::heap_index`), unbounded; a stale or garbage table base or per-lane index reads past the host heap, a page fault (TDR on AMD). `descriptor::host::near_heaps` already places a faulting VA against the heaps in DRED reports. Check: a world run with `VICTORY_D3D12_DRED=1` and see whether the fault lies past a heap. If it does: pass each heap's slot count as user data and clamp indices (or point them at a null slot) in the lowering. |
| SH-1 | P1 | not implemented | **64-bit image atomics** (README). Needs `dx.op.atomicBinOp.i64` / `atomicCompareExchange.i64` on a typed UAV annotated U64 (component type 7), shader flags Int64Ops (bit 20), AtomicInt64OnTypedResource (bit 27) and AtomicInt64OnHeapResource (bit 32) with SFI0 0x8000, 0x400000, 0x10000000 (numbers from memory of DXC's `DxilShaderFlags.h`: confirm there), host R64_UINT UAV views for those slots, and the D3D12_OPTIONS9 check. Refused today in `emit::lower` (`"a 64-bit image atomic"`). |
| SH-2 | P1 | not implemented | `tbuffer_store_format_*`: typed-format stores are refused (`buffer()`); the inverse of `format_load` (pack, round and saturate per numeric format). |
| SH-3 | P1 | not implemented | Double precision (`v_*_f64`, `v_cvt_*_f64`, f64 compares): the DXIL writer has no `double` type. |
| SH-4 | P1 | not implemented | `v_div_scale/fmas/fixup_f32` (the precise division sequence). |
| SH-5 | P1 | not implemented | `v_permlane16/x16_b32`, `v_dot2_f32_f16`, `v_dot2c_f32_f16`, `v_msad_u8`, `v_qsad/mqsad`, `image_msaa_load`, `image_*_pck*`, flat and scratch memory, `s_getreg_b32`, scalar stores and atomics, clamped integer arithmetic (VOP3 clamp), constant `v_cmp_f/t_*64`. |
| SH-6 | P1 | not implemented | Merged geometry stages; 64-lane cross-lane code on 32-lane hosts (README). |
| SH-7 | P1 | not checked | The 59 new lowerings through `dxil.dll` and on a GPU: Windows CI's `cargo test` now validates them (evaluator programs); `valu_tests` cases for a few (`v_mad_u64_u32`, `v_ldexp_f32`, `v_frexp_*`) would cover the GPU. |
| PERF-1 | P2 | open | World frames take 0.5-1.8 s (README, one machine). No measurement possible here. The existing switches (`VICTORY_GPU_TIMING`, `VICTORY_SHADER_LOOP_LIMIT`) are where to start. |

## Next session

1. On Windows: `cargo test --workspace --locked` (validates every
   evaluator program with `dxil.dll`; a rejection names the body).
2. `victory translate <game dir> --validate --list` before and after
   these commits: the count of translated shaders per stage, and which
   reasons remain most frequent.
3. If the remaining reasons are dominated by 64-bit image atomics, SH-1;
   otherwise the most frequent reason, with an evaluator test written
   from the ISA first.
4. A world run with `VICTORY_D3D12_DRED=1` to settle GPU-1.
