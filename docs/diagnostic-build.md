# The diagnostic build

> **Experimental. Not verified for gameplay.** The build is made by CI on
> machines with no GPU and no game files: no game has been run with it.
> Its checks (lints, unit tests, DXIL validation, D3D12 tests on WARP and
> the Basic Render Driver, both software) are not hardware GPU validation.
> It is not a release.

`victory.exe` built with release code generation and the `diagnostics`
feature (the README's Debugging section): debug-level logs and the
debugging switches below are compiled in. No switch is on unless its
variable is set, so without them it runs like a release build.

## What the download holds

The Windows validation workflow builds it once a push's checks pass, and
keeps it 14 days as the artifact `victory-diag-<version>-<commit>-windows-x64`
(on the workflow run's page, under Artifacts).

| File | What |
|------|------|
| `victory.exe` | the executable; `victory.exe --version` prints the build, which names the commit |
| `victory.pdb` | its symbols (line tables), for stacks in crash reports and debuggers |
| `BUILD-INFO.txt` | commit, branch, source and workflow-run links, toolchain, build command, the checks that passed |
| `SHA256SUMS.txt` | the checksums of the two above |
| `LICENSE` | GPL-3.0-or-later; the source is the commit in `BUILD-INFO.txt` |
| `THIRD-PARTY-NOTICES.txt` | the crates compiled in, their licenses and license files (`cargo xtask notices`) |
| `victory.example.toml` | the configuration template |
| `docs/` | this guide and the hardware validation procedure |

It holds nothing from any game: no game files, shaders, shader caches,
saves or keys. The job copies only the files listed and fails if the
package holds anything else.

To build the same thing from a checkout:

```bat
cargo build --locked --profile diag -p victory --features diagnostics
```

The executable is `target\diag\victory.exe` (`--version` then prints the
package version only; CI sets `VICTORY_BUILD` to add the commit).

## Before running it

- Check the download, in PowerShell in the package folder, against
  `SHA256SUMS.txt`:

  ```powershell
  Get-FileHash victory.exe, victory.pdb -Algorithm SHA256
  .\victory.exe --version
  ```

- Requirements: Windows 10 or 11 x64, a CPU with AVX2, a Direct3D 12 GPU
  with shader model 6.6 (and mesh shaders for the world), the Microsoft
  Visual C++ runtime (`vcruntime140.dll`; if Windows reports it missing,
  install Microsoft's Visual C++ Redistributable for x64), several GiB of
  free Windows commit (Task Manager → Performance → Memory → Committed),
  and a copy of the game you are entitled to use. `VICTORY_D3D12_DEBUG`
  also needs the Windows optional feature **Graphics Tools**.
- Entering the world on a real GPU can currently hang it; Windows then
  resets the display driver (the screens go black for a moment). Save
  other work first. `VICTORY_D3D12_WARP=1` renders on the software adapter
  instead, which cannot reset the driver (and is very slow).
- Configure paths outside the package folder: copy `victory.example.toml`
  to `victory.toml` next to where you run it (or point `VICTORY_CONFIG` at
  one) and set `game_dir`; `cache_dir` and `data_dir` default to
  `%LOCALAPPDATA%\victory\cache` and `%LOCALAPPDATA%\victory\data`.

## Collecting logs

In **Command Prompt** (`cmd.exe`), in the package folder; the game path is
an example:

```bat
mkdir "%USERPROFILE%\victory-logs"
victory.exe config
victory.exe run "D:\victory-data\game\title.exe"
set VICTORY_LOG=info
victory.exe run "D:\victory-data\game\title.exe" --execute 1> "%USERPROFILE%\victory-logs\stdout.txt" 2> "%USERPROFILE%\victory-logs\stderr.txt"
```

`config` shows the paths in use; `run` without `--execute` loads and binds
the title without starting it (it prints the build, the title id and
missing imports). With `--execute` the title starts:

| File | What |
|------|------|
| `stdout.txt` | the run summary: `victory` build, title build, title id, console, DLLs, imports |
| `stderr.txt` | the log (`VICTORY_LOG`: `info` by default; `debug`, or per target as `victory_gpu=debug,info`) |
| `%LOCALAPPDATA%\victory\data\<title id>\crashes.log` | every crash and host GPU failure, appended (the title id is in `stdout.txt`) |
| `%LOCALAPPDATA%\victory\cache\shaders` | translated shaders (from the game: keep, never share) |

The window caption shows `VIctory <n> fps <n> ms/frame <n> draws/frame`,
and every 5 s the log has a line `<n> fps, <n> ms/frame (<n> preparing:
copies ..., textures ...; <n> on the GPU; <n> copying back ...; <n>
waiting for another thread's submission), <n> draws/frame, dropped <n>
without a host pipeline ...`. The figures are milliseconds per frame,
averaged over the 5 s; the last one is how long title threads waited for
each other because submissions run one at a time.

Before sharing a log, read it: logs hold paths of your machine (and your
user name in them). Frame captures and saves hold the game's content;
do not publish them.

The shader cache is keyed on the executable, so a new build translates
every shader again on its first run (slower start, more log lines).

## One switch at a time

Set one variable, run, collect the logs, then clear it (`set NAME=`) before
the next. Two at once make it impossible to tell which one changed the
outcome; the exceptions are the pairs the table names (GBV needs the debug
layer). Those marked *translation* get their own shader cache entries.

| Variable | Effect | Look for |
|----------|--------|----------|
| `VICTORY_D3D12_DRED=1` | on a GPU failure: the faulting address, the allocations it hit and the commands in flight | the `error` line ending the run, also in `crashes.log`: `; device removed: ...`, `; page fault at ...` |
| `VICTORY_D3D12_DEBUG=1` | the D3D12 debug layer (slower) | `D3D12: ...` warnings; `; debug layer: ...` in the failure line |
| `VICTORY_D3D12_GBV=1` | with `VICTORY_D3D12_DEBUG=1`: GPU-based validation, bad descriptor use reported instead of faulting (very slow) | `D3D12: ...` warnings about descriptors |
| `VICTORY_SHADER_PROBES=1` | *translation*: shaders count and record heap indices past the bound heap, checked before each descriptor fetch, and lanes the loop limit stops; nothing is clamped; pipelines without root signature room for the probes are dropped in such runs | `shader probes on`, `probes: a submission's shaders indexed past a heap or hit the loop limit`, `probe: <stage> shader <hash>: heap access ...`; a failure line adds `; probes: ...`; `probes do not fit this pipeline's root signature` |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | *translation*: shader loops stop after `n` iterations per lane instead of 2^20 | `translated loops stop sooner`; compare the 5 s frame lines with a run without it |
| `VICTORY_SHADER_IMAGE_ATOMICS_64=1` | *translation*: 64-bit image atomics are translated (off by default; checked on WARP only); refused on devices without 64-bit typed atomics | `64-bit image atomics are translated`; pipeline errors naming them |
| `VICTORY_QUEUE_WAITS=0` | queue waits no longer hold work back (the old behaviour, to compare) | `queue waits do not hold work back`; without it: `queues are held back by waits not reached` after 5 s, and the held counts after the frame lines |
| `VICTORY_QUEUE_ASYNC=1` | *experimental*: every submission, signal, present and page mapping runs on the queue worker in each queue's order; title threads do not wait for the GPU when they submit | `submissions, signals, presents and page mappings run on the queue worker`; compare fps and ms/frame with a run without it; any hang, removal or wrong image |
| `VICTORY_TEXTURE_THREADS=<n>` | texture checks use at most `n` threads, 1 to 8 (default: the machine's, at most 8; 1: only the thread submitting) | `texture checks use up to <n> thread(s)`; compare the 5 s lines' textures ms and ms/frame with a run without it |
| `VICTORY_GPU_TIMING=1` | each command list of a submission run and timed on its own (slow); with `VICTORY_D3D12_DRED=1` the commands are noted too | slow lists, with their slowest commands when noted |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | draws and dispatches of pipelines with those shader hashes omitted | whether a failure or slowdown goes away |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter | `; on <adapter> (software)` in failure lines; a fault that WARP also shows points at translation, not the driver |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds | the stacks in the log (where a hang waits) |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (`VICTORY_CAPTURE_FRAMES=<list>` picks them) | the images: game content, keep private |

The README's Debugging section lists the rest. The order to try them in,
and what each result means, is [hardware-validation.md](hardware-validation.md).
