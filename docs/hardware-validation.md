# First hardware validation

A procedure for whoever next has a Windows machine with a hardware Direct3D
12 GPU and game files they are entitled to use. Nothing here has been run
yet: every GPU result so far is from software adapters on CI (WARP and the
Basic Render Driver), and the only game runs recorded are the README's, on
one machine. It assumes nobody has such a machine at hand; it is written so
that the first person who does gets results the engineering record can
use.

It needs the diagnostic build ([diagnostic-build.md](diagnostic-build.md))
and its requirements, and about two hours.

## Rules

- **One major diagnostic variable per run.** Each stage names its
  variable; set it, run, clear it (`set NAME=`). The only pairs are the
  ones a stage names.
- **Stop a stage at its first failure** and keep that run's files: the
  failure decides which stage is worth running next.
- **Save other work first.** Entering the world can hang the GPU, and
  Windows then resets the display driver (the screens go black for a
  moment). That is the watchdog (TDR) doing its job: never disable it.
- **Keep the game's content private.** Logs hold paths of your machine and
  your user name in them; captures and the shader cache hold game content.
  Review before sharing, and share only logs and notes.

## Setting up (once)

In Command Prompt, in the package folder, with `GAME` set to the title's
executable inside your game folder:

```bat
set GAMEDIR=D:\victory-data\game
set GAME=%GAMEDIR%\title.exe
set RUN=%USERPROFILE%\victory-runs
mkdir "%RUN%"
victory.exe --version > "%RUN%\build.txt"
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Format-List Name,DriverVersion; Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version; Get-CimInstance Win32_Processor | Format-List Name" > "%RUN%\machine.txt"
victory.exe run "%GAME%" > "%RUN%\bind.txt"
```

`bind.txt` gives the title id (the `title` line: eight hex digits) and any
missing imports. Set `DATA` to that title's data folder:

```bat
set DATA=%LOCALAPPDATA%\victory\data\<title id>
```

## Each run

```bat
set STAGE=01-startup
mkdir "%RUN%\%STAGE%"
victory.exe run "%GAME%" --execute 1> "%RUN%\%STAGE%\stdout.txt" 2> "%RUN%\%STAGE%\stderr.txt"
if exist "%DATA%\crashes.log" move "%DATA%\crashes.log" "%RUN%\%STAGE%\"
```

Moving `crashes.log` after each run keeps each run's crashes with it.
Write down in `%RUN%\%STAGE%\notes.txt` what was set, what you did, what
you saw and when (for example "menu at 0:40, black screen at 2:10").

What identifies a failure, in every stage:

| Failure | Where it shows |
|---------|----------------|
| crash | `crashes.log`: the faulting code (module and offset) and thread; the end of `stderr.txt` |
| host GPU failure | an `error` line in `stderr.txt` and the same text in `crashes.log`: what failed, `; on <adapter>`, `; device removed: <reason>` |
| GPU hang reset by Windows | the failure line above with a hung or removed device; Event Viewer → Windows Logs → System, source `Display` |
| hang without a failure | the window stops updating and `stderr.txt` stops growing; the next run of that stage with `VICTORY_STACKS=20,60` (seconds after start) logs every thread's stack |
| missing shaders or pipelines | the 5 s frame line's `dropped <n> without a host pipeline`, and the `dropped .../frame without a host pipeline: <why>` lines |
| black output | `a black frame after a frame that was not` warnings, with the draws of that frame |

## The stages

### 0. Baseline: the repository before this work

Every later result is compared with what the code did before the
engineering sessions began: commit `ffe310e`, the state the README's
results come from. Build it once, from a checkout of this repository:

```bat
git worktree add ..\victory-baseline ffe310e
cd ..\victory-baseline
cargo build --profile diag -p victory --features diagnostics
```

Run stages 1, 2, 3 and 9 with `..\victory-baseline\target\diag\victory.exe`
into `%RUN%\baseline-<stage>` folders, then the same stages with the
download. A stage the baseline passes and the download fails is a
regression to report first.

### Translation census

No game is started: every shader of the game folder is translated, checked
by `dxil.dll` (the Windows SDK's) and, with `--pipelines`, made into a host
pipeline on the GPU. It reports how many programs translate and why the
others do not; `--list` adds one line per program and outcome:

```bat
victory.exe translate "%GAMEDIR%" --validate --list > "%RUN%\census.txt" 2>&1
victory.exe translate "%GAMEDIR%" --validate --pipelines > "%RUN%\census-pipelines.txt" 2>&1
```

Run it with the baseline too: the difference is what this work changed in
coverage. The most frequent reasons decide the next lowering to write.

### 1. Startup

Variable: none. Start, wait for the first screen, quit from the title or
close the window after a minute. Pass: `stdout.txt` names the build and the
title, `stderr.txt` has `frame presented` lines (at frames 1, 2, 4, 8 and
so on), there is no `crashes.log`.
A crash here is a loader or runtime problem; the crash report names the
code.

### 2. Menu rendering

Variable: none. Go from the start screen through the main menu, the
settings pages and back, five minutes. Note what is missing or wrong on
each screen, and the caption's fps. Pass: every menu draws, no
`dropped ... without a host pipeline` (the README has every menu shader
translating), no black-frame warnings.

### 3. World loading

Variable: none. Load into the world (about 50 s on the README's machine)
and stay a minute. Note the loading time, whether the world appears, and
the last lines of `stderr.txt` if the screen hangs. This is the run where
the README's machine hung the GPU. A device removal here sends you to stage
4; a hang without one, to stage 7 and then 6.

### 4. Device removal (DRED)

Variable: `VICTORY_D3D12_DRED=1`. Repeat stage 3. The failure line then
adds `; page fault at <address>`, the allocations it falls in (`inside`,
`inside freed`), where it lies against each descriptor heap (`; view heap
<start>+<size>: ... past its end (slot <n>)`) and the breadcrumbs
(commands done per list). Reading it:

| Report | Means | Next |
|--------|-------|------|
| a page fault past a descriptor heap's end | a shader indexed past the heap (GPU-1) | stage 5 |
| a fault inside a freed allocation | a resource freed while the GPU still used it | report it with the breadcrumbs |
| no page fault, the device hung | a shader that runs too long, or work waiting forever | stage 7, then 6 |

### 5. Descriptor indexing

Variable: `VICTORY_SHADER_PROBES=1`. Repeat stage 3 (shaders are translated
again with probes: loading takes longer). Look for `shader probes on`, then
any `probes: a submission's shaders indexed past a heap or hit the loop
limit` lines and the `probe: <stage> shader <hash>: heap access <n> ... at
view index <i>, <k> past the bound heap's <s> slots` records. No records
for heap accesses means no shader indexed past its heap in that run (GPU-1
refuted for it; probes do not see an index inside the heap that names the
wrong descriptor). Records name the shader to fix. Nothing is clamped: a
run with records may still fault, as it would without probes. Each access
is checked before it fetches its descriptor, and its record is fenced to
memory first, so the failure line's `; probes: ...` can name the access
that removed the device; that the record survives the removal is likely,
not guaranteed. Two limits: the counts are 32-bit and wrap after 2^32
counted accesses in one submission (the records are still the first 64);
and a pipeline whose root signature has no room for the probes' 4 dwords
is dropped in this run only, logged as `probes do not fit this pipeline's
root signature` -- compare the dropped draws with stage 3 before reading
anything into missing geometry.

### 6. Queue waits

Variable: none, then `VICTORY_QUEUE_WAITS=0`. In stage 3's logs, look for
`since the last report: submissions and signals held back by queue waits`
(work our queues held for a wait, as D3D12 does) and `queues are held back
by waits not reached` (a wait lasting 5 s or more, with the waits only
held work could satisfy). Then repeat stage 3 with `VICTORY_QUEUE_WAITS=0`,
which lets that work run ahead as before this session. A hang with the
default and none with `=0`, with `only_held_work_signals` listed, means a
wait that only our held work could satisfy (report the list); flicker or
corruption with `=0` and not by default means running ahead was a real
hazard (SYNC-1).

### 7. Shader loops

Variable: `VICTORY_SHADER_LOOP_LIMIT=4096`. Repeat stage 3 and compare the
5 s frame lines with stage 3's. Much faster frames mean loops that run into
the limit are what makes frames slow (PERF-3); stage 5's records also count
the lanes the default limit stopped (`a lane stopped by the loop limit`). A
different picture with the lower limit is expected where loops genuinely
run longer than 4096 iterations; it is not by itself a bug.

### 8. 64-bit image atomics (experimental)

Variable: `VICTORY_SHADER_IMAGE_ATOMICS_64=1` (off by default; checked on
WARP only). Repeat stage 3. On a GPU without 64-bit atomics on typed
resources, those pipelines are refused (`dropped ... without a host
pipeline: 64-bit atomics on typed heap resources, which the host device
lacks`): record that and stop. Otherwise compare with stage 3: fewer
dropped draws, any failure. Only if that run is clean, repeat it with
`VICTORY_D3D12_DEBUG=1` added, then with `VICTORY_D3D12_GBV=1` added too;
`D3D12:` warnings about these views (their format) are what stops SH-1
from being turned on by default.

### 9. Frame timing

Variable: none. In the world, stand still at one place for two minutes,
then move for two. Keep the 5 s lines: fps, ms per frame, ms preparing
(copies, textures, with MB), ms on the GPU, ms copying back, ms waiting
for another thread's submission, draws and dropped draws per frame. The
waiting figure says whether running submissions one at a time costs
frame time (SYNC-1, PERF-2 in the status page). These numbers are the baseline every
optimization is measured against (target: 30 fps, 33 ms per frame; the
README's machine had 0.5-1.8 s). One more run with `VICTORY_GPU_TIMING=1`
names the slow command lists (with `VICTORY_D3D12_DRED=1` too, their
slowest commands).

The 5 s lines are averages. Presented-frame intervals one by one, which
show stutter the averages hide: `VICTORY_LOG=victory_gpu::frame=debug,info`
logs a timestamped `present` line per frame (a variable of its own: run
it separately), or an external capture with Microsoft's PresentMon while
the game runs:

```bat
PresentMon.exe --process_name victory.exe --timed 120 --terminate_after_timed --output_file "%RUN%\09-presentmon.csv"
```

Report the median and the 99th percentile of the intervals, next to the
5 s lines' CPU (preparing) and GPU times.

### 10. Memory

Variable: none. While stage 9 runs, sample memory every 5 s in a second
window (PowerShell):

```powershell
$out = "$env:USERPROFILE\victory-runs\10-memory.csv"
"time,private_bytes,working_set,gpu_dedicated,gpu_shared" | Set-Content $out
while ($p = Get-Process victory -ErrorAction SilentlyContinue) {
  $gpu = (Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage","\GPU Process Memory(pid_$($p.Id)_*)\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples
  $dedicated = ($gpu | Where-Object Path -like "*dedicated usage" | Measure-Object CookedValue -Sum).Sum
  $shared = ($gpu | Where-Object Path -like "*shared usage" | Measure-Object CookedValue -Sum).Sum
  "{0:o},{1},{2},{3},{4}" -f (Get-Date), $p.PrivateMemorySize64, $p.WorkingSet64, $dedicated, $shared | Add-Content $out
  Start-Sleep 5
}
```

Record the peaks at the menu and in the world, and Task Manager's
**Committed** total. Windows error 1455 in `crashes.log` means the system
ran out of commit (the README says how to raise it).

### 11. Graphics correctness

Variable: `VICTORY_CAPTURE=%RUN%\11-frames`, after `mkdir
"%RUN%\11-frames"` (the folder must exist). Frames whose number is a power
of two are saved, or those `VICTORY_CAPTURE_FRAMES` lists (the frame
numbers are in the `frame presented` lines). Capture a menu screen and a world view, and compare
them with how the same screens look on your console. Then capture the same
frames with `VICTORY_D3D12_WARP=1` (very slow): wrong output on both
adapters points at translation; wrong output on the GPU only, at undefined
behaviour the software adapter happens to hide, or the driver. Describe the
differences in the notes; the images are game content: keep them private.

### 12. Interactive stability

Variable: none. Play for 30 minutes from a save in the world: walk,
drive, open menus and the map, load a save. Note every crash, hang,
device removal and visual fault with its time; the stage passes only with
none. Run it once with the baseline for comparison if the baseline
reaches the world.

## Reporting

Keep `build.txt`, `machine.txt`, `bind.txt` and each stage's folder. What
each stage settles in the engineering record
([engineering-status.md](engineering-status.md)): 3 and 4, the world hang
and GPU-1; 5, GPU-1; 6, SYNC-1; 7, PERF-3; 8, SH-1; 9 and 10, PERF-1, PERF-4
and PERF-5 (their baselines); 11, the scene's correctness; 0 and the
census, what changed since `ffe310e` and which lowerings matter most; 12,
whether the game is playable at all. A stage not run,
or run with a different variable, is reported as such, never as passed.
