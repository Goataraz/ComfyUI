# ComfyUI on Beast — Sub-Project A: Smart Caching & Two-Repo Fork

**Date**: 2026-06-10
**Status**: Approved design (pending user review of written spec)
**Author**: Claude (brainstorming session with Chris)
**Bound to**: [[project-goal]] (the /goal)

## Why this sub-project

The user wants Beast to "utilize everything it has." Investigation showed that upstream ComfyUI has a modern RAM-pressure cache, dynamic VRAM, and a pin memory subsystem that Beast is not using — `--lowvram` (the flag in `start-tp.sh`) actively conflicts with `--enable-dynamic-vram`. Meanwhile Beast has 98GB of free RAM, 23GB in swap, and 4.9GB of VRAM used per rank, which is *enormous* headroom left on the table.

The first sub-project in the [[project-goal]] roadmap is to flip on the upstream caching features and set up a two-repo fork (Pattern B) so the user's "I don't want upstream changes to mess us up" goal is structurally enforced. No new caching code, no new fork code — just configuration + the protective structure.

## Goals

1. Adopt upstream's modern caching: `--cache-ram`, `--enable-dynamic-vram`, `--fast` flags enabled in `start-tp.sh`. Remove `--lowvram`.
2. Measure before/after: time-to-first-image, model-switch time, sustained GPU util, RAM used at steady state. Record in `BASELINE.md`.
3. Set up two-repo structure: `~/comfyui-portable` (Pattern B fork of ComfyUI) and `~/comfyui-beast` (config/deploy, never rebased). Pin a known-good ComfyUI release.
4. Add targeted tests that catch the "TP broke on the new base" failure mode during any future rebase, plus a rebase script that runs the suite automatically.

## Non-goals

- **No new fork code for caching.** Upstream already has it. We adopt, don't write.
- **No dashboard, telemetry, or monitoring.** Empty `comfyui-beast/monitoring/` directory reserved for future.
- **No auto-recovery beyond systemd `Restart=on-failure`.** Not in scope.
- **No TP pruning** (Cosmos/HiDream/WanVideo). Sub-project B, after A.
- **No Ollama integration** (custom nodes or otherwise). Sub-project C, can run in parallel.
- **No model dedup, no storage cleanup.** Sub-project E.
- **No service resilience work** (health-checked systemd, GPU temp/power monitoring). Sub-project D.

## Architecture

Two repos with a clean boundary. The fork tracks upstream; Beast-specific config is wholly separate.

```
~/comfyui-portable/                  ← ComfyUI fork, Pattern B
├─ master                           ← tracks upstream ComfyUI tags, fast-forward only
├─ tensor-parallelism               ← our work, rebases onto master
├─ releases/v0.22.2-pinned          ← known-good upstream release tag (initial pin)
├─ tests-unit/comfy_test/distributed/
│   ├─ test_tp.py                   ← 46 existing TP tests (preserve)
│   └─ test_rebase_safety.py        ← NEW: regression for the cache-flags contract
└─ .upstream-pin                     ← text file: tag=X, sha=Y, pinned_at=Z

~/comfyui-beast/                     ← Beast config, never rebased
├─ start-tp.sh                      ← wraps portable/main.py with our flags
├─ env.sh                           ← cuDNN paths, NCCL, TOKENIZERS_PARALLELISM
├─ systemd/
│   └─ comfyui-tp.service           ← Restart=on-failure, after=network.target
├─ paths/
│   ├─ models.yaml                  ← model dir mappings
│   └─ cache-ram.yaml               ← --cache-ram thresholds, --fast feature list
├─ deploy/
│   ├─ install.sh                   ← sets up symlinks, installs systemd unit
│   └─ rebase-from-upstream.sh      ← bumps the pin, runs the test suite
├─ tests/                           ← NEW: config + boundary tests
│   ├─ test_start_tp_flags.py
│   ├─ test_paths_yaml.py
│   └─ test_upstream_pin.py
├─ bench/                           ← NEW: performance measurement scripts
│   └─ baseline.py
├─ monitoring/                      ← (empty — future sub-project)
└─ README.md
```

**Boundary rule**: No file in `~/comfyui-portable/` knows about Beast. No model paths, no systemd, no `start-tp.sh`, no Beast-specific tuning. **No file in `~/comfyui-beast/` is a ComfyUI patch.** The two repos meet at the systemd unit that calls into `portable/main.py` with Beast's env vars.

## Data flow

When a generation request comes in:

```
HTTP POST /prompt
    ↓
[execution.py] → enqueue
    ↓
[PromptExecutor] → for each node:
    ↓
[model_management.py:LoadedModel.model_load]
    ↓
├─ dynamic VRAM check:           uses psutil + cuda mem info
│   - if args.lowvram:  conservative allocation  [REMOVED by this spec]
│   - else:             full VRAM, offload to RAM
│
├─ check pin budget:             ensure_pin_budget()
│   shortfall = TOTAL_PINNED + new_size - MAX_PINNED (108GB on Beast)
│   if shortfall > 0: free_pins() — evict LRU inactive models
│
├─ load model from disk → mmap   load_safetensors() with comfy_aimdo.model_mmap
│   [RE-ENABLE: restore the mmap backend; current working tree has open() instead]
│
├─ pin in RAM:                   TOTAL_PINNED += size
│
├─ if VRAM available:            load to GPU
└─ if not:                        leave pinned in RAM, stream chunks on demand
```

The four storage tiers, in order of access speed:

| Tier | Capacity on Beast | Where | Behavior |
|---|---|---|---|
| GPU VRAM | 16GB × 2 ranks = 32GB | RTX 5060 Ti | Hot — active layers |
| Pinned RAM | ~108GB (90% of 121GB) | system RAM | Warm — full model resident, lazy upload to GPU |
| OS page cache | automatic | kernel | Transparent — already-pinned mmap'd files are auto-cached |
| Disk | 23TB free | NVMe | Cold — initial load, mmap source |

## Component breakdown

| Component | Status | Where it lives |
|---|---|---|
| `--cache-ram`, `--enable-dynamic-vram`, `--fast` flags | Already in upstream, just turn them on | `comfyui-beast/paths/cache-ram.yaml` |
| `--lowvram` removal | Single line in start-tp.sh | `comfyui-beast/start-tp.sh` |
| `comfy/utils.py:load_safetensors` reversion | Restore mmap-based loading | `comfyui-portable/comfy/utils.py` (one-line edit) |
| Pin budget tuning | Defaults are conservative; we may need to raise MAX_PINNED_MEMORY for 121GB systems | `comfyui-beast/paths/cache-ram.yaml` |
| Test coverage | 46 existing TP tests; add 3 rebase-safety tests | `comfyui-portable/tests-unit/comfy_test/distributed/` |
| Config smoke tests | New | `comfyui-beast/tests/` |
| Rebase script | New | `comfyui-beast/deploy/rebase-from-upstream.sh` |
| Performance baseline | New measurement script + manual record | `comfyui-beast/bench/` and `docs/superpowers/specs/BASELINE.md` |
| Monitoring | Not in this spec | `comfyui-beast/monitoring/` (empty placeholder) |

## Rebase flow (the protective layer)

When upstream ComfyUI ships a new release:

1. `cd ~/comfyui-portable && ./comfyui-beast/deploy/rebase-from-upstream.sh v0.22.3`
2. Script:
   a. `git fetch upstream`
   b. Fast-forward `master` to the new tag
   c. `git rebase master` onto `tensor-parallelism`
   d. Run `pytest tests-unit/comfy_test/distributed/test_rebase_safety.py` and `pytest comfyui-beast/tests/`
3. If tests pass: update `.upstream-pin`, tag a new release branch, exit 0
4. If tests fail: stop, do NOT update `.upstream-pin`, surface the failing test names, exit non-zero
5. User decides: fix forward, stay pinned, or roll back the rebase

This makes the "upstream messes us up" risk *visible and recoverable* instead of silent.

## Error handling

Four failure categories, with explicit handling:

**Category A — User-visible failures (a model fails to load, generation errors out)**:
- ComfyUI's existing aiohttp error path takes over: returns 500 with JSON
- When a model load fails with OOM, error message includes current pin budget and remaining headroom (already in upstream; we don't change it)
- Recovery: user runs the prompt again after closing other GPU apps or adjusting `cache-ram.yaml`
- We do not modify this; we just do not break it.

**Category B — Process-level failures (a rank crashes, NCCL hangs, model file corrupted)**:
- torchrun's elastic agent SIGTERMs the surviving rank after timeout
- systemd `Restart=on-failure` restarts within 5s
- New rank rejoins NCCL via existing `init_distributed` flow
- Pin cache state resets per-process (not preserved across restarts in this sub-project)

**Category C — Upstream rebase breaks things**:
- `rebase-from-upstream.sh` runs the test suite after rebase
- If any test fails: rebase is rejected, `.upstream-pin` is NOT updated
- Script exits non-zero, surfaces the failing test names
- User decides: fix forward, stay pinned, or roll back

**Category D — Disk full / RAM full**:
- RAM pressure cache (`--cache-ram`) auto-evicts inactive pins when `psutil.virtual_memory().available` falls below configured headroom
- Disk: not a real concern with 23TB free. Tunable via `cache-ram.yaml` if it ever becomes one.

**Failure modes we explicitly do NOT handle**:
- GPU hardware errors (ECC faults, thermal throttling) — OS/driver handles
- Model file corruption on disk — existing safetensors header validation handles
- Network failures (CivitAI downloads) — LoRA Manager's download coordinator handles

## Testing

The 46 existing TP tests are the foundation. We add a small, targeted set on top. Principle: **tests cover the boundary, not the implementation.** Upstream's caching is tested upstream; we test that *we configured it correctly* and that *rebase didn't break our patches*.

### Test set 1 — Configuration smoke (in `comfyui-beast/tests/`)

- `test_start_tp_flags.py`: parses `start-tp.sh`, asserts `--lowvram` is NOT present, asserts `--cache-ram`, `--enable-dynamic-vram`, `--fast` ARE present
- `test_paths_yaml.py`: validates `cache-ram.yaml` schema (active_gb ≤ inactive_gb, both positive, inactive ≤ 0.9 × system RAM)
- `test_upstream_pin.py`: reads `.upstream-pin`, asserts the tag matches `git describe` on master

### Test set 2 — Rebase regression (in `comfyui-portable/tests-unit/comfy_test/distributed/test_rebase_safety.py`)

- The 46 existing TP tests, run as a regression suite (preserve as-is)
- 3 new tests:
  - `test_load_safetensors_uses_mmap`: asserts `comfy_aimdo.model_mmap.ModelMMAP` is the file backend (regression for the unstaged change that turned it off)
  - `test_pin_budget_default`: asserts `MAX_PINNED_MEMORY` math is sane (≤ 0.9 × system RAM)
  - `test_lowvram_flag_takes_no_effect_when_dynamic`: when both `--lowvram` and `--enable-dynamic-vram` are set, dynamic wins

### Test set 3 — Performance baseline (NOT automated; manual measurement)

- A `bench/baseline.py` script that times 5 workflows at different model sizes
- Output is a markdown table that goes into `docs/superpowers/specs/BASELINE.md`
- We do not gate the build on performance; we record before/after and reference it in the spec

### Test invocation

- `pytest tests-unit/comfy_test/distributed/test_rebase_safety.py` — runs in CI / pre-rebase
- `pytest comfyui-beast/tests/` — config smoke
- `./comfyui-beast/deploy/rebase-from-upstream.sh <tag>` — runs the full suite as part of rebase
- No "all tests must pass before commit" hook; we trust the user to run them

### What's NOT tested

- The behavior of the upstream caching subsystem (upstream's job)
- The TP code itself (already tested by 46 tests)
- End-to-end image quality (visual judgment, not a unit test)

## Out of scope reminders (so we don't drift)

- No dashboard
- No telemetry / metrics
- No service mesh or health-checked systemd (just `Restart=on-failure`)
- No TP pruning
- No Ollama integration
- No model dedup
- No storage reorg

## Open questions for the implementation phase

These are not blockers; they're decisions to make *during* implementation, not before:

1. **What pin?** Today's `master` is 25 commits ahead of `origin/master`, and the TP branch is 44 commits ahead. We need to pick a known-good upstream tag as our starting point. My recommendation: the most recent ComfyUI release tag that includes the `--cache-ram` feature (probably v0.22.x or later). Verify in implementation step 1.

2. **What `cache-ram.yaml` thresholds?** Upstream defaults are active 10% (min 2GB, max 10GB) and inactive 100% (max 96GB). On a 121GB system, the inactive cap of 96GB may be too low — we'd want to push to 108GB. Decide in implementation step 4 by reading the actual psutil output.

3. **What's the manual baseline run?** The "before" measurements need to happen *before* we flip the flags, on the current `--lowvram` config. Implementation step 2 captures that. Then we flip, then we measure "after."

4. **Git hosting for the fork?** Local-only (no remote), GitHub private, or GitLab on Beast? The user has not decided. The fork works either way; the only difference is where `git push` goes for backup. Decide in implementation step 1.
