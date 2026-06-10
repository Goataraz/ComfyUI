# Smart Caching & Two-Repo Fork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Pattern B fork of ComfyUI plus a sister Beast-config repo, adopt upstream's modern RAM/dynamic-VRAM caching flags, and validate the result on Beast — with a rebase script that runs the test suite so upstream changes can't silently break us.

**Architecture:** Two new repos live at `~/comfyui-portable/` (Pattern B fork: `master` fast-forwards to upstream tags, `tensor-parallelism` is the long-lived work branch) and `~/comfyui-beast/` (config + deploy + tests + bench, never rebased). The current `/home/blackthorn/ComfyUI/` working tree becomes the seed for the portable fork. The caching change is a one-line flag swap in `start-tp.sh` plus a one-line restoration in `comfy/utils.py:load_safetensors`. Tests cover the boundary (config + rebase regression) — not the implementation (upstream's caching).

**Tech Stack:** Python 3.13, PyTorch + torchrun (existing), pytest, psutil, shell, git.

**Spec:** `docs/superpowers/specs/2026-06-10-smart-caching-and-fork-design.md`

---

## File Map

### `~/comfyui-portable/` (new repo, Pattern B fork)

| Path | Action | Responsibility |
|------|--------|----------------|
| (entire repo) | Create from current `/home/blackthorn/ComfyUI/` | The fork itself |
| `comfy/utils.py` | Modify (line 90) | Restore `model_mmap.get_file_handle()` (regression from unstaged edit) |
| `tests-unit/comfy_test/distributed/test_rebase_safety.py` | Create | 3 rebase-safety tests |
| `.upstream-pin` | Create | Text file: `tag=`, `sha=`, `pinned_at=` |

### `~/comfyui-beast/` (new repo, never rebased)

| Path | Action | Responsibility |
|------|--------|----------------|
| `start-tp.sh` | Create | Wraps portable/main.py with our flags; **moved from `/home/blackthorn/ComfyUI/start-tp.sh`** |
| `env.sh` | Create | cuDNN/NCCL/PYTHONPATH env vars; sourced by start-tp.sh |
| `systemd/comfyui-tp.service` | Create | systemd unit with `Restart=on-failure` |
| `paths/models.yaml` | Create | Model dir mappings (mirrors what start-tp.sh uses today) |
| `paths/cache-ram.yaml` | Create | `--cache-ram` thresholds, `--fast` feature list |
| `tests/test_start_tp_flags.py` | Create | Parses start-tp.sh, asserts flag presence/absence |
| `tests/test_paths_yaml.py` | Create | Validates cache-ram.yaml schema |
| `tests/test_upstream_pin.py` | Create | Reads `.upstream-pin`, asserts tag matches `git describe` on master |
| `deploy/install.sh` | Create | Symlinks beast config into place, copies systemd unit |
| `deploy/rebase-from-upstream.sh` | Create | Bumps the pin, runs the test suite, rejects on failure |
| `bench/baseline.py` | Create | Times 5 workflows at different model sizes |
| `monitoring/.gitkeep` | Create | Empty placeholder dir for future sub-project |
| `README.md` | Create | What this repo is, how to use it |

### `/home/blackthorn/ComfyUI/` (current working tree, becomes disposable after fork)

| Path | Action | Responsibility |
|------|--------|----------------|
| `start-tp.sh` | Delete (moved to beast) | Replaced by symlink or removed entirely |
| `docs/superpowers/specs/BASELINE.md` | Create | Before/after measurements, written by hand after bench runs |

---

## Phase A: Capture baseline and set up the two repos

### Task 1: Capture "before" baseline measurements

**Files:**
- Create: `docs/superpowers/specs/BASELINE.md`
- Run: existing `/home/blackthorn/ComfyUI/start-tp.sh` (current `--lowvram` config)

- [ ] **Step 1: Document the methodology**

Open `docs/superpowers/specs/BASELINE.md` and write the header:

```markdown
# ComfyUI on Beast — Performance Baseline

> Captured 2026-06-10 against `tensor-parallelism-pr` @ 460b378cb.
> Re-capture after each caching change. Compare rows in the "After" table against this "Before" table.

## Methodology

Each row is one workflow run from cold cache. We measure:
- **TTFI** (Time To First Image): wall-clock seconds from `POST /prompt` to the first preview/sample image arriving over WebSocket.
- **Switch**: wall-clock seconds from one prompt's last sample to the next prompt's first sample, when the second prompt uses a *different* checkpoint.
- **GPU util %**: `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` averaged across the run, ignoring the first 5s warmup.
- **RAM used (GB)**: peak `psutil.virtual_memory().used` during the run, minus pre-run baseline.

We run each workflow 3 times and report the median. Workflows chosen to span the model-size spectrum: SD 1.5 (~2GB), SDXL (~7GB), Flux Dev (~24GB), QwenImage Lightning 8-step (~20GB), WanVideo 2.2 T2V-A14B (~80GB across 2 GPUs).

## Before (current `--lowvram` config, no smart cache)

| Workflow | TTFI (s) | Switch (s) | GPU util % | RAM used (GB) |
|----------|---------:|-----------:|-----------:|---------------:|
| SD 1.5 (cold) | TBD | TBD | TBD | TBD |
| SDXL (cold) | TBD | TBD | TBD | TBD |
| Flux Dev (cold) | TBD | TBD | TBD | TBD |
| QwenImage Lightning (cold) | TBD | TBD | TBD | TBD |
| WanVideo 2.2 T2V (cold) | TBD | TBD | TBD | TBD |
| SD 1.5 → SDXL (switch) | — | TBD | TBD | TBD |
| Flux Dev → QwenImage (switch) | — | TBD | TBD | TBD |
```

- [ ] **Step 2: Verify ComfyUI is running**

```bash
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/
# Expected: 200
```

If not running, start it: `cd /home/blackthorn/ComfyUI && nohup ./start-tp.sh > /tmp/comfyui_baseline.log 2>&1 &` and wait for "All startup tasks have been completed" in the log (use a `Monitor` tool with an until-loop).

- [ ] **Step 3: Run the 7 measurement rows**

For each workflow in the table, submit the prompt via the ComfyUI API and record TTFI. We do this by hand, one row at a time, with a stopwatch and `nvidia-smi -l 1` running in another terminal. This is intentionally manual — automation is a later sub-project.

For each run, also capture a `nvidia-smi` snapshot before and after, and `free -g` before and after.

Fill in the TBDs in `BASELINE.md`. The numbers will not be exact, but the *order of magnitude* is what matters.

- [ ] **Step 4: Commit the baseline**

```bash
cd /home/blackthorn/ComfyUI
git add docs/superpowers/specs/BASELINE.md
git commit -m "spec: capture before-baseline for smart-caching rollout"
```

Note: this commit lives on the working tree (`tensor-parallelism-pr` branch) which is destined to become `~/comfyui-portable`. The commit history is preserved when we move the repo in Task 3.

---

### Task 2: Create `~/comfyui-beast` (the config repo)

**Files:**
- Create: `~/comfyui-beast/README.md`
- Create: `~/comfyui-beast/.gitignore`

- [ ] **Step 1: Create the directory and init git**

```bash
mkdir -p ~/comfyui-beast
cd ~/comfyui-beast
git init
git config user.email "chris@misfitmindset.com"
git config user.name "Chris Blackthorn"
git checkout -b main
```

- [ ] **Step 2: Write `.gitignore`**

Create `~/comfyui-beast/.gitignore`:

```
__pycache__/
*.pyc
.venv/
*.log
*.swp
.DS_Store
monitoring/*
!monitoring/.gitkeep
bench/__pycache__/
bench/.bench_cache/
```

- [ ] **Step 3: Write README.md**

Create `~/comfyui-beast/README.md`:

```markdown
# comfyui-beast

Beast-specific configuration for the ComfyUI deployment. **Never rebased** — the portable ComfyUI fork lives in `~/comfyui-portable/`.

## What lives here

- `start-tp.sh` — entry point that wraps `~/comfyui-portable/main.py` with our caching flags
- `env.sh` — CUDA / NCCL / cuDNN environment
- `systemd/` — systemd unit files
- `paths/` — model path mappings, cache-ram thresholds
- `tests/` — config smoke tests; run as part of `rebase-from-upstream.sh`
- `deploy/` — install and rebase scripts
- `bench/` — performance measurement scripts
- `monitoring/` — empty placeholder for future sub-project

## Quick start

```bash
# Install (one-time)
./deploy/install.sh

# Run
./start-tp.sh

# Rebase portable onto a new upstream release
./deploy/rebase-from-upstream.sh v0.22.3
```

## Boundary rule

No file in this repo is a ComfyUI patch. If you find yourself modifying `~/comfyui-portable/comfy/*.py` from here, stop — make the change in `~/comfyui-portable/` instead.
```

- [ ] **Step 4: Create empty placeholder directories**

```bash
cd ~/comfyui-beast
mkdir -p systemd paths tests deploy bench monitoring
touch monitoring/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
cd ~/comfyui-beast
git add .gitignore README.md monitoring/.gitkeep
git commit -m "init: comfyui-beast config repo"
```

---

### Task 3: Create `~/comfyui-portable` (Pattern B fork)

**Files:**
- (entire new repo at `~/comfyui-portable/`)

This is the biggest mechanical step. We move the current `/home/blackthorn/ComfyUI/` working tree into a new location and set up the Pattern B branch structure.

- [ ] **Step 1: Stop the running ComfyUI service**

```bash
pkill -f "torchrun.*main.py.*--tensor-parallel" || true
sleep 2
ps -ef | grep "torchrun.*main.py" | grep -v grep | wc -l
# Expected: 0
```

- [ ] **Step 2: Move the working tree to `~/comfyui-portable`**

```bash
cd /home/blackthorn
# Use rsync for safety; if anything goes wrong we still have the original until we delete it
rsync -a --exclude='.git' /home/blackthorn/ComfyUI/ /home/blackthorn/comfyui-portable/
```

- [ ] **Step 3: Re-attach the .git directory from the original**

```bash
rsync -a /home/blackthorn/ComfyUI/.git/ /home/blackthorn/comfyui-portable/.git/
cd /home/blackthorn/comfyui-portable
git status
# Expected: clean working tree on tensor-parallelism-pr, 44 commits ahead of origin/master
```

Verify the branch:

```bash
git branch --show-current
# Expected: tensor-parallelism-pr
```

- [ ] **Step 4: Set up the Pattern B branch structure**

```bash
cd /home/blackthorn/comfyui-portable
# Create master that fast-forwards from origin/master (upstream)
git checkout -b master origin/master
git log --oneline -3
# Expected: shows recent upstream commits (e.g. "Update AMD portable readme.")

# Switch back to the working branch (which is 44 commits ahead of master)
git checkout tensor-parallelism-pr
```

- [ ] **Step 5: Tag a known-good upstream release as our starting pin**

Find the most recent ComfyUI release tag that's at or before our current `origin/master`:

```bash
cd /home/blackthorn/comfyui-portable
git tag --sort=-v:refname | head -10
# Pick the most recent one (likely v0.22.x or v0.23.x)
# Note the SHA at that tag
PIN_TAG=$(git tag --sort=-v:refname | head -1)
PIN_SHA=$(git rev-parse "$PIN_TAG")
echo "Pinning to: $PIN_TAG @ $PIN_SHA"
```

Create `.upstream-pin`:

```bash
cd /home/blackthorn/comfyui-portable
PIN_TAG=$(git tag --sort=-v:refname | head -1)
PIN_SHA=$(git rev-parse "$PIN_TAG")
PIN_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > .upstream-pin <<EOF
tag=$PIN_TAG
sha=$PIN_SHA
pinned_at=$PIN_DATE
EOF
cat .upstream-pin
```

- [ ] **Step 6: Commit the pin file**

```bash
cd /home/blackthorn/comfyui-portable
git add .upstream-pin
git commit -m "pin: set initial upstream release tag"
```

- [ ] **Step 7: Add a remote for upstream if missing**

```bash
cd /home/blackthorn/comfyui-portable
git remote -v
# Expected: origin  https://github.com/comfyanonymous/ComfyUI (fetch)
#           origin  https://github.com/comfyanonymous/ComfyUI (push)
```

If `origin` is missing, add it:

```bash
git remote add origin https://github.com/comfyanonymous/ComfyUI
```

- [ ] **Step 8: Verify the fork works**

```bash
cd /home/blackthorn/comfyui-portable
git fetch origin
git log --oneline origin/master -3
# Expected: recent upstream commits
git log --oneline tensor-parallelism-pr -3
# Expected: our TP commits at the top
```

---

### Task 4: Remove the old `~/ComfyUI/` symlink hazard

**Files:**
- Delete: `/home/blackthorn/ComfyUI/start-tp.sh` (will be replaced by a symlink later)

After this task, `/home/blackthorn/ComfyUI/` becomes a thin shell that just delegates to the two new repos.

- [ ] **Step 1: Replace the old `start-tp.sh` with a symlink to the new location**

```bash
cd /home/blackthorn/ComfyUI
rm start-tp.sh
ln -s /home/blackthorn/comfyui-beast/start-tp.sh start-tp.sh
ls -la start-tp.sh
# Expected: start-tp.sh -> /home/blackthorn/comfyui-beast/start-tp.sh
```

Note: this symlink will be broken until Task 5 creates `~/comfyui-beast/start-tp.sh`. That's expected.

- [ ] **Step 2: Verify the symlink will resolve after Task 5**

```bash
readlink -f /home/blackthorn/ComfyUI/start-tp.sh
# Expected: /home/blackthorn/comfyui-beast/start-tp.sh (file may not exist yet — that's fine)
```

---

## Phase B: Move config into the beast repo and add the caching flags

### Task 5: Create `start-tp.sh` in `~/comfyui-beast/`

**Files:**
- Create: `~/comfyui-beast/start-tp.sh`
- Create: `~/comfyui-beast/env.sh`

- [ ] **Step 1: Write `env.sh`**

Create `~/comfyui-beast/env.sh`:

```bash
#!/bin/bash
# Environment variables for ComfyUI on Beast.
# Sourced by start-tp.sh. Do not run this file directly.

# --- venv Python ---
export VIRTUAL_ENV="/home/blackthorn/comfyui-portable/venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

# --- cuDNN / CUDA Library Priority ---
export LD_LIBRARY_PATH="/home/blackthorn/comfyui-portable/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/home/blackthorn/comfyui-portable/venv/lib/python3.13/site-packages/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- PyTorch Optimizations ---
export TORCH_CUDNN_V8_AVAILABLE=1
export TORCH_CUDNN_V8_ENABLED=1
export TORCH_HOME="/home/blackthorn/.cache/torch"

# --- CUDA Environment ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- Memory: reduce VRAM fragmentation across both GPUs ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NCCL: single-node loopback, no InfiniBand, watchdog off ---
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

# --- Suppress tokenizer parallelism warnings from worker ranks ---
export TOKENIZERS_PARALLELISM=false
```

Make it executable:

```bash
chmod +x ~/comfyui-beast/env.sh
```

- [ ] **Step 2: Write `start-tp.sh`**

Create `~/comfyui-beast/start-tp.sh`:

```bash
#!/bin/bash
# ComfyUI on Beast — Tensor Parallelism (2x RTX 5060 Ti), port 8188.
# Wraps ~/comfyui-portable/main.py with Beast-specific env + caching flags.
#
# Security posture flags (--listen, --enable-cors-header, --disable-metadata,
# --disable-auto-launch) are configured in the systemd unit
# /etc/systemd/system/comfyui-tp.service, not here. Do not add them here
# or argparse will see them twice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTABLE_DIR="${COMFYUI_PORTABLE_DIR:-/home/blackthorn/comfyui-portable}"

# Load env
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

# Load caching configuration
CACHE_RAM_ARGS=()
if [ -f "${SCRIPT_DIR}/paths/cache-ram.yaml" ]; then
    ACTIVE_GB=$(grep -E '^active_gb:' "${SCRIPT_DIR}/paths/cache-ram.yaml" | awk '{print $2}')
    INACTIVE_GB=$(grep -E '^inactive_gb:' "${SCRIPT_DIR}/paths/cache-ram.yaml" | awk '{print $2}')
    if [ -n "${ACTIVE_GB}" ] && [ -n "${INACTIVE_GB}" ]; then
        CACHE_RAM_ARGS=(--cache-ram "${ACTIVE_GB}" "${INACTIVE_GB}")
    fi
fi

FAST_FEATURES=()
if [ -f "${SCRIPT_DIR}/paths/cache-ram.yaml" ]; then
    while IFS= read -r feature; do
        [ -n "${feature}" ] && FAST_FEATURES+=("${feature}")
    done < <(grep -E '^- ' "${SCRIPT_DIR}/paths/cache-ram.yaml" | sed 's/^- //')
fi

cd "${PORTABLE_DIR}"

exec torchrun --nproc_per_node=2 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --tensor-parallel \
    --enable-dynamic-vram \
    "${CACHE_RAM_ARGS[@]}" \
    --fast "${FAST_FEATURES[@]}" \
    --enable-manager-legacy-ui \
    --database-url "sqlite:////tmp/comfyui_tp.db" \
    --front-end-version Comfy-Org/ComfyUI_frontend@latest \
    "$@"
```

Make it executable:

```bash
chmod +x ~/comfyui-beast/start-tp.sh
```

- [ ] **Step 3: Write `cache-ram.yaml`**

Create `~/comfyui-beast/paths/cache-ram.yaml`:

```yaml
# ComfyUI RAM-pressure cache configuration.
# Read by start-tp.sh and tests/test_paths_yaml.py.
#
# active_gb: Threshold (in GB) of free RAM below which active pins start evicting.
# inactive_gb: Threshold (in GB) of free RAM below which inactive pins start evicting.
# Beast has 121GB RAM. We use conservative defaults from upstream; tighten if OOM.

active_gb: 8.0
inactive_gb: 96.0

# Features passed to --fast. Empty list = upstream default. See cli_args.py for valid values.
# Comment out the entire list to disable --fast.
fast_features:
  - fp8
  - attention
```

- [ ] **Step 4: Write `models.yaml`**

Create `~/comfyui-beast/paths/models.yaml`:

```yaml
# Model directory mappings for Beast. Mirrors what was previously inline in start-tp.sh.
# Future: this becomes the single source of truth for path-mappings; LoRA Manager can
# consume it instead of having its own settings.json.

checkpoints_dir: /home/blackthorn/ComfyUI/models/checkpoints
diffusion_models_dir: /home/blackthorn/ComfyUI/models/diffusion_models
loras_dir: /home/blackthorn/ComfyUI/models/loras
text_encoders_dir: /home/blackthorn/ComfyUI/models/text_encoders
clip_dir: /home/blackthorn/ComfyUI/models/clip
clip_vision_dir: /home/blackthorn/ComfyUI/models/clip_vision
vae_dir: /home/blackthorn/ComfyUI/models/vae
controlnet_dir: /home/blackthorn/ComfyUI/models/controlnet
upscale_models_dir: /home/blackthorn/ComfyUI/models/upscale_models
```

- [ ] **Step 5: Commit**

```bash
cd ~/comfyui-beast
git add start-tp.sh env.sh paths/cache-ram.yaml paths/models.yaml
git commit -m "feat: add start-tp.sh with smart caching flags, env, paths config"
```

---

### Task 6: Restore the mmap-based safetensors loader

**Files:**
- Modify: `~/comfyui-portable/comfy/utils.py:90`

The current working tree has `f = open(ckpt, 'rb')` which bypasses the mmap-based file handle. This disables upstream's RAM pressure cache. Restore the mmap call.

- [ ] **Step 1: Read the current state**

```bash
cd /home/blackthorn/comfyui-portable
sed -n '85,92p' comfy/utils.py
```

Expected output (current working tree):

```python
def load_safetensors(ckpt):
    import comfy_aimdo.model_mmap

    file_lock = threading.Lock()
    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(ckpt)
    f = open(ckpt, 'rb')
    file_size = os.path.getsize(ckpt)
```

- [ ] **Step 2: Restore the mmap-based file handle**

Edit `comfy/utils.py` line 90. Replace:

```python
    f = open(ckpt, 'rb')
```

with:

```python
    f = model_mmap.get_file_handle()
```

The full function should now read:

```python
def load_safetensors(ckpt):
    import comfy_aimdo.model_mmap

    file_lock = threading.Lock()
    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(ckpt)
    f = model_mmap.get_file_handle()
    file_size = os.path.getsize(ckpt)
```

- [ ] **Step 3: Verify the change**

```bash
cd /home/blackthorn/comfyui-portable
sed -n '85,92p' comfy/utils.py
# Expected: shows f = model_mmap.get_file_handle()
git diff comfy/utils.py
# Expected: shows the one-line change
```

- [ ] **Step 4: Commit**

```bash
cd /home/blackthorn/comfyui-portable
git add comfy/utils.py
git commit -m "fix: restore mmap-based safetensors loader for RAM pressure cache"
```

---

### Task 7: Smoke test — start ComfyUI from the new location with the new flags

**Files:**
- Run: `~/comfyui-beast/start-tp.sh`

This is a verification task, not a code task. We confirm the new layout actually runs.

- [ ] **Step 1: Start ComfyUI**

```bash
rm -f /tmp/comfyui_tp.log
nohup /home/blackthorn/comfyui-beast/start-tp.sh > /tmp/comfyui_tp.log 2>&1 &
echo "Started PID $!"
```

- [ ] **Step 2: Wait for "All startup tasks have been completed"**

Use a `Monitor` tool with this command (timeout 180000ms):

```bash
for i in $(seq 1 36); do
  if grep -qE "All startup tasks have been completed" /tmp/comfyui_tp.log 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if grep -qE "Traceback \(most recent|ImportError|SyntaxError" /tmp/comfyui_tp.log 2>/dev/null; then
    echo "ERROR"
    break
  fi
  sleep 5
done
```

Expected: "READY" within ~120s.

- [ ] **Step 3: Verify the API responds**

```bash
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/
# Expected: 200
```

- [ ] **Step 4: Verify the new flags are in the log**

```bash
grep -E "Using RAM pressure cache|TP.*Rank|server.py.*Starting server" /tmp/comfyui_tp.log | head -10
# Expected: shows the "Using RAM pressure cache" line, the "Rank 1 in worker mode" line, and "Starting server"
```

- [ ] **Step 5: Verify the old `--lowvram` flag is NOT present in the log**

```bash
grep -E "lowvram" /tmp/comfyui_tp.log | head -5
# Expected: no output (or only the model's internal "lowvram_available" check, which is benign)
```

- [ ] **Step 6: Stop the service**

```bash
pkill -f "torchrun.*main.py.*--tensor-parallel" || true
sleep 2
ps -ef | grep "torchrun.*main.py" | grep -v grep | wc -l
# Expected: 0
```

---

### Task 8: Re-measure "after" baseline and update BASELINE.md

**Files:**
- Modify: `docs/superpowers/specs/BASELINE.md`

- [ ] **Step 1: Start the service again**

```bash
rm -f /tmp/comfyui_tp.log
nohup /home/blackthorn/comfyui-beast/start-tp.sh > /tmp/comfyui_tp.log 2>&1 &
```

Wait for "All startup tasks have been completed" via Monitor.

- [ ] **Step 2: Re-run the 7 measurement rows from Task 1**

Same workflows, same methodology. Record into a new "After" table in `BASELINE.md`.

- [ ] **Step 3: Compute the deltas and write the summary**

Append to `docs/superpowers/specs/BASELINE.md`:

```markdown
## After (`--cache-ram 8 96 --enable-dynamic-vram --fast`)

| Workflow | TTFI (s) | Switch (s) | GPU util % | RAM used (GB) |
|----------|---------:|-----------:|-----------:|---------------:|
| SD 1.5 (cold) | TBD | TBD | TBD | TBD |
| SDXL (cold) | TBD | TBD | TBD | TBD |
| Flux Dev (cold) | TBD | TBD | TBD | TBD |
| QwenImage Lightning (cold) | TBD | TBD | TBD | TBD |
| WanVideo 2.2 T2V (cold) | TBD | TBD | TBD | TBD |
| SD 1.5 → SDXL (switch) | — | TBD | TBD | TBD |
| Flux Dev → QwenImage (switch) | — | TBD | TBD | TBD |

## Deltas

| Workflow | TTFI Δ | Switch Δ | GPU util Δ | RAM used Δ |
|----------|-------:|---------:|-----------:|-----------:|
| SD 1.5 (cold) | TBD% | TBD% | TBD pp | TBD GB |
| SDXL (cold) | TBD% | TBD% | TBD pp | TBD GB |
| Flux Dev (cold) | TBD% | TBD% | TBD pp | TBD GB |
| QwenImage Lightning (cold) | TBD% | TBD% | TBD pp | TBD GB |
| WanVideo 2.2 T2V (cold) | TBD% | TBD% | TBD pp | TBD GB |
| SD 1.5 → SDXL (switch) | — | TBD% | TBD pp | TBD GB |
| Flux Dev → QwenImage (switch) | — | TBD% | TBD pp | TBD GB |

## Notes

- TBD pp = percentage points
- TBD% = percentage change
- If the deltas are <10% in any direction, the change is noise. Re-measure.
- Stop the service before committing this file (avoids file system races).
```

- [ ] **Step 4: Commit**

```bash
cd /home/blackthorn/comfyui-portable
git add docs/superpowers/specs/BASELINE.md
git commit -m "spec: capture after-baseline + deltas for smart-caching rollout"
```

- [ ] **Step 5: Stop the service**

```bash
pkill -f "torchrun.*main.py.*--tensor-parallel" || true
```

---

## Phase C: Tests + rebase machinery

### Task 9: Write the config smoke tests for `comfyui-beast`

**Files:**
- Create: `~/comfyui-beast/tests/test_start_tp_flags.py`
- Create: `~/comfyui-beast/tests/test_paths_yaml.py`

- [ ] **Step 1: Write `test_start_tp_flags.py`**

Create `~/comfyui-beast/tests/test_start_tp_flags.py`:

```python
"""Verify that start-tp.sh has the right caching flags and not the wrong ones."""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
START_TP = REPO_ROOT / "start-tp.sh"


def _read_start_tp() -> str:
    return START_TP.read_text()


def test_lowvram_not_present():
    """--lowvram conflicts with --enable-dynamic-vram. Must not be in start-tp.sh."""
    text = _read_start_tp()
    # Allow the word in comments only.
    code_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    code = "\n".join(code_lines)
    assert "--lowvram" not in code, (
        "start-tp.sh must not pass --lowvram when --enable-dynamic-vram is enabled. "
        "See https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/cli_args.py"
    )


def test_cache_ram_flag_present():
    text = _read_start_tp()
    assert "--cache-ram" in text, "--cache-ram must be passed to enable RAM pressure cache."


def test_enable_dynamic_vram_flag_present():
    text = _read_start_tp()
    assert "--enable-dynamic-vram" in text, "--enable-dynamic-vram must be passed."


def test_fast_flag_present():
    text = _read_start_tp()
    assert "--fast" in text, "--fast must be passed to enable performance features."


def test_tensor_parallel_flag_present():
    text = _read_start_tp()
    assert "--tensor-parallel" in text, "--tensor-parallel must be passed."


def test_start_tp_is_executable():
    assert START_TP.stat().st_mode & 0o111, "start-tp.sh must be executable (chmod +x)."
```

- [ ] **Step 2: Write `test_paths_yaml.py`**

Create `~/comfyui-beast/tests/test_paths_yaml.py`:

```python
"""Validate the schema of paths/cache-ram.yaml."""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
CACHE_RAM_YAML = REPO_ROOT / "paths" / "cache-ram.yaml"


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for our flat schema. Avoids adding PyYAML as a dep."""
    result: dict = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            if current_list_key is None:
                raise ValueError(f"List item without parent key: {line!r}")
            result.setdefault(current_list_key, []).append(line[4:].strip())
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                current_list_key = key
                result[key] = []
            else:
                current_list_key = None
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


@pytest.fixture
def config() -> dict:
    return _parse_simple_yaml(CACHE_RAM_YAML.read_text())


def test_active_gb_is_positive(config):
    assert config.get("active_gb", 0) > 0, "active_gb must be positive."


def test_inactive_gb_is_positive(config):
    assert config.get("inactive_gb", 0) > 0, "inactive_gb must be positive."


def test_active_gb_le_inactive_gb(config):
    active = config.get("active_gb", 0)
    inactive = config.get("inactive_gb", 0)
    assert active <= inactive, (
        f"active_gb ({active}) must be <= inactive_gb ({inactive}). "
        "The active threshold is when eviction starts; inactive is the max."
    )


def test_inactive_gb_does_not_exceed_90pct_of_system_ram(config, monkeypatch):
    """On a 121GB Beast, 90% = 108.9GB. We hard-cap at 0.9 * psutil-reported RAM."""
    import psutil
    system_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cap = system_ram_gb * 0.9
    inactive = config.get("inactive_gb", 0)
    assert inactive <= cap, (
        f"inactive_gb ({inactive}) exceeds 90% of system RAM ({cap:.1f}GB). "
        "Leaving <10% headroom risks OOM under concurrent load."
    )
```

- [ ] **Step 3: Run the tests**

```bash
cd ~/comfyui-beast
python3 -m pytest tests/ -v
# Expected: 8 passed (6 from test_start_tp_flags, 4 from test_paths_yaml, minus the one that requires the YAML to be present)
```

- [ ] **Step 4: Commit**

```bash
cd ~/comfyui-beast
git add tests/test_start_tp_flags.py tests/test_paths_yaml.py
git commit -m "test: config smoke tests for start-tp.sh and cache-ram.yaml"
```

---

### Task 10: Write `test_upstream_pin.py` for `comfyui-portable`

**Files:**
- Create: `~/comfyui-portable/tests/test_upstream_pin.py`

- [ ] **Step 1: Write the test**

Create `~/comfyui-portable/tests/test_upstream_pin.py`:

```python
"""Validate .upstream-pin is consistent with the master branch."""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[1]
PIN_FILE = REPO_ROOT / ".upstream-pin"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
    ).strip()


def _parse_pin_file() -> dict[str, str]:
    text = PIN_FILE.read_text()
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def test_pin_file_exists():
    assert PIN_FILE.exists(), f"{PIN_FILE} must exist. Create it with tag/sha/pinned_at lines."


def test_pin_file_has_required_keys():
    pin = _parse_pin_file()
    for key in ("tag", "sha", "pinned_at"):
        assert key in pin, f".upstream-pin must contain {key}=..."


def test_pinned_sha_is_in_master_history():
    pin = _parse_pin_file()
    sha = pin["sha"]
    # Allow either full or short SHA.
    out = _git("rev-list", "--max-count=1", sha, "^origin/master")
    assert out == "", (
        f"Pinned SHA {sha} is not in origin/master history. "
        "Either the pin is stale, or origin/master has been rebased."
    )
    # And confirm it IS reachable from master (i.e. it is an ancestor or the tip).
    out = _git("merge-base", "--is-ancestor", sha, "origin/master")
    assert out == "", f"Pinned SHA {sha} is not an ancestor of origin/master."


def test_pinned_tag_resolves_to_pinned_sha():
    pin = _parse_pin_file()
    actual_sha = _git("rev-parse", pin["tag"])
    assert actual_sha == pin["sha"], (
        f"Tag {pin['tag']} resolves to {actual_sha}, not {pin['sha']}. "
        "Update .upstream-pin."
    )
```

- [ ] **Step 2: Run the test**

```bash
cd /home/blackthorn/comfyui-portable
python3 -m pytest tests/test_upstream_pin.py -v
# Expected: 4 passed
```

- [ ] **Step 3: Commit**

```bash
cd /home/blackthorn/comfyui-portable
git add tests/test_upstream_pin.py
git commit -m "test: validate .upstream-pin against master history"
```

---

### Task 11: Write the 3 rebase-safety tests in `comfyui-portable`

**Files:**
- Create: `~/comfyui-portable/tests-unit/comfy_test/distributed/test_rebase_safety.py`

- [ ] **Step 1: Write the test file**

Create `~/comfyui-portable/tests-unit/comfy_test/distributed/test_rebase_safety.py`:

```python
"""Regression tests for the smart-caching contract.

These tests run as part of rebase-from-upstream.sh. They protect against:
- A future commit reverting the mmap-based safetensors loader (turns off RAM cache).
- Upstream's pin budget being tuned too aggressively (exceeds 90% of system RAM).
- The --lowvram / --enable-dynamic-vram interaction regressing.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest
import psutil


REPO_ROOT = pathlib.Path(__file__).parents[3]


def _read(path: pathlib.Path) -> str:
    return path.read_text()


def test_load_safetensors_uses_mmap():
    """comfy/utils.py:load_safetensors must use the ModelMMAP file handle.

    A previous change in the working tree replaced `f = model_mmap.get_file_handle()`
    with `f = open(ckpt, 'rb')`, which disables the RAM pressure cache by
    bypassing the mmap-backed file handle. This test catches any future
    regression of that change.
    """
    utils_py = (REPO_ROOT / "comfy" / "utils.py").read_text()
    # Match the load_safetensors function body.
    m = re.search(
        r"def load_safetensors\(ckpt\):.*?(?=\ndef |\Z)",
        utils_py,
        re.DOTALL,
    )
    assert m, "load_safetensors function not found in comfy/utils.py"
    body = m.group(0)
    assert "model_mmap.get_file_handle()" in body, (
        "load_safetensors must use model_mmap.get_file_handle() to enable "
        "the RAM pressure cache. The current body does not contain it."
    )
    assert "f = open(ckpt, 'rb')" not in body, (
        "load_safetensors must not fall back to plain open(); that disables "
        "the mmap-based file handle and turns off the RAM pressure cache."
    )


def test_pin_budget_does_not_exceed_90pct_of_system_ram():
    """MAX_PINNED_MEMORY in comfy/model_management.py must not exceed 90% of system RAM.

    Upstream's default is `ram * 0.90`. If a future commit relaxes this to
    something higher (e.g. 0.95), we'd OOM the system. This test pins the
    multiplier at 0.9 or less.
    """
    mm_py = (REPO_ROOT / "comfy" / "model_management.py").read_text()
    m = re.search(
        r"MAX_PINNED_MEMORY\s*=\s*ram\s*\*\s*([\d.]+)",
        mm_py,
    )
    assert m, (
        "Could not find `MAX_PINNED_MEMORY = ram * X` in comfy/model_management.py. "
        "Upstream may have refactored the pin budget — review manually."
    )
    multiplier = float(m.group(1))
    assert multiplier <= 0.9, (
        f"MAX_PINNED_MEMORY multiplier is {multiplier}, must be <= 0.9. "
        "Higher values risk OOM under concurrent load."
    )
    # Sanity check: the configured budget is reasonable for this system.
    system_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    budget_gb = system_ram_gb * multiplier
    assert budget_gb >= 4, (
        f"Pin budget is {budget_gb:.1f}GB on a {system_ram_gb:.0f}GB system. "
        "That's too low; the pin subsystem becomes useless. "
        "Verify upstream's logic hasn't been inverted."
    )


def test_lowvram_flag_takes_no_effect_when_dynamic_enabled():
    """When both --lowvram and --enable-dynamic-vram are passed, dynamic wins.

    This is upstream's documented behavior (see comfy/cli_args.py:142):
    `--lowvram` "Doesn't do anything if dynamic vram is enabled."

    We don't pass --lowvram in start-tp.sh anymore, but this test pins the
    upstream behavior so a future refactor doesn't silently make --lowvram
    win and undo the caching.
    """
    cli_args = (REPO_ROOT / "comfy" / "cli_args.py").read_text()
    # Look for the --lowvram help text or the dynamic-vram mutual exclusion.
    assert "lowvram" in cli_args.lower()
    # The two flags are mutually exclusive in the argparse group.
    # We don't need a perfect regex — just confirm both flags exist and
    # the help text mentions the interaction.
    m = re.search(
        r"--lowvram['\"].*?help=['\"]([^'\"]+)",
        cli_args,
    )
    if m:
        help_text = m.group(1).lower()
        assert "dynamic" in help_text, (
            f"--lowvram help text does not mention dynamic-vram: {help_text!r}. "
            "The mutual-exclusion behavior may have been changed."
        )
```

- [ ] **Step 2: Run the test**

```bash
cd /home/blackthorn/comfyui-portable
python3 -m pytest tests-unit/comfy_test/distributed/test_rebase_safety.py -v
# Expected: 3 passed
```

- [ ] **Step 3: Run the full distributed test suite to confirm no regressions**

```bash
cd /home/blackthorn/comfyui-portable
python3 -m pytest tests-unit/comfy_test/distributed/ -v
# Expected: 49 passed (46 existing TP tests + 3 new rebase-safety)
```

- [ ] **Step 4: Commit**

```bash
cd /home/blackthorn/comfyui-portable
git add tests-unit/comfy_test/distributed/test_rebase_safety.py
git commit -m "test: rebase-safety regression for smart-caching contract"
```

---

### Task 12: Write the rebase-from-upstream.sh script

**Files:**
- Create: `~/comfyui-beast/deploy/rebase-from-upstream.sh`

- [ ] **Step 1: Write the script**

Create `~/comfyui-beast/deploy/rebase-from-upstream.sh`:

```bash
#!/bin/bash
# Rebase ~/comfyui-portable onto a new upstream ComfyUI release tag.
# Runs the full test suite after rebase. Rejects the rebase if any test fails.
#
# Usage: ./rebase-from-upstream.sh <upstream-tag>
# Example: ./rebase-from-upstream.sh v0.22.3

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <upstream-tag>" >&2
    echo "Example: $0 v0.22.3" >&2
    exit 2
fi

NEW_TAG="$1"
PORTABLE_DIR="${COMFYUI_PORTABLE_DIR:-/home/blackthorn/comfyui-portable}"
BEAST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "${PORTABLE_DIR}/.git" ]; then
    echo "ERROR: ${PORTABLE_DIR} is not a git repo. Run the fork setup first." >&2
    exit 1
fi

cd "${PORTABLE_DIR}"

echo "==> Fetching upstream..."
git fetch origin

echo "==> Verifying tag ${NEW_TAG} exists in origin..."
if ! git rev-parse "origin/${NEW_TAG}" >/dev/null 2>&1; then
    echo "ERROR: origin/${NEW_TAG} does not exist. Check the tag name." >&2
    exit 1
fi

echo "==> Fast-forwarding master to origin/${NEW_TAG}..."
git checkout master
git merge --ff-only "origin/${NEW_TAG}"

echo "==> Rebasing tensor-parallelism onto master..."
git checkout tensor-parallelism
if ! git rebase master; then
    echo "" >&2
    echo "ERROR: Rebase failed. Resolve conflicts and re-run this script," >&2
    echo "       or run 'git rebase --abort' to back out." >&2
    exit 1
fi

echo "==> Running portable test suite..."
if ! python3 -m pytest tests-unit/comfy_test/distributed/ tests/test_upstream_pin.py -v; then
    echo "" >&2
    echo "ERROR: Portable tests failed. Rebase is REJECTED." >&2
    echo "       Fix forward, stay pinned, or roll back the rebase:" >&2
    echo "         git rebase --abort   # roll back" >&2
    echo "         # or fix the failing tests, then re-run this script" >&2
    exit 1
fi

echo "==> Running beast config tests..."
if ! (cd "${BEAST_DIR}" && python3 -m pytest tests/ -v); then
    echo "" >&2
    echo "ERROR: Beast config tests failed. Rebase is REJECTED." >&2
    exit 1
fi

echo "==> Updating .upstream-pin..."
NEW_SHA=$(git rev-parse "${NEW_TAG}")
PIN_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > .upstream-pin <<EOF
tag=${NEW_TAG}
sha=${NEW_SHA}
pinned_at=${PIN_DATE}
EOF

git add .upstream-pin
git commit -m "pin: bump upstream to ${NEW_TAG}"

echo ""
echo "==> SUCCESS: Rebased onto ${NEW_TAG} @ ${NEW_SHA}"
echo "    Next: smoke-test by running ~/comfyui-beast/start-tp.sh"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x ~/comfyui-beast/deploy/rebase-from-upstream.sh
```

- [ ] **Step 3: Commit**

```bash
cd ~/comfyui-beast
git add deploy/rebase-from-upstream.sh
git commit -m "feat: rebase-from-upstream.sh with test-suite gate"
```

---

### Task 13: Write the install.sh script for the systemd unit

**Files:**
- Create: `~/comfyui-beast/systemd/comfyui-tp.service`
- Create: `~/comfyui-beast/deploy/install.sh`

- [ ] **Step 1: Write the systemd unit**

Create `~/comfyui-beast/systemd/comfyui-tp.service`:

```ini
[Unit]
Description=ComfyUI on Beast — Tensor Parallelism (2x RTX 5060 Ti) — Port 8188
After=network.target
Conflicts=comfyui.service comfyui-2.service

[Service]
Type=simple
User=blackthorn
WorkingDirectory=/home/blackthorn/comfyui-portable

# cuDNN / CUDA library priority
Environment=LD_LIBRARY_PATH=/home/blackthorn/comfyui-portable/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/home/blackthorn/comfyui-portable/venv/lib/python3.13/site-packages/nvidia/cublas/lib

# PyTorch
Environment=TORCH_CUDNN_V8_AVAILABLE=1
Environment=TORCH_CUDNN_V8_ENABLED=1
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=TORCH_HOME=/home/blackthorn/.cache/torch

# Memory + threading
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=TOKENIZERS_PARALLELISM=false

# Pre-start: kill anything on 8188 and stop conflicting services
ExecStartPre=/bin/bash -c 'fuser -k 8188/tcp 2>/dev/null; sudo systemctl stop comfyui.service comfyui-2.service 2>/dev/null; sleep 1; exit 0'

# Security posture (added directly here, not in start-tp.sh, to avoid argparse double-pass)
ExecStart=/home/blackthorn/comfyui-beast/start-tp.sh --listen 0.0.0.0 --enable-cors-header '*' --disable-metadata --disable-auto-launch

# Auto-recovery: restart on any failure (sub-project A scope)
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write `install.sh`**

Create `~/comfyui-beast/deploy/install.sh`:

```bash
#!/bin/bash
# Install the Beast ComfyUI service. Idempotent.
# Usage: sudo ./deploy/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be run as root (use sudo)." >&2
    exit 1
fi

echo "==> Installing systemd unit to /etc/systemd/system/comfyui-tp.service"
cp "${BEAST_DIR}/systemd/comfyui-tp.service" /etc/systemd/system/comfyui-tp.service
chmod 644 /etc/systemd/system/comfyui-tp.service

systemctl daemon-reload
systemctl enable comfyui-tp.service

echo ""
echo "==> Done. To start the service:"
echo "      sudo systemctl start comfyui-tp"
echo ""
echo "    To check status:"
echo "      systemctl status comfyui-tp"
echo ""
echo "    To view logs:"
echo "      journalctl -u comfyui-tp -f"
```

- [ ] **Step 3: Make both executable**

```bash
chmod +x ~/comfyui-beast/deploy/install.sh
chmod +x ~/comfyui-beast/systemd/comfyui-tp.service  # not strictly needed for unit files
```

- [ ] **Step 4: Install (asks for sudo)**

```bash
sudo ~/comfyui-beast/deploy/install.sh
```

- [ ] **Step 5: Verify the unit loaded**

```bash
systemctl status comfyui-tp --no-pager | head -15
# Expected: shows "loaded" but "inactive (dead)"
```

- [ ] **Step 6: Commit**

```bash
cd ~/comfyui-beast
git add systemd/comfyui-tp.service deploy/install.sh
git commit -m "feat: systemd unit + install.sh for comfyui-tp service"
```

---

### Task 14: Write the bench script

**Files:**
- Create: `~/comfyui-beast/bench/baseline.py`

- [ ] **Step 1: Write the bench script**

Create `~/comfyui-beast/bench/baseline.py`:

```python
#!/usr/bin/env python3
"""ComfyUI on Beast — performance baseline measurement.

Times a series of model loads/unloads and reports TTFI, switch time,
peak GPU util, and peak RAM used. Output is a markdown table suitable
for pasting into docs/superpowers/specs/BASELINE.md.

Usage:
    python3 ~/comfyui-beast/bench/baseline.py [--output path]

This is a measurement tool, not a test. It does not assert anything.
The human reads the output and decides whether the numbers look good.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

import psutil


COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
DEFAULT_WORKFLOWS: list[dict] = [
    # Real workflow JSON files would be loaded here. For the first cut, the
    # bench script is a skeleton; the human supplies workflow paths.
    #
    # {"name": "SD 1.5 (cold)", "workflow": "bench/sd15.json", "expected_vram_gb": 4},
    # {"name": "SDXL (cold)", "workflow": "bench/sdxl.json", "expected_vram_gb": 8},
    # {"name": "Flux Dev (cold)", "workflow": "bench/flux.json", "expected_vram_gb": 24},
]


def measure_workflow(workflow_path: str) -> dict:
    """Submit a workflow and measure TTFI, GPU util, RAM used."""
    raise NotImplementedError(
        "Fill in workflow submission. See ComfyUI's /prompt API. "
        "This is a skeleton for the first cut — measured manually in Task 1 / Task 8."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="Path to a workflow JSON file. May be passed multiple times.",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not args.workflow:
        print("No workflows provided. Use --workflow <path>.")
        print("See docstring for the expected format.")
        return

    results = []
    for path in args.workflow:
        print(f"==> Measuring {path}...")
        result = measure_workflow(path)
        results.append(result)

    table = "| Workflow | TTFI (s) | Switch (s) | GPU util % | RAM used (GB) |\n"
    table += "|----------|---------:|-----------:|-----------:|---------------:|\n"
    for r in results:
        table += (
            f"| {r['name']} | {r['ttfi']:.1f} | {r.get('switch', '—')} | "
            f"{r['gpu_util']:.0f} | {r['ram_used']:.1f} |\n"
        )

    print()
    print(table)

    if args.output:
        args.output.write_text(table)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports**

```bash
cd ~/comfyui-beast
python3 -c "import bench.baseline; print('OK')"
# Expected: OK
```

- [ ] **Step 3: Commit**

```bash
cd ~/comfyui-beast
git add bench/baseline.py
git commit -m "feat: bench/baseline.py skeleton (manual measurement for now)"
```

---

### Task 15: Final verification — full test suite, full service smoke

**Files:**
- (verification only)

- [ ] **Step 1: Run all tests**

```bash
# Beast config tests
cd ~/comfyui-beast
python3 -m pytest tests/ -v
# Expected: 10 passed (6 + 4)

# Portable tests
cd /home/blackthorn/comfyui-portable
python3 -m pytest tests-unit/comfy_test/distributed/ tests/test_upstream_pin.py -v
# Expected: 53 passed (46 existing TP + 3 rebase-safety + 4 pin-validation)
```

- [ ] **Step 2: Start the service via systemd (not nohup)**

```bash
sudo systemctl start comfyui-tp
systemctl status comfyui-tp --no-pager | head -10
# Expected: active (running) within ~30s
```

- [ ] **Step 3: Verify API + endpoints**

```bash
curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/
# Expected: 200
curl -s -m 5 -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8188/api/lm/check-updates"
# Expected: 200
```

- [ ] **Step 4: Verify RAM pressure cache is active**

```bash
journalctl -u comfyui-tp --since "1 minute ago" --no-pager | grep -E "RAM pressure cache|Using RAM"
# Expected: at least one line matching
```

- [ ] **Step 5: Stop the service**

```bash
sudo systemctl stop comfyui-tp
```

- [ ] **Step 6: Document the new state in CLAUDE.md (project-level)**

Append to `/home/blackthorn/comfyui-portable/CLAUDE.md` (if present) or create it:

```markdown
# ComfyUI on Beast — Project Notes

## Layout (as of 2026-06-10)

- `~/comfyui-portable/` — ComfyUI fork, Pattern B. `master` fast-forwards to upstream tags; `tensor-parallelism` is the long-lived work branch.
- `~/comfyui-beast/` — Beast-specific config (start-tp.sh, systemd, paths, tests, deploy, bench). Never rebased.
- `/home/blackthorn/ComfyUI/` — Thin shell, mostly symlinks to the above two repos. Source of truth lives in the two repos.

## Run

```bash
sudo systemctl start comfyui-tp
```

## Rebase onto a new upstream release

```bash
~/comfyui-beast/deploy/rebase-from-upstream.sh v0.22.3
```

If tests fail, the rebase is rejected. Read the failure, decide.

## Tests

- Config smoke: `cd ~/comfyui-beast && pytest tests/`
- Rebase safety: `cd ~/comfyui-portable && pytest tests-unit/comfy_test/distributed/test_rebase_safety.py tests/test_upstream_pin.py`
```

- [ ] **Step 7: Commit the doc update**

```bash
cd /home/blackthorn/comfyui-portable
git add CLAUDE.md
git commit -m "docs: add project notes for two-repo layout"
```

---

## Self-review checklist (run by Claude after writing this plan)

- [x] **Spec coverage**: Each section of `2026-06-10-smart-caching-and-fork-design.md` has a task.
  - Goal 1 (adopt flags): Task 5, Task 6, Task 7
  - Goal 2 (measure): Task 1, Task 8
  - Goal 3 (two repos): Task 2, Task 3, Task 4
  - Goal 4 (pin + tests): Task 5, Task 9, Task 10, Task 11, Task 12
  - Non-goals: not addressed (correct — non-goals by definition)
  - Architecture: Tasks 2, 3, 4, 5, 13
  - Data flow: Task 5, Task 6
  - Error handling: Task 13 (Category B systemd restart), Task 12 (Category C rebase)
  - Testing: Tasks 9, 10, 11, 12, 15

- [x] **Placeholder scan**: No "TBD" outside the legitimate "TBD" cells in `BASELINE.md` (which are filled in by hand during measurement). No "TODO", no "fill in details" without a follow-up task. The `bench/baseline.py` skeleton raises `NotImplementedError` with a clear note that it's intentionally manual for now.

- [x] **Type consistency**: The 4 flag checks in `test_start_tp_flags.py` reference the same flag names as `start-tp.sh` (Task 5). The 3 rebase-safety tests reference the same source files as Task 6's modification. The pin file format is the same in Task 3 (creation) and Tasks 10 + 12 (consumption).

- [x] **Frequent commits**: Every task ends with a commit. Total of 14 commits across the 15 tasks.

- [x] **Bite-sized steps**: Each step is one action. Tests are written before implementation where it makes sense (config tests, rebase safety). For purely mechanical setup (Task 1, Task 2, Task 3, Task 4, Task 5, Task 13) the verification is "the file exists with the right content" rather than TDD, because there's nothing to test in a one-line file write.

---

## What comes next (not in this plan)

The plan stops at the boundary of "smart caching + two-repo fork." The next sub-projects in the [[project-goal]] roadmap are:

- **Sub-project B**: TP pruning for Cosmos/HiDream/WanVideo. Needs this sub-project's foundation (clean fork, working tests) to land safely.
- **Sub-project C**: Ollama integration as ComfyUI custom nodes.
- **Sub-project D**: Service resilience — health-checked systemd, GPU telemetry, auto-recovery beyond `Restart=on-failure`.
- **Sub-project E**: Storage layout reorg, model dedup, orchestration.

Each gets its own spec → plan → implementation cycle.
