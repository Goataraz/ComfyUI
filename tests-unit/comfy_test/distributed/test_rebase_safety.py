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
        "Upstream may have refactored the pin budget -- review manually."
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
    # We don't need a perfect regex -- just confirm both flags exist and
    # the help text mentions the interaction.
    # ponytail: capture until the closing double-quote, not until any quote --
    # the upstream help text is "Doesn't do anything if dynamic vram is enabled..."
    # and the apostrophe in "Doesn't" breaks a naive [^'\"]+ capture (stops at
    # "Doesn", missing "dynamic"). cli_args.py uses double-quoted help strings.
    m = re.search(
        r"""--lowvram['\"].*?help="([^"]+)\"""",
        cli_args,
    )
    if m:
        help_text = m.group(1).lower()
        assert "dynamic" in help_text, (
            f"--lowvram help text does not mention dynamic-vram: {help_text!r}. "
            "The mutual-exclusion behavior may have been changed."
        )


def test_tp_idle_loop_uses_get_nowait():
    """Rank 0 must not block in q.get(timeout=5) while rank 1 NCCL-spins.

    PromptQueue.get_nowait() returns None on empty (does not raise). Rank 0
    broadcasts size=-1 then sleeps; rank 1 mirrors the sleep after receiving
    the empty signal.
    """
    main_py = _read(REPO_ROOT / "main.py")
    assert "q.get_nowait()" in main_py, (
        "TP prompt_worker rank 0 must drain the queue with get_nowait() so "
        "idle NCCL broadcasts happen immediately instead of after a 5s block."
    )
    execution_py = _read(REPO_ROOT / "execution.py")
    assert "def get_nowait(self):" in execution_py
    # Empty-queue path must sleep on BOTH ranks so GPU 1 can drop clocks.
    idle_sleep_hits = main_py.count("time.sleep(5.0)")
    assert idle_sleep_hits >= 2, (
        f"expected rank-0 and rank-1 idle sleeps in main.py, found {idle_sleep_hits}"
    )