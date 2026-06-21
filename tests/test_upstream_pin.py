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


def test_pinned_sha_is_ancestor_of_master_branch():
    """The pinned SHA must be an ancestor of the local `master` branch we ship from.

    We pin origin/master tip (not a tag), so master fast-forwards to the pin and
    the pin must remain reachable from master. If someone bumps origin/master
    without updating .upstream-pin, this test still passes (old tip is still an
    ancestor) -- test_pinned_sha_is_in_master_history covers staleness vs origin.
    This test catches the opposite: the pin somehow drifted ahead of master
    (e.g. a partial rebase left master behind the pin).
    """
    pin = _parse_pin_file()
    sha = pin["sha"]
    # merge-base --is-ancestor exits 0 if sha is an ancestor of master
    rc = subprocess.call(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", sha, "master"]
    )
    assert rc == 0, (
        f"Pinned SHA {sha} is not an ancestor of local master. "
        "master must fast-forward to at least the pinned SHA."
    )