"""Developer workflow helpers.

CLI:
  python -m tools.devtools start-day
  python -m tools.devtools end-day
  python -m tools.devtools status
  python -m tools.devtools guard-push
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from typing import Iterable, Optional, Sequence, Tuple


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            check=check,
            capture_output=capture_output,
            text=text,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        msg = f"Error: command failed ({' '.join(args)}), exit={exc.returncode}"
        if stdout:
            msg += f"\n--- stdout ---\n{stdout}"
        if stderr:
            msg += f"\n--- stderr ---\n{stderr}"
        raise SystemExit(msg) from exc


def _git(*git_args: str, check: bool = True) -> str:
    cp = _run(["git", *git_args], check=check)
    return (cp.stdout or "").rstrip("\n")


def _ensure_git_repo() -> None:
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.strip().lower() != "true":
        raise SystemExit("Error: not inside a git repository.")


def _today_branch_name() -> str:
    d = _dt.date.today().isoformat()
    return f"wip/{d}-work"


def _current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD").strip()


def _is_dirty() -> bool:
    return bool(_git("status", "--porcelain").strip())


def _changed_files_count() -> int:
    out = _git("status", "--porcelain").splitlines()
    return len([ln for ln in out if ln.strip()])


def _ahead_behind() -> Tuple[Optional[int], Optional[int]]:
    # Returns (ahead, behind) relative to upstream; None/None if no upstream.
    try:
        _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    except SystemExit:
        return (None, None)

    counts = _git("rev-list", "--left-right", "--count", "@{u}...HEAD").strip()
    # Format: "<behind>\t<ahead>"
    parts = counts.split()
    if len(parts) != 2:
        return (None, None)
    behind = int(parts[0])
    ahead = int(parts[1])
    return (ahead, behind)


def _print_status_summary(*, include_diffstat_if_dirty: bool) -> None:
    branch = _current_branch()
    dirty = _is_dirty()
    changed = _changed_files_count()
    ahead, behind = _ahead_behind()

    print(f"Branch: {branch}")
    ab = "n/a" if (ahead is None or behind is None) else f"ahead {ahead}, behind {behind}"
    print(f"Working tree: {'DIRTY' if dirty else 'clean'} ({changed} changed paths)")
    print(f"Upstream: {ab}")
    print(_git("status", "-sb"))

    if include_diffstat_if_dirty and dirty:
        diffstat = _git("diff", "--stat").strip()
        if diffstat:
            print("\nDiffstat:")
            print(diffstat)


def _checkout_or_create(branch: str) -> None:
    # If branch exists locally: checkout. Else create and checkout.
    exists = _run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"] , check=False)
    if exists.returncode == 0:
        _git("checkout", branch)
    else:
        _git("checkout", "-b", branch)


def cmd_start_day(_: argparse.Namespace) -> None:
    _ensure_git_repo()
    branch = _today_branch_name()
    _checkout_or_create(branch)
    _print_status_summary(include_diffstat_if_dirty=True)


def cmd_status(_: argparse.Namespace) -> None:
    _ensure_git_repo()
    _print_status_summary(include_diffstat_if_dirty=True)


def _prompt_non_empty(prompt: str) -> str:
    while True:
        try:
            msg = input(prompt).strip()
        except EOFError:
            msg = ""
        if msg:
            return msg
        print("Commit message cannot be empty.")


def cmd_end_day(_: argparse.Namespace) -> None:
    _ensure_git_repo()

    print(_git("status", "-sb"))
    stat = _git("diff", "--stat").strip()
    if stat:
        print("\nDiffstat:")
        print(stat)

    if not _is_dirty():
        print("\nNothing to commit.")
        return

    msg = _prompt_non_empty("Checkpoint commit message: ")
    _git("add", "-A")
    _git("commit", "-m", msg)


def cmd_guard_push(_: argparse.Namespace) -> None:
    _ensure_git_repo()
    branch = _current_branch()
    if branch == "main":
        raise SystemExit("Error: pushing from 'main' is blocked by policy. Create/use a wip/* branch.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.devtools")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start-day", help="Create/checkout daily wip branch and print status")
    s.set_defaults(func=cmd_start_day)

    st = sub.add_parser("status", help="Print a short repo status summary")
    st.set_defaults(func=cmd_status)

    e = sub.add_parser("end-day", help="Print status and optionally create a checkpoint commit")
    e.set_defaults(func=cmd_end_day)

    g = sub.add_parser("guard-push", help="Block pushing from main")
    g.set_defaults(func=cmd_guard_push)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)
    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
