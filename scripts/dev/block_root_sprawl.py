from __future__ import annotations

import subprocess
import sys


ALLOWLIST = {
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".pre-commit-config.yaml",
    "LICENSE",
    "environment.yml",
    "requirements.txt",
}


def _run_git(args: list[str]) -> str:
    try:
        cp = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("Error: git not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        out = (exc.stdout or "").strip()
        err = (exc.stderr or "").strip()
        msg = f"Error: git {' '.join(args)} failed (exit {exc.returncode})."
        if out:
            msg += f"\n--- stdout ---\n{out}"
        if err:
            msg += f"\n--- stderr ---\n{err}"
        raise SystemExit(msg) from exc
    return (cp.stdout or "").strip()


def main() -> int:
    staged = _run_git(["diff", "--cached", "--name-only"]).splitlines()

    offenders: list[str] = []
    for path in staged:
        p = path.strip()
        if not p:
            continue

        is_root_level = ("/" not in p) and ("\\" not in p)
        if is_root_level and p not in ALLOWLIST:
            offenders.append(p)

    if offenders:
        offenders_sorted = "\n".join(f"- {p}" for p in sorted(set(offenders)))
        allow_sorted = "\n".join(f"- {p}" for p in sorted(ALLOWLIST))
        print(
            "Root-level file sprawl blocked.\n\n"
            "These staged root-level files are not allowed:\n"
            f"{offenders_sorted}\n\n"
            "Allowed root-level files:\n"
            f"{allow_sorted}\n",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
