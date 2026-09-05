"""The prime-rl pin, read from `pyproject.toml`.

Single source of truth for the commit. Every phase 3 artifact writer stamps its
provenance block from `provenance()`, so the SHA is written down exactly once
in the repo and never retyped into a JSON file by hand -- the same discipline
`tolerance/` already applies to measured thresholds.

The pin lives in a `[tool.weight_sync_bench.phase3]` table rather than in
`[project] dependencies` because prime-rl is inert data here, not a dependency:
no resolver reads a `[tool.*]` table. `pyproject.toml`'s own comment above the
table carries the reason.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"

TABLE_PATH = ("tool", "weight_sync_bench", "phase3")
_REQUIRED = ("prime_rl_commit", "prime_rl_tag", "prime_rl_pinned_date", "vllm_version")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class PinError(RuntimeError):
    """The pin table is missing, incomplete, or malformed."""


@dataclass(frozen=True)
class Pin:
    """The pinned prime-rl revision and the vLLM version its lock resolves.

    `vllm_version` is not decoration: it is the version phase 2's bf16 floor
    and Qwen3 layout table were both measured at, and it is why this particular
    prime-rl commit was chosen over a release tag. A pin whose lock resolved a
    different vLLM would invalidate those measurements rather than reuse them.
    """

    prime_rl_commit: str
    prime_rl_tag: str
    prime_rl_pinned_date: str
    vllm_version: str

    @property
    def short_commit(self) -> str:
        return self.prime_rl_commit[:12]

    @property
    def tarball_url(self) -> str:
        """Where the source at this pin can be read without cloning."""
        return (
            "https://api.github.com/repos/PrimeIntellect-ai/prime-rl/tarball/"
            f"{self.prime_rl_commit}"
        )


def _read_table(pyproject: Path) -> dict[str, Any]:
    if not pyproject.is_file():
        # Reachable when the package is imported from an installed wheel, which
        # carries no pyproject.toml. Raise rather than fall back to a literal:
        # a hardcoded default here would be a second source of truth, and it
        # would go stale silently, which is the whole failure mode this module
        # exists to prevent.
        raise PinError(
            f"{pyproject} not found; the pin is readable only from a source checkout"
        )
    data = tomllib.loads(pyproject.read_text())
    table: Any = data
    for key in TABLE_PATH:
        if not isinstance(table, dict) or key not in table:
            dotted = ".".join(TABLE_PATH)
            raise PinError(f"[{dotted}] missing from {pyproject}")
        table = table[key]
    if not isinstance(table, dict):
        raise PinError(f"[{'.'.join(TABLE_PATH)}] is not a table in {pyproject}")
    return table


def pin(pyproject: Path = PYPROJECT) -> Pin:
    """Returns the pinned revision, validating it on the way out.

    Validation is deliberate. A truncated or mistyped SHA still looks like a
    plausible pin in a diff and would be stamped into every artifact this phase
    writes; catching it at read time makes it a loud failure in one place
    instead of a quiet wrong string in many.
    """
    table = _read_table(pyproject)
    missing = [k for k in _REQUIRED if k not in table]
    if missing:
        raise PinError(
            f"[{'.'.join(TABLE_PATH)}] in {pyproject} is missing: {', '.join(missing)}"
        )
    unknown = sorted(set(table) - set(_REQUIRED))
    if unknown:
        raise PinError(
            f"[{'.'.join(TABLE_PATH)}] in {pyproject} has unrecognized keys: "
            f"{', '.join(unknown)}. Add them to _REQUIRED and to Pin, or remove them."
        )
    for key in _REQUIRED:
        if not isinstance(table[key], str) or not table[key]:
            raise PinError(f"{key} must be a non-empty string, got {table[key]!r}")
    commit = table["prime_rl_commit"]
    if not _FULL_SHA.match(commit):
        raise PinError(
            f"prime_rl_commit must be a full 40-character lowercase hex SHA, got {commit!r}"
        )
    return Pin(**{k: table[k] for k in _REQUIRED})


def provenance(pyproject: Path = PYPROJECT) -> dict[str, Any]:
    """The pin as an artifact provenance block.

    `source` names how the pinned tree is read, so a reader of a phase 3
    artifact can reproduce the source read without first finding this module.
    """
    p = pin(pyproject)
    block = asdict(p)
    block["source"] = p.tarball_url
    return block


if __name__ == "__main__":  # pragma: no cover - convenience for a quick check
    print(pin())
