"""Consistent machine-readable output for all public workflows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(start: Path) -> dict[str, Any]:
    """Collect Git identity when available; never make output writing depend on Git."""

    try:
        root = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"repository_root": root, "commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        current = start.resolve()
        for candidate in (current, *current.parents):
            git_dir = candidate / ".git"
            if not git_dir.is_dir():
                continue
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                reference = head.removeprefix("ref: ")
                loose_ref = git_dir / reference
                commit = loose_ref.read_text(encoding="utf-8").strip() if loose_ref.is_file() else None
                if commit is None and (git_dir / "packed-refs").is_file():
                    for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                        if line and not line.startswith(("#", "^")):
                            value, name = line.split(" ", 1)
                            if name == reference:
                                commit = value
                                break
            else:
                commit = head
            return {"repository_root": str(candidate), "commit": commit, "dirty": None}
        return {"repository_root": None, "commit": None, "dirty": None}


def runtime_metadata(
    *,
    input_files: list[str | Path] | tuple[str | Path, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build submission-facing environment and input provenance metadata."""

    packages = {}
    for package in ("sirdwell", "mhn", "numpy", "pandas", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    inputs = []
    for raw_path in input_files:
        path = Path(raw_path).resolve()
        inputs.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_state(Path.cwd()),
        "inputs": inputs,
    }
    if extra:
        metadata["workflow"] = extra
    return metadata


class ResultWriter:
    """Write tables and metadata under one result directory.

    Workflows return DataFrames to callers and optionally use this writer.
    Plotting is deliberately outside this class and outside the workflows.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        input_files: list[str | Path] | tuple[str | Path, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []
        self._input_files = tuple(input_files)
        self._metadata = dict(metadata or {})

    def table(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.tables / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        self._files.append(path)
        return path

    def json(self, name: str, value: Any) -> Path:
        path = self.root / f"{name}.json"
        if is_dataclass(value):
            value = asdict(value)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        self._files.append(path)
        return path

    def track(self, path: str | Path) -> Path:
        """Include an already-written file in the result manifest."""

        tracked = Path(path)
        if not tracked.is_file():
            raise FileNotFoundError(tracked)
        self._files.append(tracked)
        return tracked

    def manifest(self) -> Path:
        if not any(path.name == "run_metadata.json" for path in self._files):
            self.json(
                "run_metadata",
                runtime_metadata(input_files=self._input_files, extra=self._metadata),
            )
        rows = []
        for path in sorted(set(self._files)):
            digest = sha256_file(path)
            rows.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        path = self.root / "result_manifest.tsv"
        pd.DataFrame(rows, columns=["path", "size_bytes", "sha256"]).to_csv(path, sep="\t", index=False)
        return path
