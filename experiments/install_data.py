"""Verify and install the bundled longitudinal inputs without changing records.

Run from the repository root with ``python -m experiments.install_data``. Compressed
parts are consumed in manifest order, and existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("difg_glass", "coadread_mskcc", "mnm_washu_2016")
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Part:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class InputFile:
    study_id: str
    filename: str
    compression: str
    sha256: str
    size_bytes: int
    parts: tuple[Part, ...]


def _keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label}: expected fields {', '.join(sorted(expected))}")
    return value


def _simple_filename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", "\x00"))
    ):
        raise ValueError(f"{label}: must be a simple filename")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label}: must be a 64-character SHA-256 digest")
    return value.lower()


def _size(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}: must be a nonnegative integer")
    return value


def load_manifest(repo_root: Path) -> tuple[InputFile, ...]:
    """Validate schema and keep every packaged path within its repository study."""
    repo_root = repo_root.resolve()
    raw = json.loads((repo_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    raw = _keys(raw, {"schema_version", "files"}, "manifest")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("manifest: unsupported schema_version; expected 1")
    if not isinstance(raw["files"], list) or not raw["files"]:
        raise ValueError("manifest: files must be a nonempty list")

    entries = []
    destinations = set()
    packaged_paths = set()
    for index, value in enumerate(raw["files"]):
        label = f"manifest.files[{index}]"
        value = _keys(
            value,
            {"study_id", "filename", "compression", "sha256", "size_bytes", "parts"},
            label,
        )
        study_id = value["study_id"]
        if not isinstance(study_id, str) or study_id not in STUDIES:
            raise ValueError(f"{label}: unsupported study_id")
        filename = _simple_filename(value["filename"], f"{label}.filename")
        destination = (study_id, filename)
        if destination in destinations:
            raise ValueError(f"{label}: duplicate destination {study_id}/{filename}")
        destinations.add(destination)
        compression = value["compression"]
        if compression not in ("none", "gzip"):
            raise ValueError(f"{label}: compression must be none or gzip")
        if not isinstance(value["parts"], list) or not value["parts"]:
            raise ValueError(f"{label}: parts must be a nonempty ordered list")

        parts = []
        prefix = ("data", "raw", "longitudinal", "cbioportal", study_id)
        for part_index, part_value in enumerate(value["parts"]):
            part_label = f"{label}.parts[{part_index}]"
            part_value = _keys(part_value, {"path", "sha256", "size_bytes"}, part_label)
            raw_path = part_value["path"]
            if not isinstance(raw_path, str) or "\\" in raw_path or "\x00" in raw_path:
                raise ValueError(f"{part_label}: invalid repository-relative path")
            relative = PurePosixPath(raw_path)
            # Compare the literal components too, rejecting dot and empty segments.
            components = tuple(raw_path.split("/"))
            if relative.is_absolute() or len(components) != 6 or components[:5] != prefix:
                raise ValueError(f"{part_label}: path must be under data/raw/longitudinal/cbioportal/{study_id}/")
            _simple_filename(components[-1], f"{part_label}.path")
            path = (repo_root / relative).resolve()
            expected_parent = repo_root / Path(*prefix)
            if path.parent != expected_parent:
                raise ValueError(f"{part_label}: path escapes its repository study directory")
            if path in packaged_paths:
                raise ValueError(f"{part_label}: duplicate packaged path")
            packaged_paths.add(path)
            parts.append(
                Part(
                    path=path,
                    sha256=_digest(part_value["sha256"], f"{part_label}.sha256"),
                    size_bytes=_size(part_value["size_bytes"], f"{part_label}.size_bytes"),
                )
            )
        entries.append(
            InputFile(
                study_id=study_id,
                filename=filename,
                compression=compression,
                sha256=_digest(value["sha256"], f"{label}.sha256"),
                size_bytes=_size(value["size_bytes"], f"{label}.size_bytes"),
                parts=tuple(parts),
            )
        )
    return tuple(entries)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _verify_file(path: Path, expected_digest: str, expected_size: int) -> None:
    if not path.is_file():
        raise ValueError(f"Required file is missing or is not a regular file: {path}")
    digest, size = _file_digest(path)
    if size != expected_size or digest != expected_digest:
        raise ValueError(
            f"Integrity check failed: {path}; expected {expected_size} bytes / "
            f"SHA-256 {expected_digest}, found {size} bytes / SHA-256 {digest}"
        )


def _target(data_root: Path, entry: InputFile) -> Path:
    target = data_root / entry.study_id / entry.filename
    if target.parent.resolve() != data_root / entry.study_id:
        raise ValueError(f"Destination study directory escapes data root: {target.parent}")
    return target


def _existing_matches(target: Path, entry: InputFile) -> bool:
    if target.is_symlink():
        raise ValueError(f"Refusing a symbolic-link destination: {target}")
    if not target.exists():
        return False
    try:
        _verify_file(target, entry.sha256, entry.size_bytes)
    except ValueError as exc:
        raise ValueError(f"Existing destination differs; refusing to overwrite {target}: {exc}") from exc
    return True


def _copy(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)


def _restore(entry: InputFile, destination: BinaryIO) -> tuple[str, int]:
    """Restore ordered bytes with bounded memory and check the declared raw size."""
    digest = hashlib.sha256()
    size = 0

    def copy_raw(source: BinaryIO) -> None:
        nonlocal size
        while chunk := source.read(CHUNK_SIZE):
            size += len(chunk)
            if size > entry.size_bytes:
                raise ValueError(f"Restored data exceeds declared size: {entry.study_id}/{entry.filename}")
            digest.update(chunk)
            destination.write(chunk)

    if entry.compression == "none":
        for part in entry.parts:
            with part.path.open("rb") as source:
                copy_raw(source)
    else:
        # Parts are byte ranges of a gzip stream, not independently gzipped files.
        # A temporary concatenation lets gzip verify its trailer and all members.
        with tempfile.TemporaryFile(mode="w+b") as compressed:
            for part in entry.parts:
                with part.path.open("rb") as source:
                    _copy(source, compressed)
            compressed.seek(0)
            with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
                copy_raw(source)
    return digest.hexdigest(), size


def _install(entry: InputFile, target: Path) -> str:
    if _existing_matches(target, entry):
        return "already installed"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".sirdwell-", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            digest, size = _restore(entry, destination)
            if digest != entry.sha256 or size != entry.size_bytes:
                raise ValueError(f"Restored data failed integrity check: {entry.study_id}/{entry.filename}")
            destination.flush()
            os.fsync(destination.fileno())
        try:
            # Atomic no-clobber publication: unlike replace(), link() cannot
            # overwrite a file that another installer created during restoration.
            os.link(temporary, target)
        except FileExistsError:
            if _existing_matches(target, entry):
                return "already installed"
            raise ValueError(f"Destination appeared during installation: {target}")
        return "installed"
    finally:
        temporary.unlink(missing_ok=True)


def install_data(repo_root: Path, data_root: Path, studies: tuple[str, ...], *, verify_only: bool = False) -> None:
    """Verify all selected packaged parts before writing any destination files."""
    entries = tuple(entry for entry in load_manifest(repo_root) if entry.study_id in studies)
    missing_studies = set(studies).difference(entry.study_id for entry in entries)
    if missing_studies:
        raise ValueError(f"Manifest has no files for requested studies: {', '.join(sorted(missing_studies))}")
    data_root = data_root.expanduser().resolve()
    for entry in entries:
        for part in entry.parts:
            _verify_file(part.path, part.sha256, part.size_bytes)
    # Detect existing conflicts before installing any files in this invocation.
    targets = [(entry, _target(data_root, entry)) for entry in entries]
    existing = [_existing_matches(target, entry) for entry, target in targets]
    for (entry, target), present in zip(targets, existing):
        if verify_only:
            status = "verified packaged and installed files" if present else "verified packaged files; destination absent"
        else:
            status = _install(entry, target)
        print(f"{entry.study_id}/{entry.filename}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", choices=STUDIES, help="Select one study; repeatable. Defaults to all studies.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "longitudinal" / "cbioportal", help="Destination root containing study directories.")
    parser.add_argument("--verify-only", action="store_true", help="Check packaged parts and existing destinations without writing files.")
    args = parser.parse_args()
    studies = tuple(dict.fromkeys(args.study or STUDIES))
    try:
        install_data(REPO_ROOT, args.data_root, studies, verify_only=args.verify_only)
    except (OSError, ValueError, EOFError) as exc:
        parser.exit(1, f"Data installation failed: {exc}\n")


if __name__ == "__main__":
    main()
