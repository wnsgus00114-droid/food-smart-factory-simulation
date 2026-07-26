#!/usr/bin/env python3
"""Shared provenance and safe-publication helpers for HTST command-line tools."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping, Sequence


class ProvenanceError(ValueError):
    """Raised when an artifact provenance contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(files: Mapping[str, Path]) -> dict[str, object]:
    """Hash named source files and return a path-independent aggregate digest."""
    if not files:
        raise ValueError("source provenance requires at least one file")
    digests: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name, path in sorted(files.items()):
        if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError(f"unsafe source provenance name: {name!r}")
        digest = sha256_file(path)
        digests[name] = digest
        encoded_name = name.encode("utf-8")
        aggregate.update(len(encoded_name).to_bytes(8, "big"))
        aggregate.update(encoded_name)
        aggregate.update(bytes.fromhex(digest))
    return {
        "algorithm": "sha256",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": digests,
    }


def _safe_relative_path(output_dir: Path, path: Path) -> Path:
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ProvenanceError(f"artifact is outside output directory: {path}") from exc
    if relative == Path(".") or ".." in relative.parts:
        raise ProvenanceError(f"unsafe artifact path: {path}")
    return relative


def write_checksums(
    output_dir: Path,
    files: Sequence[Path],
    *,
    filename: str = "checksums.sha256",
) -> Path:
    """Write deterministic SHA-256 records for a duplicate-free artifact set."""
    indexed: dict[str, Path] = {}
    for path in files:
        relative = _safe_relative_path(output_dir, path).as_posix()
        if relative in indexed:
            raise ProvenanceError(f"duplicate artifact in checksum set: {relative}")
        if not path.is_file():
            raise ProvenanceError(f"artifact does not exist: {path}")
        indexed[relative] = path
    checksum_path = output_dir / filename
    lines = [
        f"{sha256_file(indexed[relative])}  {relative}"
        for relative in sorted(indexed)
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def read_checksums(checksum_path: Path) -> dict[str, str]:
    """Parse a checksum file while rejecting duplicate and escaping paths."""
    if not checksum_path.is_file():
        raise ProvenanceError(f"checksum file not found: {checksum_path}")
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative = line.partition("  ")
        candidate = PurePosixPath(relative)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
        ):
            raise ProvenanceError(
                f"invalid checksum record at {checksum_path}:{line_number}"
            )
        if relative in records:
            raise ProvenanceError(f"duplicate checksum path: {relative}")
        records[relative] = digest
    if not records:
        raise ProvenanceError(f"checksum file is empty: {checksum_path}")
    return records


def verify_checksums(output_dir: Path, checksum_path: Path) -> dict[str, str]:
    """Verify all listed files and return the validated relative-path mapping."""
    records = read_checksums(checksum_path)
    root = output_dir.resolve()
    for relative, expected in records.items():
        path = (root / Path(PurePosixPath(relative))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(f"checksum path escapes output directory: {relative}") from exc
        if not path.is_file():
            raise ProvenanceError(f"checksummed artifact not found: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ProvenanceError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
    return records


def _unused_sibling(parent: Path, prefix: str) -> Path:
    placeholder = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    placeholder.rmdir()
    return placeholder


def publish_directory(staging: Path, target: Path) -> None:
    """Publish a complete staging directory, rolling back an existing target on error."""
    staging = staging.resolve()
    target = target.resolve()
    if not staging.is_dir():
        raise ValueError(f"staging directory does not exist: {staging}")
    if staging.parent != target.parent:
        raise ValueError("staging and target must share a filesystem parent")
    if target.exists() and not target.is_dir():
        raise ValueError(f"output target exists and is not a directory: {target}")
    if not target.exists():
        os.replace(staging, target)
        return

    backup = _unused_sibling(target.parent, f".{target.name}.backup-")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        os.replace(backup, target)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        warnings.warn(
            f"published {target}, but old output remains at {backup}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


@contextmanager
def staged_output_directory(target: Path) -> Iterator[Path]:
    """Yield an empty sibling directory and publish it only after successful exit."""
    resolved_target = target.expanduser().resolve()
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    if resolved_target.exists() and not resolved_target.is_dir():
        raise ValueError(
            f"output target exists and is not a directory: {resolved_target}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_target.name}.staging-",
            dir=resolved_target.parent,
        )
    )
    try:
        yield staging
        publish_directory(staging, resolved_target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
