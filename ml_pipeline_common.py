#!/usr/bin/env python3
"""Shared, dependency-free utilities for the HTST ML data pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = BASE_DIR / "ml_contract.json"
SEMANTIC_VERSION = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")


def load_contract(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the versioned ML contract."""
    contract_path = path or CONTRACT_PATH
    with contract_path.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    required = {
        "contract_version",
        "model_version",
        "taxonomy",
        "signal_context_fields",
        "feature_sets",
        "always_forbidden_features",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"ML contract is missing keys: {missing}")
    for version_field in ("contract_version", "model_version"):
        value = str(contract[version_field])
        if not SEMANTIC_VERSION.fullmatch(value):
            raise ValueError(f"{version_field} must be a semantic version, got {value!r}")
    codes = [item["code"] for item in contract["taxonomy"]]
    if len(codes) != len(set(codes)):
        raise ValueError("Canonical taxonomy codes must be unique")
    canonical_names = [item["canonical_scenario"] for item in contract["taxonomy"]]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("Canonical taxonomy scenario names must be unique")
    behavior_owners: dict[str, str] = {}
    for item in contract["taxonomy"]:
        required_item_fields = {
            "code",
            "canonical_scenario",
            "simulator_scenarios",
            "implemented",
            "is_fault",
            "kind",
        }
        missing_item = sorted(required_item_fields - item.keys())
        if missing_item:
            raise ValueError(
                f"Taxonomy item {item.get('code', '<unknown>')} is missing {missing_item}"
            )
        scenarios = list(item["simulator_scenarios"])
        if item["implemented"] and not scenarios:
            raise ValueError(f"Implemented taxonomy item {item['code']} has no simulator path")
        if item["implemented"] and item["canonical_scenario"] not in scenarios:
            raise ValueError(
                f"Implemented taxonomy item {item['code']} must use its canonical scenario"
            )
        aliases = set(item.get("aliases", []))
        duplicate_spellings = aliases & set(scenarios)
        if duplicate_spellings:
            raise ValueError(
                f"Taxonomy item {item['code']} lists aliases as distinct behaviors: "
                f"{sorted(duplicate_spellings)}"
            )
        for scenario in scenarios:
            previous = behavior_owners.setdefault(scenario, item["code"])
            if previous != item["code"]:
                raise ValueError(
                    f"Simulator behavior {scenario!r} belongs to both {previous} and {item['code']}"
                )
    feature_set_names = list(contract["feature_sets"])
    if feature_set_names != ["S0-core", "S1-pressure", "S2-derived", "S3-context"]:
        raise ValueError("Feature sets must preserve the ordered S0/S1/S2/S3 v2 contract")
    previous_fields: set[str] = set()
    for name in feature_set_names:
        current = set(feature_fields(contract, name))
        if not previous_fields <= current:
            raise ValueError(f"Feature set {name} is not a superset of the preceding set")
        previous_fields = current
    return contract


def stable_seed(master_seed: int, *parts: object) -> int:
    """Return a stable 32-bit seed independent of Python hash randomization."""
    payload = "|".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_checksums(directory: Path, relative_paths: Iterable[str]) -> Path:
    checksum_path = directory / "checksums.sha256"
    lines = [
        f"{sha256_file(directory / relative_path)}  {relative_path}"
        for relative_path in sorted(set(relative_paths))
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def read_checksums(directory: Path) -> dict[str, str]:
    checksum_path = directory / "checksums.sha256"
    checksums: dict[str, str] = {}
    if not checksum_path.exists():
        return checksums
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed checksum line {line_number} in {checksum_path}"
            ) from exc
        checksums[relative_path] = digest
    return checksums


def verify_checksums(directory: Path) -> list[str]:
    failures: list[str] = []
    checksums = read_checksums(directory)
    if not checksums:
        return ["checksums.sha256 is missing or empty"]
    for relative_path, expected in checksums.items():
        path = directory / relative_path
        if not path.is_file():
            failures.append(f"missing checksummed file: {relative_path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )
    return failures


def write_csv_header(handle: Any, fieldnames: Sequence[str]) -> csv.DictWriter:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return writer


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def taxonomy_indexes(
    contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return lookup maps by code and by any accepted scenario spelling."""
    by_code: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for raw_item in contract["taxonomy"]:
        item = dict(raw_item)
        by_code[item["code"].upper()] = item
        names = {
            item["canonical_scenario"],
            *item.get("simulator_scenarios", []),
            *item.get("aliases", []),
        }
        for name in names:
            if name in by_name and by_name[name]["code"] != item["code"]:
                raise ValueError(f"Taxonomy name {name!r} maps to multiple codes")
            by_name[name] = item
    return by_code, by_name


def select_taxonomy(
    contract: Mapping[str, Any], requested: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Resolve codes, canonical names and aliases, then canonicalize duplicates."""
    by_code, by_name = taxonomy_indexes(contract)
    if requested:
        raw_tokens: list[str] = []
        for value in requested:
            raw_tokens.extend(token.strip() for token in value.split(",") if token.strip())
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for token in raw_tokens:
            item = by_code.get(token.upper()) or by_name.get(token)
            if item is None:
                choices = sorted(set(by_name) | set(by_code))
                raise ValueError(f"Unknown scenario {token!r}; choose from {choices}")
            if not item["implemented"]:
                raise ValueError(
                    f"{item['code']} ({item['canonical_scenario']}) is explicitly unimplemented: "
                    f"{item.get('reason', 'no simulator path')}"
                )
            if item["code"] not in seen:
                selected.append(item)
                seen.add(item["code"])
        return selected
    return [dict(item) for item in contract["taxonomy"] if item["implemented"]]


def feature_fields(contract: Mapping[str, Any], feature_set: str = "S3-context") -> list[str]:
    if feature_set not in contract["feature_sets"]:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(contract['feature_sets'])}"
        )
    fields = list(contract["feature_sets"][feature_set])
    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise ValueError(f"Feature set {feature_set} contains duplicates: {duplicates}")
    forbidden = set(contract["always_forbidden_features"])
    overlap = sorted(forbidden & set(fields))
    if overlap:
        raise ValueError(f"Feature set {feature_set} contains forbidden inputs: {overlap}")
    return fields


def field_type(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolation percentile without third-party dependencies."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
