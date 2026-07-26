#!/usr/bin/env python3
"""Create deterministic profile-level splits for an isolated D3-RUL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .d3_rul_common import (
        BASE_DIR,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        read_csv_rows,
        sha256_file,
        verify_checksums,
        write_checksums,
        write_csv_header,
        write_json,
    )
except ImportError:  # pragma: no cover
    from d3_rul_common import (
        BASE_DIR,
        load_d3_contract,
        model_version_is_compatible,
        prepare_empty_output,
        read_csv_rows,
        sha256_file,
        verify_checksums,
        write_checksums,
        write_csv_header,
        write_json,
    )


SPLITTER_VERSION = "1.0.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    return parser


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _split_counts(size: int, train_ratio: float, validation_ratio: float) -> tuple[int, int]:
    if size <= 1:
        return size, 0
    if size == 2:
        return 1, 0
    train = max(1, int(size * train_ratio))
    validation = max(1, int(size * validation_ratio))
    while train + validation >= size:
        if train > validation and train > 1:
            train -= 1
        elif validation > 1:
            validation -= 1
        else:
            break
    return train, validation


def _assign_profiles(
    domains: Mapping[str, str], seed: int, train_ratio: float, validation_ratio: float
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    identity = sorted(
        (profile for profile, domain in domains.items() if domain.upper() == "ID"),
        key=lambda profile: (_rank(seed, profile), profile),
    )
    train_count, validation_count = _split_counts(
        len(identity), train_ratio, validation_ratio
    )
    for index, profile in enumerate(identity):
        assignments[profile] = (
            "train_id"
            if index < train_count
            else "validation_id"
            if index < train_count + validation_count
            else "test_id"
        )
    for profile, domain in domains.items():
        if domain.upper() != "ID":
            assignments[profile] = "test_ood_profile"
    return assignments


def split_dataset(args: argparse.Namespace) -> Path:
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train-ratio must be in (0, 1)")
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be in [0, 1)")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("train-ratio + validation-ratio must be below 1")
    failures = verify_checksums(args.dataset)
    if failures:
        raise ValueError("Dataset checksum verification failed: " + "; ".join(failures))
    manifest_path = args.dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = load_d3_contract()
    if not (
        manifest.get("dataset_family") == "D3-RUL"
        and manifest.get("contract_version") == contract["contract_version"]
        and model_version_is_compatible(str(manifest.get("model_version", "")), contract)
        and str(manifest.get("generator_version", "")).startswith("1.")
    ):
        raise ValueError("Splitter requires a compatible D3-RUL dataset")
    profiles = list(read_csv_rows(args.dataset / "profiles.csv"))
    trajectories = list(read_csv_rows(args.dataset / "trajectories.csv"))
    if not profiles or not trajectories:
        raise ValueError("profiles.csv and trajectories.csv must be non-empty")
    domains = {row["plant_profile_id"]: row["domain"] for row in profiles}
    if len(domains) != len(profiles):
        raise ValueError("profiles.csv contains duplicate profile IDs")
    assignments = _assign_profiles(
        domains, args.seed, args.train_ratio, args.validation_ratio
    )
    prepare_empty_output(args.output)
    fields = [
        "trajectory_id",
        "life_family_id",
        "plant_profile_id",
        "profile_config_hash",
        "domain",
        "noise_seed",
        "degradation_seed",
        "censor_seed",
        "split",
    ]
    counts: Counter[str] = Counter()
    profile_sets: dict[str, set[str]] = defaultdict(set)
    path = args.output / "trajectory_splits.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = write_csv_header(handle, fields)
        for row in sorted(trajectories, key=lambda value: value["trajectory_id"]):
            profile = row["plant_profile_id"]
            if profile not in assignments or row["domain"] != domains[profile]:
                raise ValueError(f"Trajectory has invalid profile/domain: {row['trajectory_id']}")
            split = assignments[profile]
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field != "split"},
                    "split": split,
                }
            )
            counts[split] += 1
            profile_sets[split].add(profile)
    split_manifest = {
        "dataset_family": "D3-RUL",
        "splitter_version": SPLITTER_VERSION,
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "seed": args.seed,
        "policy": {
            "unit": "plant_profile_id",
            "outcomes_used_for_assignment": False,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
            "ood_policy": "Every non-ID profile is held out as test_ood_profile.",
            "life_family_policy": "Every trajectory/life family remains with its plant profile.",
        },
        "counts": {
            split: {
                "profiles": len(profile_sets[split]),
                "trajectories": counts[split],
            }
            for split in sorted(counts)
        },
        "provenance": {
            "split_d3_rul_dataset.py": sha256_file(Path(__file__).resolve()),
            "d3_rul_common.py": sha256_file(BASE_DIR / "d3_rul_common.py"),
        },
        "artifacts": [
            "trajectory_splits.csv",
            "split_manifest.json",
            "checksums.sha256",
        ],
    }
    write_json(args.output / "split_manifest.json", split_manifest)
    write_checksums(
        args.output, ["trajectory_splits.csv", "split_manifest.json"]
    )
    return args.output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = split_dataset(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Wrote D3-RUL profile split: {(output / 'split_manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
