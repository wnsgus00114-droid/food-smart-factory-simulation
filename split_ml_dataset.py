#!/usr/bin/env python3
"""Create a deterministic, plant-profile-level split for an HTST ML dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from ml_pipeline_common import (
    BASE_DIR,
    load_contract,
    read_csv_rows,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_csv_header,
    write_json,
)


SPLITTER_VERSION = "2.2.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    return parser


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _counts(size: int, train_ratio: float, validation_ratio: float) -> tuple[int, int]:
    if size <= 0:
        return 0, 0
    if size == 1:
        return 1, 0
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
    profile_domains: Mapping[str, str], seed: int, train_ratio: float, validation_ratio: float
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    id_profiles = sorted(
        (profile for profile, domain in profile_domains.items() if domain.upper() == "ID"),
        key=lambda profile: (_rank(seed, profile), profile),
    )
    train_count, validation_count = _counts(
        len(id_profiles), train_ratio, validation_ratio
    )
    for index, profile in enumerate(id_profiles):
        if index < train_count:
            split = "train_id"
        elif index < train_count + validation_count:
            split = "validation_id"
        else:
            split = "test_id"
        assignments[profile] = split
    for profile, domain in profile_domains.items():
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
    checksum_failures = verify_checksums(args.dataset)
    if checksum_failures:
        raise ValueError("Dataset checksum verification failed: " + "; ".join(checksum_failures))
    dataset_manifest_path = args.dataset / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    contract = load_contract()
    if not (
        dataset_manifest.get("contract_version") == contract["contract_version"] == "2.2.0"
        and dataset_manifest.get("model_version") == contract["model_version"] == "2.2.0"
        and str(dataset_manifest.get("generator_version", "")).startswith("2.")
    ):
        raise ValueError("Splitter requires a model/contract/generator v2 dataset")
    output = args.output or args.dataset / "splits"
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use a new directory."
            )
    else:
        output.mkdir(parents=True)

    episodes = list(read_csv_rows(args.dataset / "episodes.csv"))
    if not episodes:
        raise ValueError("episodes.csv is empty")
    episode_ids = [row["episode_id"] for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episodes.csv contains duplicate episode_id values")
    profile_domains: dict[str, str] = {}
    for row in episodes:
        profile = row["plant_profile_id"]
        domain = row.get("domain", "ID")
        previous = profile_domains.setdefault(profile, domain)
        if previous != domain:
            raise ValueError(f"Profile {profile} has multiple domains")
    assignments = _assign_profiles(
        profile_domains, args.seed, args.train_ratio, args.validation_ratio
    )

    fields = [
        "episode_id",
        "plant_profile_id",
        "counterfactual_group_id",
        "profile_config_hash",
        "noise_seed",
        "sampling_seed",
        "canonical_code",
        "domain",
        "split",
    ]
    split_path = output / "episode_splits.csv"
    episode_counts: Counter[str] = Counter()
    profile_sets: dict[str, set[str]] = defaultdict(set)
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = write_csv_header(handle, fields)
        for row in sorted(episodes, key=lambda item: item["episode_id"]):
            split = assignments[row["plant_profile_id"]]
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "split": split,
                }
            )
            episode_counts[split] += 1
            profile_sets[split].add(row["plant_profile_id"])

    manifest = {
        "splitter_version": SPLITTER_VERSION,
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "seed": args.seed,
        "provenance": {
            "split_ml_dataset.py": sha256_file(Path(__file__).resolve()),
            "ml_pipeline_common.py": sha256_file(BASE_DIR / "ml_pipeline_common.py"),
        },
        "policy": {
            "unit": "plant_profile_id",
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
            "ood_policy": "all non-ID profiles are held out as test_ood_profile",
            "counterfactual_policy": "all episodes in a counterfactual group stay together",
        },
        "counts": {
            split: {
                "profiles": len(profile_sets[split]),
                "episodes": episode_counts[split],
            }
            for split in sorted(episode_counts)
        },
        "artifacts": ["episode_splits.csv", "split_manifest.json", "checksums.sha256"],
    }
    write_json(output / "split_manifest.json", manifest)
    write_checksums(output, ["episode_splits.csv", "split_manifest.json"])
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = split_dataset(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Wrote deterministic profile split to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
