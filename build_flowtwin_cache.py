#!/usr/bin/env python3
"""Build a checksummed per-episode cache for FlowTwin training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "The FlowTwin cache requires optional ML dependencies; install requirements-ml.txt"
    ) from exc

from flowtwin_guard.cache import build_episode_cache
from flowtwin_guard.data import load_target_contract
from flowtwin_guard.graph import build_process_graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-set", default="S3-context")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        graph = build_process_graph(feature_set=args.feature_set)
        targets = load_target_contract(graph=graph)
        output = build_episode_cache(
            args.dataset, args.splits, args.output, graph, targets
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Wrote FlowTwin episode cache to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
