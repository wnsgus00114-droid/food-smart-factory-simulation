"""FlowTwin-Guard research model for the HTST reference simulator.

The package deliberately keeps its top-level import free of optional deep-
learning dependencies.  Import :mod:`flowtwin_guard.model` only in the ML
environment described by ``requirements-ml.txt``.
"""

from .graph import (
    FEATURE_NODE_MAP,
    FEATURE_RELATION_MAP,
    ProcessGraph,
    build_process_graph,
)

__all__ = [
    "FEATURE_NODE_MAP",
    "FEATURE_RELATION_MAP",
    "ProcessGraph",
    "build_process_graph",
]

__version__ = "0.1.0"
