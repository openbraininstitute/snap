"""Type aliases used in the type annotations.

This module is private: the aliases may change while the annotations are being added.
"""

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from bluepysnap.circuit_ids import CircuitEdgeIds, CircuitNodeIds
from bluepysnap.circuit_ids_types import CircuitEdgeId, CircuitNodeId

#: A single node: a node id, or a node id with its population.
NodeId: TypeAlias = int | np.integer | CircuitNodeId

#: A ``group`` that selects any number of nodes: :data:`NodeGroup` without the single node ids.
#: Used in the overloads, to separate the single node case, which returns different types.
MultiNodeGroup: TypeAlias = (
    CircuitNodeIds
    | Sequence[int]
    | Sequence[CircuitNodeId]
    | npt.NDArray[np.integer]
    | str
    | Mapping[str, Any]
    | None
)

#: The ``group`` argument used to select nodes, see the Group Concept in
#: :mod:`bluepysnap.nodes.node_population`.
NodeGroup: TypeAlias = NodeId | MultiNodeGroup

#: A single edge: an edge id, or an edge id with its population.
EdgeId: TypeAlias = int | np.integer | CircuitEdgeId

#: The argument used to select edges, see :meth:`bluepysnap.edges.EdgePopulation.ids`.
EdgeGroup: TypeAlias = (
    EdgeId
    | CircuitEdgeIds
    | Sequence[int]
    | Sequence[CircuitEdgeId]
    | npt.NDArray[np.integer]
    | Mapping[str, Any]
    | None
)

#: A list of property names. A single property name is a plain ``str``.
#: Tuples are not included because pandas interprets a tuple as a single column label.
PropertyList: TypeAlias = list[str] | npt.NDArray[np.str_]

#: Array of node or edge ids, always of dtype ``IDS_DTYPE``.
IdsArray: TypeAlias = npt.NDArray[np.int64]
