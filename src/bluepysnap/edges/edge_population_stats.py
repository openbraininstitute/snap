"""EdgePopulation stats helper."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from bluepysnap._typing import NodeGroup
from bluepysnap.exceptions import BluepySnapError

if TYPE_CHECKING:
    from bluepysnap.edges.edge_population import EdgePopulation


class StatsHelper:
    """EdgePopulation stats helper."""

    def __init__(self, edge_population: "EdgePopulation") -> None:
        """Initialize StatsHelper with an EdgePopulation instance."""
        self._edge_population = edge_population

    def divergence(
        self,
        source: NodeGroup,
        target: NodeGroup,
        by: Literal["synapses", "connections"],
        sample: int | None = None,
    ) -> npt.NDArray[np.int64]:
        """`source` -> `target` divergence.

        Calculate the divergence based on number of `"connections"` or `"synapses"` each `source`
        cell shares with the cells specified in `target`.

        * `connections`: number of unique target cells each source cell shares a connection with
        * `synapses`: number of unique synapses between a source cell and its target cells

        Args:
            source: source nodes
            target: target nodes
            by: 'synapses' or 'connections'
            sample: if specified, sample size for source group

        Returns:
            Array with synapse / connection count per each cell from `source` sample
            (taking into account only connections to cells in `target`).
        """
        by_alternatives = {"synapses", "connections"}
        if by not in by_alternatives:
            raise BluepySnapError(f"`by` should be one of {by_alternatives}; got: {by}")

        source_sample = self._edge_population.source.ids(source, sample=sample)

        result = {id_: 0 for id_ in source_sample}
        if by == "synapses":
            connections = self._edge_population.iter_connections(
                source_sample, target, return_edge_count=True
            )
            for (_, pre_gid), _, synapse_count in connections:
                result[pre_gid] += synapse_count
        else:
            connections = self._edge_population.iter_connections(source_sample, target)
            for (_, pre_gid), (_, _) in connections:
                result[pre_gid] += 1

        return np.array(list(result.values()))

    def convergence(
        self,
        source: NodeGroup,
        target: NodeGroup,
        by: Literal["synapses", "connections"] | None = None,
        sample: int | None = None,
    ) -> npt.NDArray[np.int64]:
        """`source` -> `target` convergence.

        Calculate the convergence based on number of `"connections"` or `"synapses"` each `target`
        cell shares with the cells specified in `source`.

        * `connections`: number of unique source cells each target cell shares a connection with
        * `synapses`: number of unique synapses between a target cell and its source cells

        Args:
            source: source nodes
            target: target nodes
            by: 'synapses' or 'connections'
            sample: if specified, sample size for target group

        Returns:
            Array with synapse / connection count per each cell from `target` sample
            (taking into account only connections from cells in `source`).
        """
        by_alternatives = {"synapses", "connections"}
        if by not in by_alternatives:
            raise BluepySnapError(f"`by` should be one of {by_alternatives}; got: {by}")

        target_sample = self._edge_population.target.ids(target, sample=sample)

        result = {id_: 0 for id_ in target_sample}
        if by == "synapses":
            connections = self._edge_population.iter_connections(
                source, target_sample, return_edge_count=True
            )
            for (_, _), (_, post_gid), synapse_count in connections:
                result[post_gid] += synapse_count
        else:
            connections = self._edge_population.iter_connections(source, target_sample)
            for (_, _), (_, post_gid) in connections:
                result[post_gid] += 1

        return np.array(list(result.values()))
