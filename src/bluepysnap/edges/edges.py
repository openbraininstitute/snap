# Copyright (c) 2019, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Edges access."""

from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd

from bluepysnap._doctools import AbstractDocSubstitutionMeta
from bluepysnap._typing import (
    Connection,
    ConnectionWithCount,
    ConnectionWithEdgeIds,
    EdgeGroup,
    NodeGroup,
)
from bluepysnap.circuit_ids import CircuitEdgeIds, CircuitNodeIds
from bluepysnap.edges.edge_population import EdgePopulation
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.network import NetworkObject

if TYPE_CHECKING:
    from bluepysnap.circuit import Circuit


class Edges(
    NetworkObject[EdgePopulation],
    metaclass=AbstractDocSubstitutionMeta,
    source_word="NetworkObject",
    target_word="Edge",
):
    """The top level Edges accessor."""

    _population_class = EdgePopulation

    def __init__(self, circuit: "Circuit") -> None:  # pylint: disable=useless-super-delegation
        """Initialize the top level Edges accessor."""
        super().__init__(circuit)

    @property
    def population_names(self) -> list[str]:
        """Defines all sorted edge population names from the Circuit."""
        return sorted(self._circuit.to_libsonata.edge_populations)

    def ids(
        self, group: EdgeGroup = None, sample: int | None = None, limit: int | None = None
    ) -> CircuitEdgeIds:
        """Edge CircuitEdgeIds corresponding to edges ``edge_ids``.

        Args:
            group: Which IDs will be
                returned depends on the type of the ``group`` argument:

                - ``None``: return all CircuitEdgeIds.
                - ``CircuitEdgeId``: return the ID in a CircuitEdgeIds object.
                - ``CircuitEdgeIds``: return the IDs in a CircuitNodeIds object.
                - ``int``: returns a CircuitEdgeIds object containing the corresponding edge ID
                  for all populations.
                - ``sequence``: returns a CircuitEdgeIds object containing the corresponding edge
                  IDs for all populations.
            sample: If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of all the EdgePopulations then all ids are taken and shuffled.
            limit: If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of all the populations
                all node IDs are returned.

        Returns:
            returns a CircuitEdgeIds containing all the edge IDs and the
            corresponding populations. For performance reasons we do not test if the edge ids
            are present or not in the circuit.

        Notes:
            This envision also the maybe future selection of edges on queries.
        """
        if isinstance(group, CircuitEdgeIds):
            diff = np.setdiff1d(group.get_populations(unique=True), self.population_names)
            if diff.size != 0:
                raise BluepySnapError(f"Population {diff} does not exist in the circuit.")
        fun = lambda x: (x.ids(group), x.name)
        return self._get_ids_from_pop(fun, CircuitEdgeIds, sample=sample, limit=limit)

    @overload
    def get(self, edge_ids: EdgeGroup = None, properties: None = None) -> CircuitEdgeIds: ...
    @overload
    def get(
        self, edge_ids: EdgeGroup, properties: str | Collection[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    @overload
    def get(
        self, edge_ids: EdgeGroup = None, *, properties: str | Collection[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    # the parameter is renamed on purpose, to document that only edge ids are accepted,
    # and the return type depends on properties, so the override is reported as incompatible.
    # A bare "type: ignore" is used because it's honored by both ty and pyright.
    def get(  # type: ignore
        self, edge_ids: EdgeGroup = None, properties: str | Collection[str] | None = None
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Edge properties by iterating populations.

        Args:
            edge_ids: same as Edges.ids().
            properties: an edge property name or a list of edge property names.
                If set to None ids are returned.

        Returns:
            - CircuitEdgeIds if ``properties`` is None.
            - Otherwise a generator, that yields tuples of
              ``(<population_name>, pandas.DataFrame)``, where the DataFrame is indexed by
              CircuitEdgeIds and contains the properties from ``properties``.

        Notes:
            The Edges.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        if edge_ids is None:
            raise BluepySnapError("You need to set edge_ids in get.")
        if properties is None:
            return self.ids(edge_ids)
        return super().get(edge_ids, properties)

    def afferent_nodes(self, target: NodeGroup, unique: bool = True) -> CircuitNodeIds:
        """Get afferent CircuitNodeIDs for given target ``node_id``.

        Notes:
            Afferent nodes are nodes projecting an outgoing edge to one of the ``target`` node.

        Args:
            target: the target you want to resolve
            and use as target nodes.
            unique: If ``True``, return only unique afferent node IDs.

        Returns:
            Afferent CircuitNodeIDs for all the targets from all edge population.
        """
        target_ids = self._circuit.nodes.ids(target)
        result = self._get_ids_from_pop(
            lambda x: (x.afferent_nodes(target_ids), x.source.name), CircuitNodeIds
        )
        if unique:
            result.unique(inplace=True)
        return result

    def efferent_nodes(self, source: NodeGroup, unique: bool = True) -> CircuitNodeIds:
        """Get efferent node IDs for given source ``node_id``.

        Notes:
            Efferent nodes are nodes receiving an incoming edge from one of the ``source`` node.

        Args:
            source: the source you want to resolve
                and use as source nodes.
            unique: If ``True``, return only unique afferent node IDs.

        Returns:
            Efferent CircuitNodeIDs for all the sources from all edge population.
        """
        source_ids = self._circuit.nodes.ids(source)
        result = self._get_ids_from_pop(
            lambda x: (x.efferent_nodes(source_ids), x.target.name), CircuitNodeIds
        )
        if unique:
            result.unique(inplace=True)
        return result

    @overload
    def pathway_edges(
        self, source: NodeGroup = None, target: NodeGroup = None, properties: None = None
    ) -> CircuitEdgeIds: ...
    @overload
    def pathway_edges(
        self, source: NodeGroup, target: NodeGroup, properties: str | Collection[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    @overload
    def pathway_edges(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        *,
        properties: str | Collection[str],
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    def pathway_edges(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        properties: str | Collection[str] | None = None,
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Get edges corresponding to ``source`` -> ``target`` connections.

        Args:
            source: source node group
            target: target node group
            properties: None / edge property name / list of edge property names

        Returns:
            - CircuitEdgeIds, if ``properties`` is None or empty;
            - Otherwise a generator, as returned by :meth:`Edges.get`.
        """
        return self._pathway_edges(source, target, properties)

    def _pathway_edges(
        self, source: NodeGroup, target: NodeGroup, properties: str | Collection[str] | None
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Implementation of pathway_edges, without the overloads.

        The methods that forward ``properties`` call it, because ty fails to match a call to the
        overloads of pathway_edges when all the arguments are unions (pyright accepts it).
        """
        if source is None and target is None:
            raise BluepySnapError("Either `source` or `target` should be specified")

        source_ids = self._circuit.nodes.ids(source)
        target_ids = self._circuit.nodes.ids(target)

        result = self._get_ids_from_pop(
            lambda x: (x.pathway_edges(source_ids, target_ids), x.name), CircuitEdgeIds
        )

        # len instead of a truthiness test, which raises for numpy arrays with more than one item
        if properties is not None and len(properties) > 0:
            return self.get(result, properties)
        return result

    @overload
    def afferent_edges(self, node_id: NodeGroup, properties: None = None) -> CircuitEdgeIds: ...
    @overload
    def afferent_edges(
        self, node_id: NodeGroup, properties: str | Collection[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    def afferent_edges(
        self, node_id: NodeGroup, properties: str | Collection[str] | None = None
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Get afferent edges for given ``node_id``.

        Args:
            node_id: Target node ID.
            properties: An edge property name, a list of edge property names, or None.

        Returns:
            - CircuitEdgeIds, if ``properties`` is None;
            - Otherwise a generator, as returned by :meth:`Edges.get`.
        """
        return self._pathway_edges(source=None, target=node_id, properties=properties)

    @overload
    def efferent_edges(self, node_id: NodeGroup, properties: None = None) -> CircuitEdgeIds: ...
    @overload
    def efferent_edges(
        self, node_id: NodeGroup, properties: str | Collection[str]
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    def efferent_edges(
        self, node_id: NodeGroup, properties: str | Collection[str] | None = None
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Get efferent edges for given ``node_id``.

        Args:
            node_id: source node ID
            properties: None / edge property name / list of edge property names

        Returns:
            - CircuitEdgeIds, if ``properties`` is None;
            - Otherwise a generator, as returned by :meth:`Edges.get`.
        """
        return self._pathway_edges(source=node_id, target=None, properties=properties)

    @overload
    def pair_edges(
        self, source_node_id: NodeGroup, target_node_id: NodeGroup, properties: None = None
    ) -> CircuitEdgeIds: ...
    @overload
    def pair_edges(
        self,
        source_node_id: NodeGroup,
        target_node_id: NodeGroup,
        properties: str | Collection[str],
    ) -> Iterator[tuple[str, pd.DataFrame]]: ...
    def pair_edges(
        self,
        source_node_id: NodeGroup,
        target_node_id: NodeGroup,
        properties: str | Collection[str] | None = None,
    ) -> CircuitEdgeIds | Iterator[tuple[str, pd.DataFrame]]:
        """Get edges corresponding to ``source_node_id`` -> ``target_node_id`` connection.

        Args:
            source_node_id: source node ID
            target_node_id: target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            - CircuitEdgeIds, if ``properties`` is None;
            - Otherwise a generator, as returned by :meth:`Edges.get`.
        """
        return self._pathway_edges(
            source=source_node_id, target=target_node_id, properties=properties
        )

    @overload
    def iter_connections(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        return_edge_ids: Literal[False] = False,
        return_edge_count: Literal[False] = False,
    ) -> Iterator[Connection]: ...
    @overload
    def iter_connections(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        *,
        return_edge_ids: Literal[True],
        return_edge_count: Literal[False] = False,
    ) -> Iterator[ConnectionWithEdgeIds]: ...
    @overload
    def iter_connections(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        return_edge_ids: Literal[False] = False,
        *,
        return_edge_count: Literal[True],
    ) -> Iterator[ConnectionWithCount]: ...
    @overload
    def iter_connections(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        return_edge_ids: bool = False,
        return_edge_count: bool = False,
    ) -> Iterator[Connection | ConnectionWithCount | ConnectionWithEdgeIds]: ...
    def iter_connections(
        self,
        source: NodeGroup = None,
        target: NodeGroup = None,
        return_edge_ids: bool = False,
        return_edge_count: bool = False,
    ) -> Iterator[Connection | ConnectionWithCount | ConnectionWithEdgeIds]:
        """Iterate through ``source`` -> ``target`` connections.

        Args:
            source: source node group
            target: target node group
            return_edge_count: if True, edge count is added to yield result
            return_edge_ids: if True, edge ID list is added to yield result

        ``return_edge_count`` and ``return_edge_ids`` are mutually exclusive.

        Yields:
            - (source_node_id, target_node_id, edge_ids) if ``return_edge_ids`` is True;
            - (source_node_id, target_node_id, edge_count) if ``return_edge_count`` is True;
            - (source_node_id, target_node_id) otherwise.
        """
        if return_edge_ids and return_edge_count:
            raise BluepySnapError(
                "`return_edge_count` and `return_edge_ids` are mutually exclusive"
            )
        for pop in self.values():
            yield from pop.iter_connections(
                source=source,
                target=target,
                return_edge_ids=return_edge_ids,
                return_edge_count=return_edge_count,
            )

    def __getstate__(self) -> "Circuit":
        """Make Edges pickle-able, without storing state of caches."""
        return self._circuit

    def __setstate__(self, state: "Circuit") -> None:
        """Load from pickle state."""
        self.__init__(state)
