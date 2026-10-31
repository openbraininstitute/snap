# Copyright (c) 2020, EPFL/Blue Brain Project

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

"""Access to node set data.

For more information see:
https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#node-sets-file
"""

import copy
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any, Self

import libsonata
import numpy as np
import numpy.typing as npt

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class NodeSet:
    """Access to single node set."""

    def __init__(self, node_sets: libsonata.NodeSets, name: str) -> None:
        """Initializes a single node set object.

        Args:
            node_sets: libsonata NodeSets instance.
            name: name of the node set.

        Returns:
            A NodeSet object.
        """
        self._node_sets = node_sets
        self._name = name

    def get_ids(
        self, population: libsonata.NodePopulation, raise_missing_property: bool = True
    ) -> npt.NDArray[np.integer] | list[int]:
        """Get the resolved node set as ids."""
        try:
            return self._node_sets.materialize(self._name, population).flatten()
        except libsonata.SonataError as e:
            if not raise_missing_property and "No such attribute" in e.args[0]:
                return []
            raise BluepySnapError(*e.args) from e


class NodeSets:
    """Access to node sets data."""

    def __init__(self, content: dict[str, Any], instance: libsonata.NodeSets) -> None:
        """Initializes a node set object from a node sets file.

        Args:
            content: Node sets as a dictionary.
            instance: ``libsonata`` node sets instance.

        Returns:
            A NodeSets object.
        """
        self._content = content
        self._instance = instance

    @classmethod
    def from_file(cls, filepath: str | os.PathLike[str]) -> Self:
        """Create NodeSets instance from a file."""
        content = utils.load_json(filepath)
        instance = libsonata.NodeSets.from_file(filepath)
        return cls(content, instance)

    @classmethod
    def from_string(cls, content: str) -> Self:
        """Create NodeSets instance from a JSON string."""
        instance = libsonata.NodeSets(content)
        return cls(json.loads(content), instance)

    @classmethod
    def from_dict(cls, content: Mapping[str, Any]) -> Self:
        """Create NodeSets instance from a dict."""
        return cls.from_string(json.dumps(content))

    @property
    def content(self) -> dict[str, Any]:
        """Access (a copy of) the node sets contents."""
        return copy.deepcopy(self._content)

    @property
    def to_libsonata(self) -> libsonata.NodeSets:
        """Libsonata instance of the NodeSets."""
        return self._instance

    def update(self, node_sets: "NodeSets") -> set[str]:
        """Update the contents of the node set.

        Args:
            node_sets: The node set to extend this node set with.

        Returns:
            Names of any overwritten node sets.
        """
        if isinstance(node_sets, NodeSets):
            overwritten = self._instance.update(node_sets.to_libsonata)
            self._content.update(node_sets.content)
            return overwritten

        raise BluepySnapError(
            f"Unexpected type: '{type(node_sets).__name__}' (expected: '{self.__class__.__name__}')"
        )

    def __contains__(self, name: object) -> bool:
        """Check if node set exists."""
        if isinstance(name, str):
            return name in self._instance.names

        raise BluepySnapError(f"Unexpected type: '{type(name).__name__}' (expected: 'str')")

    def __getitem__(self, name: str) -> NodeSet:
        """Return a node set instance for the given node set name."""
        if name not in self:
            raise BluepySnapError(f"Undefined node set: '{name}'")
        return NodeSet(self._instance, name)

    def __iter__(self) -> Iterator[str]:
        """Iter through the different node sets names."""
        return iter(self._instance.names)
