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

"""Circuit ids."""

import abc
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, Self, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from bluepysnap import utils
from bluepysnap._doctools import AbstractDocSubstitutionMeta
from bluepysnap.circuit_ids_types import CircuitEdgeId, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError


class CircuitIds(abc.ABC):
    """High performances CircuitID container.

    This abstract class aims at defining the global ids for the circuit. It is the base class of
    CircuitNodeIds and CircuitEdgeIds.
    The pandas.MultiIndex class is used as backend and can be accessed using CircuitIds.index.
    A global circuit node id is the combination of a population and an ID inside this
    population.
    """

    def __init__(self, index: pd.MultiIndex, sort_index: bool = True) -> None:
        """Return an instance of CircuitIds.

        Args:
            index: a multi index from pandas with the population names as
                first level and node IDs as the second level.
            sort_index: if true sort the index per population and then node IDs.
        """
        if not isinstance(index, pd.MultiIndex):
            raise BluepySnapError("index must be a pandas.MultiIndex object.")
        if sort_index:
            # best perf compared to sort_values. Sorted by population and ids.
            index = index.sortlevel()[0]
        self.index = index

    @classmethod
    def _instance(cls, index: pd.MultiIndex, sort_index: bool = True) -> Self:
        """The instance returned by the functions."""
        return cls(index, sort_index=sort_index)

    @classmethod
    def from_arrays(
        cls, populations: npt.ArrayLike, population_ids: npt.ArrayLike, sort_index: bool = True
    ) -> Self:
        """Create a set of ids using two arrays population(s) and id(s).

        Args:
            populations: a sequence of populations. If the population is a
                string then all the ids will be connected to this population.
            population_ids: a sequence of node IDs or a single node ID.
            sort_index: will sort the index if set to True. Otherwise the ordering from the
                user inputs is kept. Sorting the index can result in better performances.

        Returns:
            a set of global node IDs created via the populations and the node IDs
            provided.
        """
        populations = np.asarray(populations)
        population_ids = utils.ensure_ids(population_ids)

        if len(populations) != len(population_ids):
            raise BluepySnapError(
                "populations and population_ids must have the same size. "
                f"{len(populations)} != {len(population_ids)}"
            )

        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index, sort_index=sort_index)

    @classmethod
    def from_dict(cls, ids_dictionary: Mapping[str, npt.ArrayLike]) -> Self:
        """Create a set of ids using a dictionary as input.

        Args:
            ids_dictionary: a dictionary with the population as keys and node IDs as
                values.

        Notes:
            with the support of python 2 we cannot guaranty the ordering so we force the sorting
            of ids.

        Returns:
            a set of global node IDs created via the provided dictionary.

        Examples:
            >>> CircuitIds.from_dict({"pop1": [0, 2, 4], "pop2": [1, 2, 5]})
        """
        populations = np.empty((0,), dtype=str)
        population_ids = np.empty((0,), dtype=utils.IDS_DTYPE)
        for population, ids in ids_dictionary.items():
            ids = utils.ensure_ids(ids)
            population_ids = np.append(population_ids, ids)
            populations = np.append(populations, np.full(ids.shape, fill_value=population))
        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index, sort_index=True)

    @classmethod
    def from_tuples(
        cls, circuit_id_list: Sequence[tuple[str, int]], sort_index: bool = True
    ) -> Self:
        """Create a set of ids using a list of CircuitId or tuples as input.

        Args:
            circuit_id_list: a list of CircuitId or list of tuples.
            sort_index: will sort the index if set to True. Otherwise the ordering from the
                user inputs is kept. Sorting the index can result in better performances.

        Returns:
            a set of global node IDs created via the provided dictionary.

        Examples:
            >>> CircuitIds.from_tuples([("pop1", 0), ("pop1", 2), ("pop2", 0)])
            >>> CircuitIds.from_tuples([CircuitId("pop1", 0), CircuitId("pop1", 2),
            >>>                             CircuitId("pop2", 0)])
        """
        return cls(pd.MultiIndex.from_tuples(circuit_id_list), sort_index=sort_index)

    def _locate(self, population: str) -> npt.NDArray[np.intp] | list[int]:
        """Returns the index indices corresponding to a given population.

        Args:
            population: the population name you want to locate inside the MultiIndex.

        Returns:
            indices corresponding to the population.
        """
        try:
            return self.index.get_locs(utils.ensure_list(population))
        except KeyError:
            return []

    @overload
    def filter_population(self, population: str, inplace: Literal[False] = False) -> Self: ...
    @overload
    def filter_population(self, population: str, inplace: Literal[True]) -> None: ...
    def filter_population(self, population: str, inplace: bool = False) -> Self | None:
        """Filter the IDs corresponding to a population.

        Args:
            population: the population you want to extract.
            inplace: if set to True. Do the transformation inplace.

        Returns:
            a filtered CircuitIds containing only IDs for the given population.
        """
        if not inplace:
            return self._instance(self.index[self._locate(population)], sort_index=False)
        self.index = self.index[self._locate(population)]
        return None

    def get_populations(self, unique: bool = False) -> npt.NDArray[Any]:
        """Returns all population values from the circuit node IDs."""
        if unique:
            return self.index.levels[0].to_numpy()
        return self.index.get_level_values(0).to_numpy()

    def get_ids(self, unique: bool = False) -> npt.NDArray[np.int64]:
        """Returns all the ID values from the circuit node IDs."""
        if unique:
            return self.index.levels[1].to_numpy()
        return self.index.get_level_values(1).to_numpy()

    def copy(self) -> Self:
        """Copy a CircuitIds."""
        return self._instance(self.index.copy())

    @overload
    def sort(self, inplace: Literal[False] = False) -> Self: ...
    @overload
    def sort(self, inplace: Literal[True]) -> None: ...
    def sort(self, inplace: bool = False) -> Self | None:
        """Sort the CircuitIds by population and then by ids.

        Args:
            inplace: if set to True. Do the transformation inplace.

        Notes:
            sorting a CircuitIds will gives better perf for the population extraction and the
            data extraction from a sorted dataframe.
        """
        if not inplace:
            return self._instance(self.index, sort_index=True)
        self.index = self.index.sortlevel()[0]
        return None

    @overload
    def append(self, other: "CircuitIds", inplace: Literal[False] = False) -> Self: ...
    @overload
    def append(self, other: "CircuitIds", inplace: Literal[True]) -> None: ...
    def append(self, other: "CircuitIds", inplace: bool = False) -> Self | None:
        """Append a CircuitIds to the current one.

        Args:
            other: the other CircuitIds to append to the current one.
            inplace: if set to True. Do the transformation inplace.
        """
        if not inplace:
            return self._instance(self.index.append(other.index))
        self.index = self.index.append(other.index)
        return None

    def _apply(self, fun: Callable[[pd.MultiIndex], pd.MultiIndex], inplace: bool) -> Self | None:
        """Apply fun on index."""
        res = self if inplace else self.copy()
        res.index = fun(res.index)
        if not inplace:
            return res
        return None

    @overload
    def sample(self, sample_size: int, inplace: Literal[False] = False) -> Self: ...
    @overload
    def sample(self, sample_size: int, inplace: Literal[True]) -> None: ...
    def sample(self, sample_size: int, inplace: bool = False) -> Self | None:
        """Sample a CircuitIds.

        Notes:
            this function takes randomly ``sample_size`` values of a circuit node ids.

        Args:
            sample_size: the size of the sample. If the size of the sample is greater than
                the size of the CircuitIds then all ids are taken and shuffled.
            inplace: if set to True. Do the transformation inplace.
        """
        sampling = np.random.choice(len(self), size=min(sample_size, len(self)), replace=False)
        return self._apply(lambda x: x[sampling], inplace)

    @overload
    def limit(self, limit_size: int, inplace: Literal[False] = False) -> Self: ...
    @overload
    def limit(self, limit_size: int, inplace: Literal[True]) -> None: ...
    def limit(self, limit_size: int, inplace: bool = False) -> Self | None:
        """Limit the size of a CircuitIds.

        Notes:
            this function takes the first ``limit_size`` values of a circuit node ids.

        Args:
            limit_size: the size of the sample. If the size of the sample is greater than
                the size of the CircuitIds then all ids are kept.
            inplace: if set to True. Do the transformation inplace.
        """
        return self._apply(lambda x: x[:limit_size], inplace)

    @overload
    def unique(self, inplace: Literal[False] = False) -> Self: ...
    @overload
    def unique(self, inplace: Literal[True]) -> None: ...
    def unique(self, inplace: bool = False) -> Self | None:
        """Returns only unique CircuitIds.

        Notes:
            this function does not sort the ids
        """
        return self._apply(lambda x: x.unique(), inplace)

    @overload
    def intersection(self, circuit_ids: "CircuitIds", inplace: Literal[False] = False) -> Self: ...
    @overload
    def intersection(self, circuit_ids: "CircuitIds", inplace: Literal[True]) -> None: ...
    def intersection(self, circuit_ids: "CircuitIds", inplace: bool = False) -> Self | None:
        """Take the intersection of this CircuitIds and the input.

        The index of the resulting object is sorted if ``inplace=False``.
        Otherwise, the orginal order of the index is kept.

        Args:
            circuit_ids: The CircuitIds to intersect with.
            inplace: if set to True, do the transformation inplace.
        """
        return self._apply(lambda x: x.intersection(circuit_ids.index), inplace)

    def to_csv(self, filepath: str | os.PathLike[str]) -> None:
        """Save CircuitIds to csv format."""
        self.index.to_frame(index=False).to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath: str | os.PathLike[str]) -> Self:
        """Load CircuitIds from csv."""
        return cls(pd.MultiIndex.from_frame(pd.read_csv(filepath)))

    def tolist(self) -> list[tuple[str, int]]:
        """Convert the CircuitIds to a list of tuples."""
        return self.index.tolist()

    def __call__(self, *args: Any, **kwargs: Any) -> pd.MultiIndex:
        """Allows to use the CircuitIds as normal indices in a DataFrame."""
        return self.index

    def __len__(self) -> int:
        """Return the length of the CircuitIds."""
        return len(self.index)

    def __repr__(self) -> str:
        """Correct repr of the CircuitIds."""
        return repr(self.index).replace("MultiIndex", self.__class__.__name__, 1)

    def __eq__(self, other: object) -> bool:
        """Equality for the CircuitIds."""
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        return self.index.equals(other.index)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[str, int]]:  # pragma: no cover
        """Iterator on the CircuitIds."""

    @abc.abstractmethod
    def __getitem__(self, item: int) -> tuple[str, int]:  # pragma: no cover
        """Getter on the CircuitIds."""


class CircuitNodeIds(
    CircuitIds,
    metaclass=AbstractDocSubstitutionMeta,
    source_word="CircuitId",
    target_word="CircuitNodeId",
):
    """High performances CircuitNodeID container."""

    def __init__(self, index: pd.MultiIndex, sort_index: bool = True) -> None:
        """Return an instance of CircuitNodeIds.

        Args:
            index: a multi index from pandas with the population names as
                first level and node IDs as the second level.
            sort_index: if true sort the index per population and then node IDs.
        """
        super().__init__(index, sort_index)
        self.index.names = ["population", "node_ids"]

    def __iter__(self) -> Iterator[CircuitNodeId]:
        """Iterator on the CircuitNodeIds."""
        for index in self.index:
            yield CircuitNodeId(*index)

    def __getitem__(self, item: int) -> CircuitNodeId:
        """Getter on the CircuitNodeIds."""
        # pandas-stubs types an item of a MultiIndex as Hashable, it's a tuple
        return CircuitNodeId(*self.index[item])  # ty: ignore[invalid-argument-type]


class CircuitEdgeIds(
    CircuitIds,
    metaclass=AbstractDocSubstitutionMeta,
    source_word="CircuitId",
    target_word="CircuitEdgeId",
):
    """High performances CircuitEdgeID container."""

    def __init__(self, index: pd.MultiIndex, sort_index: bool = True) -> None:
        """Return an instance of CircuitEdgeIds.

        Args:
            index: a multi index from pandas with the population names as
                first level and node IDs as the second level.
            sort_index: if true sort the index per population and then node IDs.
        """
        super().__init__(index, sort_index)
        self.index.names = ["population", "edge_ids"]

    def __iter__(self) -> Iterator[CircuitEdgeId]:
        """Iterator on the CircuitEdgeIds."""
        for index in self.index:
            yield CircuitEdgeId(*index)

    def __getitem__(self, item: int) -> CircuitEdgeId:
        """Getter on the CircuitEdgeIds."""
        # pandas-stubs types an item of a MultiIndex as Hashable, it's a tuple
        return CircuitEdgeId(*self.index[item])  # ty: ignore[invalid-argument-type]
