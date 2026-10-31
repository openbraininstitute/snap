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
"""Spike report access."""

from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
import pandas as pd
from libsonata import SonataError, SpikeReader

import bluepysnap._plotting
from bluepysnap import query
from bluepysnap._typing import IdsArray, NodeGroup
from bluepysnap.circuit_ids_types import IDS_DTYPE
from bluepysnap.exceptions import BluepySnapError

if TYPE_CHECKING:
    from bluepysnap.node_sets import NodeSets
    from bluepysnap.nodes.node_population import NodePopulation
    from bluepysnap.simulation import Simulation


def _get_reader(config: Any) -> SpikeReader:
    path = str(Path(config.output_dir) / config.spikes_file)
    return SpikeReader(path)


def _collect_spikes(spike_report: "SpikeReport") -> dict[str, "PopulationSpikeReport"]:
    return {
        population: PopulationSpikeReport(spike_report, population)
        for population in spike_report.population_names
    }


class PopulationSpikeReport:
    """Access to PopulationSpikeReport data."""

    def __init__(self, spike_report: "SpikeReport", population_name: str) -> None:
        """Initializes a PopulationSpikeReport object from a SpikeReport.

        Args:
            spike_report: SpikeReport containing this spike report population.
            population_name: the population name corresponding to this report.

        Returns:
            A PopulationSpikeReport object.
        """
        self.spike_report = spike_report
        self._spike_population = _get_reader(self.spike_report.config)[population_name]
        self._population_name = population_name

    @property
    def _node_sets(self) -> "NodeSets":
        """Access to simulation node sets."""
        return self.spike_report.simulation.node_sets

    @property
    def _sorted_by(self) -> str:
        """Access to the sorting attribute.

        Returns:
            the type of sorting used for this spike report.

            Returned values are:

               - ``'by_id'`` if the report is sorted by population node_ids
               - ``'by_time'`` if the report is sorted by timestamps
               - ``'none'`` if not sorted.
        """
        return self._spike_population.sorting

    @property
    def name(self) -> str:
        """Return the name of the population."""
        return self._population_name

    @cached_property
    def nodes(self) -> "NodePopulation":
        """Return the NodePopulation corresponding to this spike report."""
        return self.spike_report.simulation.circuit.nodes[self._population_name]

    def resolve_nodes(self, group: NodeGroup, raise_missing_property: bool = True) -> IdsArray:
        """Transform a node group into a node_id array."""
        if isinstance(group, str):
            group = self._node_sets[group]
        elif isinstance(group, Mapping):
            group = query.resolve_nodesets(
                self._node_sets, self.nodes, group, raise_missing_property
            )
        return self.nodes.ids(group=group, raise_missing_property=raise_missing_property)

    def get(
        self, group: NodeGroup = None, t_start: float | None = None, t_stop: float | None = None
    ) -> pd.Series:
        """Fetch spikes from the report.

        Args:
            group: Get spikes filtered by group. See NodePopulation.
            t_start: Include only spikes occurring at or after this time.
            t_stop: Include only spikes occurring at or before this time.

        Returns:
            return spiking node_ids indexed by sorted spike time.
        """
        node_ids = self.resolve_nodes(group).tolist()

        series_name = "ids"
        try:
            res = self._spike_population.get(node_ids=node_ids, tstart=t_start, tstop=t_stop)
        except SonataError as e:
            raise BluepySnapError(e) from e

        if not res:
            return pd.Series(
                data=[], index=pd.Index([], name="times"), name=series_name, dtype=IDS_DTYPE
            )

        # pylint: disable=unsubscriptable-object
        res = pd.DataFrame(data=res, columns=[series_name, "times"]).set_index("times")[series_name]
        if self._sorted_by != "by_time":
            res = res.sort_index()
        return res.astype(IDS_DTYPE)

    @cached_property
    def node_ids(self) -> IdsArray:
        """Returns the node ids present in the report.

        Returns:
            Numpy array containing the node_ids included in the report
        """
        return np.unique(self.get())


class FilteredSpikeReport:
    """Access to filtered SpikeReport data."""

    def __init__(
        self,
        spike_report: "SpikeReport",
        group: NodeGroup = None,
        t_start: float | None = None,
        t_stop: float | None = None,
    ) -> None:
        """Initialize a FilteredSpikeReport.

        A FilteredSpikeReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            spike_report: The SpikeReport to filter.
            group: Get spikes filtered by group. See NodePopulation.
            t_start: Include only frames occurring at or after this time.
            t_stop: Include only frames occurring at or before this time.

        Returns:
            A FilteredSpikeReport object.
        """
        self.spike_report = spike_report
        self.group = group
        self.t_start = t_start
        self.t_stop = t_stop

    @cached_property
    def report(self) -> pd.DataFrame:
        """Access to the report data.

        Returns:
            A DataFrame containing the data from the report. Row's indices are the
            different timestamps and the columns are ids and population names.
        """
        dfs = []
        for population in self.spike_report.population_names:
            spikes = self.spike_report[population]
            ids = spikes.resolve_nodes(self.group, raise_missing_property=False)
            data = spikes.get(group=ids, t_start=self.t_start, t_stop=self.t_stop).to_frame()
            data["population"] = np.full(len(data), population)

            dfs.append(data)

        if all(df.empty for df in dfs):
            res = dfs[0][:0]
        else:
            res = pd.concat([df for df in dfs if not df.empty])

        res["population"] = res["population"].astype("category")

        return res.sort_index()

    # pylint: disable=protected-access
    raster = bluepysnap._plotting.spike_raster
    firing_rate_histogram = bluepysnap._plotting.spikes_firing_rate_histogram
    isi = bluepysnap._plotting.spikes_isi
    firing_animation = bluepysnap._plotting.spikes_firing_animation


class SpikeReport:
    """Access to SpikeReport data."""

    def __init__(self, simulation: "Simulation") -> None:
        """Initializes a SpikeReport object from a simulation object.

        Args:
            simulation: Simulation containing this spike report.

        Returns:
            A SpikeReport object.
        """
        self._simulation = simulation

    @property
    def config(self) -> Any:
        """Access to the spike 'output' config part."""
        return self._simulation.output

    @property
    def time_start(self) -> float:
        """Returns the starting time of the simulation. Default is zero."""
        return self._simulation.time_start

    @property
    def time_stop(self) -> float:
        """Returns the stopping time of the simulation."""
        return self._simulation.time_stop

    @property
    def dt(self) -> float:
        """Returns the frequency of reporting in milliseconds."""
        return self._simulation.dt

    @property
    def time_units(self) -> str:
        """Returns the time unit of reporting."""
        raise NotImplementedError

    @property
    def simulation(self) -> "Simulation":
        """Return the Simulation object related to this spike report."""
        return self._simulation

    @contextmanager
    def log(self) -> Generator[TextIO]:
        """Context manager for the spike log file."""
        path = Path(self.config.output_dir) / self.config.log_file
        if not path.exists():
            raise BluepySnapError("Cannot find the log file for the spike report.")
        with path.open("r", encoding="utf-8") as f:
            yield f

    @cached_property
    def _spike_reader(self) -> SpikeReader:
        """Access to the libsonata SpikeReader."""
        return _get_reader(self.config)

    @cached_property
    def population_names(self) -> list[str]:
        """Returns the population names included in this report."""
        return sorted(self._spike_reader.get_population_names())

    @cached_property
    def _population(self) -> dict[str, PopulationSpikeReport]:
        """Collect the different PopulationSpikeReports."""
        return _collect_spikes(self)

    def __getitem__(self, population_name: str) -> PopulationSpikeReport:
        """Access the PopulationSpikeReport corresponding to the population 'population_name'."""
        return self._population[population_name]

    def __iter__(self) -> Iterator[str]:
        """Allows iteration over the different PopulationSpikeReports."""
        return iter(self._population)

    def filter(
        self, group: NodeGroup = None, t_start: float | None = None, t_stop: float | None = None
    ) -> FilteredSpikeReport:
        """Returns a FilteredSpikeReport.

        A FilteredSpikeReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            group: Get spikes filtered by group. See NodePopulation.
            t_start: Include only frames occurring at or after this time.
            t_stop: Include only frames occurring at or before this time.

        Returns:
            A FilteredSpikeReport object.
        """
        return FilteredSpikeReport(self, group, t_start, t_stop)
