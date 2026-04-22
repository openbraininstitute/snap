import numpy.testing as npt
import pytest
from utils import TEST_DATA_DIR

import bluepysnap
from bluepysnap.exceptions import BluepySnapError

PRE = "Population_default"
POST = "Population_default"

"""
In [14]: a.source_nodes(a.select_all())
Out[14]: array([2, 0, 0, 2], dtype=uint64)
In [15]: a.target_nodes(a.select_all())
Out[15]: array([0, 1, 1, 1], dtype=uint64)
"""


@pytest.fixture
def stats():
    return bluepysnap.Circuit(TEST_DATA_DIR / "circuit_config.json").edges["default"].stats


def test_divergence_by_synapses(stats):
    actual = stats.divergence(PRE, POST, by="synapses")
    npt.assert_equal(actual, [2, 0, 2])


def test_divergence_by_connections(stats):
    actual = stats.divergence(PRE, POST, by="connections")
    npt.assert_equal(actual, [1, 0, 2])


def test_divergence_error(stats):
    pytest.raises(BluepySnapError, stats.divergence, PRE, POST, by="err")


def test_convergence_by_synapses(stats):
    actual = stats.convergence(PRE, POST, by="synapses")
    npt.assert_equal(actual, [1, 3, 0])


def test_convergence_by_connections(stats):
    actual = stats.convergence(PRE, POST, by="connections")
    npt.assert_equal(actual, [1, 2, 0])


def test_convergence_error(stats):
    pytest.raises(BluepySnapError, stats.convergence, PRE, POST, by="err")
