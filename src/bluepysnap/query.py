"""Module to process search queries of nodes/edges."""

import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError

if TYPE_CHECKING:
    from bluepysnap.node_sets import NodeSets
    from bluepysnap.nodes.node_population import NodePopulation

# A boolean mask selecting the rows of a DataFrame. True and False select all or none of the rows.
Mask: TypeAlias = bool | npt.NDArray[np.bool_]
# A query, as described in the documentation of NodePopulation.ids: a dict from property names
# or special keys (e.g. ``$or``, ``$node_set``) to values.
Queries: TypeAlias = dict[str, Any]

# this constant is not part of the sonata standard
NODE_ID_KEY = "node_id"
EDGE_ID_KEY = "edge_id"
POPULATION_KEY = "population"
POPULATION_TYPE_KEY = "population_type"
OR_KEY = "$or"
AND_KEY = "$and"
REGEX_KEY = "$regex"
NODE_SET_KEY = "$node_set"
VALUE_KEYS = {REGEX_KEY}
ALL_KEYS = {
    NODE_ID_KEY,
    EDGE_ID_KEY,
    POPULATION_KEY,
    POPULATION_TYPE_KEY,
    OR_KEY,
    AND_KEY,
    NODE_SET_KEY,
} | VALUE_KEYS


def _logical_and(masks: Iterable[bool | npt.ArrayLike]) -> Mask:
    """Calculate and return the "logical and" of the given masks.

    Args:
        masks: list of array-like, or booleans.

    Returns:
        the resulting mask as a numpy array, or a bool.
        Return True if the masks list is empty, similarly to all([]).
    """
    filtered_masks = []
    for mask in masks:
        if mask is False:
            return False
        if mask is True:
            continue
        filtered_masks.append(mask)
    if not filtered_masks:
        return True
    # reduce(operator.and_, masks) is faster than np.logical_and.reduce() or np.all()
    return reduce(operator.and_, (np.asarray(m, dtype=bool) for m in filtered_masks))


def _logical_or(masks: Iterable[bool | npt.ArrayLike]) -> Mask:
    """Calculate and return the "logical or" of the given masks.

    Args:
        masks: list of array-like, or booleans.

    Returns:
        the resulting mask as a numpy array, or a bool.
        Return False if the masks list is empty, similarly to any([]).
    """
    filtered_masks = []
    for mask in masks:
        if mask is True:
            return True
        if mask is False:
            continue
        filtered_masks.append(mask)
    if not filtered_masks:
        return False
    # reduce(operator.or_, masks) is faster than np.logical_or.reduce() or np.any()
    return reduce(operator.or_, (np.asarray(m, dtype=bool) for m in filtered_masks))


def _complex_query(prop: pd.Series, query: Mapping[str, str]) -> Mask:
    result: Mask = True
    for key, value in query.items():
        if key == REGEX_KEY:
            result = _logical_and([result, prop.str.match(value + r"\Z")])
        else:
            raise BluepySnapError(f"Unknown query modifier: '{key}'")
    return result


def _positional_mask(
    data: pd.DataFrame, ids: int | Sequence[int] | npt.NDArray[np.integer] | None
) -> Mask:
    """Positional mask for the node IDs.

    Args:
        data: data
        ids: the ids array. If None all ids are selected.

    Examples:
        if the data set contains 5 nodes:
        _positional_mask(data, [0,2]) --> [True, False, True, False, False]
    """
    if ids is None:
        return True
    if isinstance(ids, int):
        ids = [ids]
    elif len(ids) == 0:
        return False
    mask = np.full(len(data), fill_value=False)
    indices = data.index.get_indexer(pd.Index(ids, copy=False))
    mask[indices[indices > -1]] = True
    return mask


def _circuit_mask(
    data: pd.DataFrame, population_name: str, population_type: str, queries: dict[str, Any]
) -> tuple[dict[str, Any], Mask]:
    """Handle the population, node ID queries."""
    populations = queries.pop(POPULATION_KEY, None)
    types = queries.pop(POPULATION_TYPE_KEY, None)
    if populations is not None and population_name not in utils.ensure_list(populations):
        ids = []
    elif types is not None and population_type not in utils.ensure_list(types):
        ids = []
    else:
        ids = queries.pop(NODE_ID_KEY, queries.pop(EDGE_ID_KEY, None))
    return queries, _positional_mask(data, ids)


def _properties_mask(
    data: pd.DataFrame, population_name: str, population_type: str, queries: dict[str, Any]
) -> Mask:
    """Return mask of IDs matching `props` dict."""
    unknown_props = set(queries) - set(data.columns) - ALL_KEYS
    if unknown_props:
        return False

    queries, mask = _circuit_mask(data, population_name, population_type, queries)
    if mask is False or isinstance(mask, np.ndarray) and not mask.any():
        # Avoid fail and/or processing time if wrong population or no nodes
        return False

    for prop, values in queries.items():
        prop = data[prop]
        if np.issubdtype(prop.dtype.type, np.floating):
            v1, v2 = values
            prop_mask = np.logical_and(prop >= v1, prop <= v2)
        elif isinstance(values, Mapping):
            prop_mask = _complex_query(prop, values)
        elif utils.is_iterable(values):
            prop_mask = prop.isin(values)
        else:
            prop_mask = prop == values
        mask = _logical_and([mask, prop_mask])
    return mask


def traverse_queries_bottom_up(
    queries: dict[str, Any], traverse_fn: Callable[[dict[str, Any], str], None]
) -> None:
    """Traverse queries tree from leaves to root, left to right.

    Args:
        queries: queries
        traverse_fn: function to execute on each node of `queries` in traverse order
    """
    for key in list(queries.keys()):
        if key in {OR_KEY, AND_KEY}:
            for subquery in queries[key]:
                traverse_queries_bottom_up(subquery, traverse_fn)
        elif isinstance(queries[key], Mapping):
            if VALUE_KEYS & set(queries[key]):
                if not set(queries[key]).issubset(VALUE_KEYS):
                    raise BluepySnapError("Value operators can't be used with plain values")
            else:
                traverse_queries_bottom_up(queries[key], traverse_fn)
        traverse_fn(queries, key)


def get_properties(queries: dict[str, Any]) -> set[str]:
    """Extracts properties names from `queries`.

    Args:
        queries: queries

    Returns:
        set of properties names
    """

    def _collect(_: dict[str, Any], query_key: str) -> None:
        if query_key not in ALL_KEYS:
            props.add(query_key)

    props: set[str] = set()
    traverse_queries_bottom_up(queries, _collect)
    return props


def resolve_nodesets(
    node_sets: "NodeSets",
    population: "NodePopulation",
    queries: Queries,
    raise_missing_prop: bool,
) -> dict[str, Any]:
    """Resolve node sets in queries.

    Args:
        node_sets: node sets instance
        population: node population
        queries: queries to resolve
        raise_missing_prop: raise if property not present in population

    Returns:
        queries with resolved node sets
    """

    def _resolve(queries: dict[str, Any], queries_key: str) -> None:
        if queries_key == NODE_SET_KEY:
            if AND_KEY not in queries:
                queries[AND_KEY] = []
            node_set = node_sets[queries[queries_key]]
            queries[AND_KEY].append(
                {NODE_ID_KEY: node_set.get_ids(population.to_libsonata, raise_missing_prop)}
            )
            del queries[queries_key]

    resolved_queries = deepcopy(queries)
    traverse_queries_bottom_up(resolved_queries, _resolve)

    return resolved_queries


def resolve_ids(
    data: pd.DataFrame, population_name: str, population_type: str, queries: Queries
) -> npt.NDArray[np.bool_]:
    """Returns an index mask of `data` for given `queries`.

    Args:
        data: data
        population_name: population name of `data`
        population_type: population type
        queries: queries

    Returns:
        index mask
    """

    def _merge_queries_masks(queries: dict[str, Any]) -> Mask:
        return _logical_and(queries.values())

    def _collect(queries: dict[str, Any], queries_key: str) -> None:
        # each queries value is replaced with a bit mask of corresponding ids
        if queries_key == OR_KEY:
            # children are already resolved masks due to traverse order
            children_mask = [_merge_queries_masks(query) for query in queries[queries_key]]
            queries[queries_key] = _logical_or(children_mask)
        elif queries_key == AND_KEY:
            # children are already resolved masks due to traverse order
            children_mask = [_merge_queries_masks(query) for query in queries[queries_key]]
            queries[queries_key] = _logical_and(children_mask)
        else:
            queries[queries_key] = _properties_mask(
                data, population_name, population_type, {queries_key: queries[queries_key]}
            )

    queries = deepcopy(queries)
    traverse_queries_bottom_up(queries, _collect)
    result = _merge_queries_masks(queries)
    return np.full(len(data), fill_value=result) if isinstance(result, bool) else result
