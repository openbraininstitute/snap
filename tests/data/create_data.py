#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv
import itertools as it
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import h5py
import libsonata
import numpy as np


@dataclass
class SonataAttribute:
    name: str
    type: type
    prefix: bool


NODE_TYPES = [
    SonataAttribute("node_type_id", type=np.int32, prefix=False),
    SonataAttribute("node_group_id", type=np.uint32, prefix=False),
    SonataAttribute("node_group_index", type=np.uint32, prefix=False),
    SonataAttribute("etype", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("layer", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("model_template", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("model_type", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("morph_class", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("morphology", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("mtype", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("synapse_class", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("other1", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("other2", type=np.int64, prefix=True),
    SonataAttribute("orientation_w", type=np.float32, prefix=True),
    SonataAttribute("orientation_x", type=np.float32, prefix=True),
    SonataAttribute("orientation_y", type=np.float32, prefix=True),
    SonataAttribute("orientation_z", type=np.float32, prefix=True),
    SonataAttribute("rotation_angle_xaxis", type=np.float32, prefix=True),
    SonataAttribute("rotation_angle_yaxis", type=np.float32, prefix=True),
    SonataAttribute("rotation_angle_zaxis", type=np.float32, prefix=True),
    SonataAttribute("x", type=np.float32, prefix=True),
    SonataAttribute("y", type=np.float32, prefix=True),
    SonataAttribute("z", type=np.float32, prefix=True),
    SonataAttribute("@dynamics:holding_current", type=np.float32, prefix=True),
    SonataAttribute("@dynamics:input_resistance", type=np.float32, prefix=True),
    SonataAttribute("@dynamics:threshold_current", type=np.float32, prefix=True),
]
NODE_TYPES = {attr.name: attr for attr in NODE_TYPES}

EDGE_TYPES = [
    SonataAttribute("edge_type_id", type=np.int8, prefix=False),
    SonataAttribute("edge_group_id", type=np.int8, prefix=False),
    SonataAttribute("edge_group_index", type=np.int8, prefix=False),
    SonataAttribute("afferent_center_x", type=np.float32, prefix=True),
    SonataAttribute("afferent_center_y", type=np.float32, prefix=True),
    SonataAttribute("afferent_center_z", type=np.float32, prefix=True),
    SonataAttribute("afferent_section_id", type=np.uint64, prefix=True),
    SonataAttribute("afferent_section_pos", type=np.float32, prefix=True),
    SonataAttribute("afferent_section_type", type=np.uint8, prefix=True),
    SonataAttribute("afferent_segment_id", type=np.uint8, prefix=True),
    SonataAttribute("afferent_segment_offset", type=np.float32, prefix=True),
    SonataAttribute("afferent_surface_x", type=np.float32, prefix=True),
    SonataAttribute("afferent_surface_y", type=np.float32, prefix=True),
    SonataAttribute("afferent_surface_z", type=np.float32, prefix=True),
    SonataAttribute("efferent_center_x", type=np.float32, prefix=True),
    SonataAttribute("efferent_center_y", type=np.float32, prefix=True),
    SonataAttribute("efferent_center_z", type=np.float32, prefix=True),
    SonataAttribute("efferent_section_id", type=np.uint64, prefix=True),
    SonataAttribute("efferent_section_pos", type=np.float32, prefix=True),
    SonataAttribute("efferent_surface_x", type=np.float32, prefix=True),
    SonataAttribute("efferent_surface_y", type=np.float32, prefix=True),
    SonataAttribute("efferent_surface_z", type=np.float32, prefix=True),
    SonataAttribute("conductance", type=np.float32, prefix=True),
    SonataAttribute("decay_time", type=np.float32, prefix=True),
    SonataAttribute("delay", type=np.float32, prefix=True),
    SonataAttribute("depression_time", type=np.float32, prefix=True),
    SonataAttribute("facilitation_time", type=np.float32, prefix=True),
    SonataAttribute("n_rrp_vesicles", type=np.uint8, prefix=True),
    SonataAttribute("spine_length", type=np.float32, prefix=True),
    SonataAttribute("syn_type_id", type=np.uint8, prefix=True),
    SonataAttribute("syn_weight", type=np.float32, prefix=True),
    SonataAttribute("u_syn", type=np.float32, prefix=True),
    SonataAttribute("@dynamics:param1", type=np.float64, prefix=True),
    SonataAttribute("other1", type=h5py.string_dtype(), prefix=True),
    SonataAttribute("other2", type=np.int32, prefix=True),
]
EDGE_TYPES = {attr.name: attr for attr in EDGE_TYPES}


@dataclass
class Edges:
    src: str
    tgt: str
    connections: list[tuple[int, int]]


def _expand_values(attr, value, count):
    if isinstance(value, str):
        ds_value = [value] * count
    elif isinstance(value, Sequence):
        assert len(value) == count, f"For {attr}, {len(value)} != (count) {count}"
        ds_value = value
    elif isinstance(value, Iterable):
        ds_value = list(it.islice(value, count))
    else:
        ds_value = [value] * count

    return ds_value


def _attr_to_ds_name(attr_name):
    """Convert attribute name to HDF5 dataset path within group '0'.

    '@dynamics:foo' -> 'dynamics_params/foo'
    """
    if attr_name.startswith("@dynamics:"):
        return "dynamics_params/" + attr_name[len("@dynamics:"):]
    return attr_name


def make_nodes(filename, populations):
    """Create a nodes HDF5 file with multiple populations.

    Args:
        filename: output HDF5 file path
        populations: dict of {pop_name: (count, wanted_attributes)}
    """
    with h5py.File(filename, "w") as h5:
        for pop_name, (count, wanted_attributes) in populations.items():
            dg = h5.create_group(f"/nodes/{pop_name}")

            for attr, value in wanted_attributes.items():
                typ = NODE_TYPES[attr]
                if typ.prefix:
                    ds_name = "0/" + _attr_to_ds_name(attr)
                else:
                    ds_name = typ.name
                ds_value = _expand_values(attr, value, count)
                dg.create_dataset(name=ds_name, data=ds_value, dtype=typ.type)

            # group "0" is required by libsonata function open_population
            if "0" not in dg:
                dg.create_group("0")


def make_edges(filename, populations):
    """Create an edges HDF5 file with multiple populations.

    Args:
        filename: output HDF5 file path
        populations: dict of {pop_name: (edges, wanted_attributes)}
    """
    with h5py.File(filename, "w") as h5:
        for pop_name, (edges, wanted_attributes) in populations.items():
            src_ids, tgt_ids = zip(*edges.connections, strict=True)
            count = len(src_ids)

            dg = h5.create_group(f"/edges/{pop_name}")

            for attr, value in wanted_attributes.items():
                typ = EDGE_TYPES[attr]
                if typ.prefix:
                    ds_name = "0/" + _attr_to_ds_name(attr)
                else:
                    ds_name = typ.name
                ds_value = _expand_values(attr, value, count)
                dg.create_dataset(name=ds_name, data=ds_value, dtype=typ.type)

            ds = dg.create_dataset("source_node_id", data=np.array(src_ids, dtype=np.uint64))
            ds.attrs["node_population"] = edges.src
            ds = dg.create_dataset("target_node_id", data=np.array(tgt_ids, dtype=np.uint64))
            ds.attrs["node_population"] = edges.tgt

    # Write indices after file is closed and reopened by libsonata
    for pop_name, (edges, _) in populations.items():
        src_ids, tgt_ids = zip(*edges.connections, strict=True)
        libsonata.EdgePopulation.write_indices(
            filename,
            pop_name,
            source_node_count=max(src_ids) + 1,
            target_node_count=max(tgt_ids) + 1,
        )


def make_default_nodes():
    """Create nodes.h5 with 'default' (3 nodes) and 'default2' (4 nodes) populations."""
    default_attrs = {
        "node_type_id": [-1, -1, -1],
        "node_group_id": [0, 0, 0],
        "node_group_index": [0, 1, 2],
        "etype": ["etype1", "etype0", "etype1"],
        "layer": ["layer2", "layer6", "layer6"],
        "model_template": ["hoc:small_bio-A", "hoc:small_bio-B", "hoc:small_bio-C"],
        "model_type": ["biophysical", "biophysical", "biophysical"],
        "morph_class": ["PYR", "INT", "PYR"],
        "morphology": ["morph-A", "morph-B", "morph-C"],
        "mtype": ["L2_X", "L6_Y", "L6_Y"],
        "synapse_class": ["EXC", "INH", "ECX"],
        "orientation_w": [1.0, 1.0, 1.0],
        "orientation_x": [0.0, 0.0, 0.0],
        "orientation_y": [0.0, 0.0, 0.0],
        "orientation_z": [0.0, 0.0, 0.0],
        "rotation_angle_xaxis": [-0.0, -0.24526736, -0.0],
        "rotation_angle_yaxis": [0.74036866, 0.23011371, 1.0894345],
        "rotation_angle_zaxis": [-0.0, 2.6707377, -0.0],
        "x": [101.0, 201.0, 301.0],
        "y": [102.0, 202.0, 302.0],
        "z": [103.0, 203.0, 303.0],
        "@dynamics:holding_current": [1.1, 2.2, 3.3],
        "@dynamics:input_resistance": [1.0, 1.1, 1.2],
        "@dynamics:threshold_current": [1.1, 2.2, 3.3],
    }

    default2_attrs = {
        "node_type_id": [1, 1, 1, 1],
        "node_group_id": [0, 0, 0, 0],
        "node_group_index": [0, 1, 2, 3],
        "etype": ["etype1", "etype0", "etype1", "etype1"],
        "layer": ["layer7", "layer8", "layer8", "layer2"],
        "model_template": ["hoc:small_bio", "hoc:small_bio", "hoc:small_bio", "hoc:small_bio"],
        "model_type": ["biophysical", "biophysical", "biophysical", "biophysical"],
        "morph_class": ["PYR", "INT", "PYR", "PYR"],
        "morphology": ["morph-D", "morph-E", "morph-F", "morph-G"],
        "mtype": ["L7_X", "L8_Y", "L8_Y", "L9_Z"],
        "synapse_class": ["EXC", "INH", "EXC", "EXC"],
        "other1": ["A", "B", "C", "D"],
        "other2": [10, 11, 12, 13],
        "orientation_w": [1.0, 1.0, 1.0, 1.0],
        "orientation_x": [0.0, 0.0, 0.0, 0.0],
        "orientation_y": [0.0, 0.0, 0.0, 0.0],
        "orientation_z": [0.0, 0.0, 0.0, 0.0],
        "rotation_angle_xaxis": [0.0, 0.0, 0.0, 0.0],
        "rotation_angle_yaxis": [0.0, 0.0, 0.0, 0.0],
        "rotation_angle_zaxis": [0.0, 0.0, 0.0, 0.0],
        "x": [401.0, 501.0, 601.0, 701.0],
        "y": [402.0, 502.0, 602.0, 702.0],
        "z": [403.0, 503.0, 603.0, 703.0],
        "@dynamics:holding_current": [1.1, 2.2, 3.3, 4.4],
        "@dynamics:threshold_current": [1.1, 2.2, 3.3, 4.4],
    }

    make_nodes(
        filename="nodes.h5",
        populations={
            "default": (3, default_attrs),
            "default2": (4, default2_attrs),
        },
    )


def make_default_edges():
    """Create edges.h5 with 'default' and 'default2' populations (4 edges each)."""
    connections = [(2, 0), (0, 1), (0, 1), (2, 1)]

    common_attrs = {
        "edge_type_id": [-1, -1, -1, -1],
        "edge_group_id": [0, 0, 0, 0],
        "edge_group_index": [0, 1, 2, 3],
        "afferent_center_x": [1110.0, 1111.0, 1112.0, 1113.0],
        "afferent_center_y": [1120.0, 1121.0, 1122.0, 1123.0],
        "afferent_center_z": [1130.0, 1131.0, 1132.0, 1133.0],
        "afferent_section_id": [0, 0, 0, 0],
        "afferent_section_pos": [0.0, 0.0, 0.0, 0.0],
        "afferent_section_type": [1, 2, 3, 4],
        "afferent_segment_id": [2, 3, 4, 5],
        "afferent_segment_offset": [1.0, 2.0, 3.0, 4.0],
        "afferent_surface_x": [1210.0, 1211.0, 1212.0, 1213.0],
        "afferent_surface_y": [1220.0, 1221.0, 1222.0, 1223.0],
        "afferent_surface_z": [1230.0, 1231.0, 1232.0, 1233.0],
        "efferent_center_x": [2110.0, 2111.0, 2112.0, 2113.0],
        "efferent_center_y": [2120.0, 2121.0, 2122.0, 2123.0],
        "efferent_center_z": [2130.0, 2131.0, 2132.0, 2133.0],
        "efferent_section_id": [0, 0, 0, 0],
        "efferent_section_pos": [0.0, 0.0, 0.0, 0.0],
        "efferent_surface_x": [2210.0, 2211.0, 2212.0, 2213.0],
        "efferent_surface_y": [2220.0, 2221.0, 2222.0, 2223.0],
        "efferent_surface_z": [2230.0, 2231.0, 2232.0, 2233.0],
        "conductance": [99.571655, 13.985464, 96.29266, 82.18058],
        "decay_time": [2.0, 3.0, 4.0, 5.0],
        "delay": [99.894485, 88.186226, 52.188114, 11.105833],
        "depression_time": [3.0, 4.0, 5.0, 6.0],
        "facilitation_time": [4.0, 5.0, 6.0, 7.0],
        "n_rrp_vesicles": [3, 4, 5, 6],
        "spine_length": [6.0, 7.0, 8.0, 9.0],
        "syn_type_id": [4, 5, 6, 7],
        "syn_weight": [1.0, 1.0, 1.0, 1.0],
        "u_syn": [5.0, 6.0, 7.0, 8.0],
        "@dynamics:param1": [0.0, 1.0, 2.0, 3.0],
    }

    default2_attrs = {
        **common_attrs,
        "other1": ["A", "B", "C", "D"],
        "other2": [10, 11, 12, 13],
    }

    edges_default = Edges(src="default", tgt="default", connections=connections)
    edges_default2 = Edges(src="default", tgt="default", connections=connections)

    make_edges(
        filename="edges.h5",
        populations={
            "default": (edges_default, common_attrs),
            "default2": (edges_default2, default2_attrs),
        },
    )


if __name__ == "__main__":
    make_default_nodes()
    make_default_edges()
    print("Generated nodes.h5 and edges.h5")
