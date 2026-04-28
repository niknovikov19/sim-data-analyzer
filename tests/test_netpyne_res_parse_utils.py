import importlib.util
import inspect
import unittest
from pathlib import Path

import numpy as np

from sim_data_analyzer import netpyne_res_parse_utils as collected


SHARED_COLLECTED_NAMES = [
    "get_pop_names",
    "get_lfp_coords",
    "get_record_times",
    "get_lfp",
    "get_pop_lfps",
    "get_pop_ylim",
    "get_layer_borders",
    "get_net_params",
    "get_pop_params",
    "get_pop_cell_gids",
    "get_sim_data",
    "get_pop_size",
    "get_net_size",
    "get_sim_duration",
    "get_pop_spikes",
    "get_net_spikes",
]


A1_ONLY_COLLECTED_NAMES = [
    "get_timestep",
    "get_pop_voltages",
    "get_voltages",
]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_sim_result():
    return {
        "simConfig": {
            "recordLFP": [[0, 100], [0, 300]],
            "recordStep": 1.0,
            "dt": 0.1,
            "duration": 6.0,
            "sizeY": 1000,
        },
        "net": {
            "pops": {
                "IT2": {"cellGids": [0, 1], "tags": {"ynormRange": [0.0, 0.2]}},
                "PV2": {"cellGids": [2], "tags": {"ynormRange": [0.2, 0.3]}},
            },
            "cells": [
                {"tags": {"pop": "IT2"}},
                {"tags": {"pop": "IT2"}},
                {"tags": {"pop": "PV2"}},
            ],
            "params": {
                "popParams": {
                    "IT2": {"cellType": "IT"},
                    "PV2": {"cellType": "PV"},
                }
            },
        },
        "simData": {
            "LFP": [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            "LFPPops": {
                "IT2": [
                    [10.0, 20.0],
                    [30.0, 40.0],
                    [50.0, 60.0],
                    [70.0, 80.0],
                    [90.0, 100.0],
                    [110.0, 120.0],
                ],
                "PV2": [
                    [1.5, 2.5],
                    [3.5, 4.5],
                    [5.5, 6.5],
                    [7.5, 8.5],
                    [9.5, 10.5],
                    [11.5, 12.5],
                ],
            },
            "spkid": [0, 1, 2, 0, 2],
            "spkt": [1.0, 2.0, 3.0, 4.5, 5.5],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "V_soma": {
                "cell_0": [-65.0, -64.0, -63.0, -62.0, -61.0],
                "cell_1": [-66.0, -65.0, -64.0, -63.0, -62.0],
                "cell_2": [-60.0, -59.0, -58.0, -57.0, -56.0],
            },
        },
    }


def _assert_equal_nested(testcase: unittest.TestCase, left, right):
    if isinstance(left, np.ndarray):
        testcase.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, list):
        testcase.assertIsInstance(right, list)
        testcase.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            _assert_equal_nested(testcase, x, y)
        return
    if isinstance(left, tuple):
        testcase.assertIsInstance(right, tuple)
        testcase.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            _assert_equal_nested(testcase, x, y)
        return
    if isinstance(left, dict):
        testcase.assertIsInstance(right, dict)
        testcase.assertEqual(set(left), set(right))
        for key in left:
            _assert_equal_nested(testcase, left[key], right[key])
        return
    testcase.assertEqual(left, right)


class TestCollectedNetPyNEParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.a1 = _load_module(
            "a1_parse_utils",
            str(repo_root / "A1-OUinp/analysis/ou_tuning/netpyne_res_parse_utils.py"),
        )
        cls.model_tuner = _load_module(
            "mt_parse_utils",
            str(repo_root / "model_tuner/model_tuner/data_proc/netpyne_res_parse_utils.py"),
        )
        cls.sim_result = _make_sim_result()

    def test_smoke_exports(self):
        for name in SHARED_COLLECTED_NAMES + A1_ONLY_COLLECTED_NAMES:
            self.assertTrue(hasattr(collected, name), name)

    def test_signatures_match_source_modules(self):
        for name in SHARED_COLLECTED_NAMES:
            collected_sig = inspect.signature(getattr(collected, name))
            a1_sig = inspect.signature(getattr(self.a1, name))
            mt_sig = inspect.signature(getattr(self.model_tuner, name))
            self.assertEqual(str(collected_sig), str(a1_sig), name)
            self.assertEqual(str(collected_sig), str(mt_sig), name)
        for name in A1_ONLY_COLLECTED_NAMES:
            collected_sig = inspect.signature(getattr(collected, name))
            a1_sig = inspect.signature(getattr(self.a1, name))
            self.assertEqual(str(collected_sig), str(a1_sig), name)

    def test_metadata_and_lfp_equivalence(self):
        call_map = {
            "get_pop_names": (self.sim_result,),
            "get_lfp_coords": (self.sim_result,),
            "get_record_times": (self.sim_result,),
            "get_lfp": (self.sim_result,),
            "get_pop_lfps": (self.sim_result,),
            "get_pop_ylim": (self.sim_result, "IT2"),
            "get_layer_borders": (self.sim_result,),
            "get_net_params": (self.sim_result,),
            "get_pop_params": (self.sim_result,),
            "get_pop_cell_gids": (self.sim_result, "IT2"),
            "get_sim_data": (self.sim_result,),
            "get_pop_size": (self.sim_result, "IT2"),
            "get_net_size": (self.sim_result,),
            "get_sim_duration": (self.sim_result,),
        }
        for name, args in call_map.items():
            expected_a1 = getattr(self.a1, name)(*args)
            expected_mt = getattr(self.model_tuner, name)(*args)
            actual = getattr(collected, name)(*args)
            _assert_equal_nested(self, expected_a1, expected_mt)
            _assert_equal_nested(self, actual, expected_a1)

    def test_a1_voltage_helper_equivalence(self):
        call_map = {
            "get_timestep": [
                ((self.sim_result,), {}),
            ],
            "get_pop_voltages": [
                ((self.sim_result, "IT2"), {}),
                ((self.sim_result, "IT2"), {"t_limits": (1.0, 3.0)}),
            ],
            "get_voltages": [
                ((self.sim_result,), {}),
                ((self.sim_result,), {"t_limits": (1.0, 3.0)}),
            ],
        }
        for name, calls in call_map.items():
            for args, kwargs in calls:
                expected = getattr(self.a1, name)(*args, **kwargs)
                actual = getattr(collected, name)(*args, **kwargs)
                _assert_equal_nested(self, actual, expected)

    def test_spike_helpers_equivalence(self):
        spike_calls = [
            ("get_pop_spikes", (self.sim_result, "IT2")),
            ("get_pop_spikes", (self.sim_result, "IT2"), {"combine_cells": False}),
            ("get_pop_spikes", (self.sim_result, "IT2"), {"t0": 0.002, "tmax": 0.005, "subtract_t0": False}),
            ("get_pop_spikes", (self.sim_result, "IT2"), {"ms": True}),
            ("get_net_spikes", (self.sim_result,)),
            ("get_net_spikes", (self.sim_result,), {"combine_cells": False}),
            ("get_net_spikes", (self.sim_result,), {"pop_names": ["PV2"], "ms": True}),
        ]
        for item in spike_calls:
            if len(item) == 2:
                name, args = item
                kwargs = {}
            else:
                name, args, kwargs = item
            expected_a1 = getattr(self.a1, name)(*args, **kwargs)
            expected_mt = getattr(self.model_tuner, name)(*args, **kwargs)
            actual = getattr(collected, name)(*args, **kwargs)
            _assert_equal_nested(self, expected_a1, expected_mt)
            _assert_equal_nested(self, actual, expected_a1)

    def test_combined_spikes_keep_list_wrapped_shape(self):
        spikes = collected.get_pop_spikes(self.sim_result, "IT2", combine_cells=True)
        self.assertIsInstance(spikes, list)
        self.assertEqual(len(spikes), 1)
        self.assertIsInstance(spikes[0], np.ndarray)

    def test_zero_tmax_is_not_replaced_by_full_duration(self):
        spikes = collected.get_pop_spikes(
            self.sim_result, "IT2", combine_cells=True, tmax=0
        )
        self.assertIsInstance(spikes, list)
        self.assertEqual(len(spikes), 1)
        np.testing.assert_array_equal(spikes[0], np.array([]))


if __name__ == "__main__":
    unittest.main()
