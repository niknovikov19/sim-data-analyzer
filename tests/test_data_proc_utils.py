import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

from sim_data_analyzer import data_proc_utils as collected


SHARED_RATE_CV_NAMES = [
    "calc_pop_rate",
    "calc_net_rates",
    "calc_pop_cv",
    "calc_net_cvs",
]


A1_DYNAMICS_NAMES = [
    "calc_pop_rate_dynamics",
    "calc_net_rate_dynamics",
]


def _load_module(name: str, path: str, prepend_sys_path: str | None = None):
    if prepend_sys_path is not None:
        sys.path.insert(0, prepend_sys_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if prepend_sys_path is not None:
            sys.path.pop(0)
    return module


def _assert_equal_nested(testcase: unittest.TestCase, left, right):
    if isinstance(left, np.ndarray):
        testcase.assertIsInstance(right, np.ndarray)
        np.testing.assert_allclose(left, right)
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


def _make_spikes():
    pop_cell = [
        np.array([0.10, 0.20, 0.40, 0.90]),
        np.array([0.15, 0.45, 0.75]),
    ]
    pop_combined = [np.array([0.10, 0.15, 0.20, 0.40, 0.45, 0.75, 0.90])]
    net_cell = {
        "IT2": pop_cell,
        "PV2": [np.array([0.05, 0.35, 0.65])],
    }
    net_combined = {
        "IT2": pop_combined,
        "PV2": [np.array([0.05, 0.35, 0.65])],
    }
    net_size = {"IT2": 2, "PV2": 1}
    return pop_cell, pop_combined, net_cell, net_combined, net_size


class TestCollectedDataProcUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.a1 = _load_module(
            "a1_data_proc_utils",
            str(repo_root / "A1-OUinp/analysis/ou_tuning/data_proc_utils.py"),
        )
        cls.model_tuner = _load_module(
            "mt_data_proc_utils",
            str(repo_root / "model_tuner/model_tuner/data_proc/data_proc_utils.py"),
        )
        sim_res_analyzer_dir = repo_root / "sim_res_analyzer/code"
        cls.sim_res_analyzer = _load_module(
            "sim_res_analyzer_parse_utils",
            str(sim_res_analyzer_dir / "sim_res_parse_utils.py"),
            prepend_sys_path=str(sim_res_analyzer_dir),
        )
        (cls.pop_cell, cls.pop_combined,
         cls.net_cell, cls.net_combined, cls.net_size) = _make_spikes()

    def test_smoke_exports(self):
        for name in SHARED_RATE_CV_NAMES + A1_DYNAMICS_NAMES:
            self.assertTrue(hasattr(collected, name), name)

    def test_signatures_match_sources(self):
        for name in SHARED_RATE_CV_NAMES:
            collected_sig = inspect.signature(getattr(collected, name))
            a1_sig = inspect.signature(getattr(self.a1, name))
            mt_sig = inspect.signature(getattr(self.model_tuner, name))
            self.assertEqual(str(collected_sig), str(a1_sig), name)
            self.assertEqual(str(collected_sig), str(mt_sig), name)
        for name in A1_DYNAMICS_NAMES:
            collected_sig = inspect.signature(getattr(collected, name))
            a1_sig = inspect.signature(getattr(self.a1, name))
            self.assertEqual(str(collected_sig), str(a1_sig), name)

    def test_shared_rate_cv_equivalence(self):
        call_map = [
            ("calc_pop_rate", (self.pop_combined, (0.0, 1.0)), {"ncells": 2}),
            ("calc_pop_rate", (self.pop_cell, (0.0, 1.0)), {}),
            ("calc_net_rates", (self.net_combined, (0.0, 1.0)), {"ncells": self.net_size}),
            ("calc_net_rates", (self.net_cell, (0.0, 1.0)), {}),
            ("calc_pop_cv", (self.pop_cell, (0.0, 1.0)), {}),
            ("calc_pop_cv", (self.pop_cell, (0.0, 1.0)), {"avg_result": False}),
            ("calc_net_cvs", (self.net_cell, (0.0, 1.0)), {}),
            ("calc_net_cvs", (self.net_cell, (0.0, 1.0)), {"avg_result": False}),
        ]
        for name, args, kwargs in call_map:
            expected_a1 = getattr(self.a1, name)(*args, **kwargs)
            expected_mt = getattr(self.model_tuner, name)(*args, **kwargs)
            actual = getattr(collected, name)(*args, **kwargs)
            _assert_equal_nested(self, expected_a1, expected_mt)
            _assert_equal_nested(self, actual, expected_a1)

    def test_a1_guard_on_calc_pop_rate(self):
        with self.assertRaises(ValueError):
            collected.calc_pop_rate(self.pop_cell, (0.0, 1.0), ncells=2)

    def test_rate_dynamics_equivalence_to_a1(self):
        call_map = [
            ("calc_pop_rate_dynamics", (self.pop_combined, (0.0, 1.0)), {"dt_bin": 0.1, "ncells": 2}),
            ("calc_pop_rate_dynamics", (self.pop_cell, (0.0, 1.0)), {"dt_bin": 0.1}),
            ("calc_pop_rate_dynamics", (self.pop_combined, (0.0, 1.0)), {"dt_bin": 0.1, "ncells": 2, "epoch_len": 0.5}),
            ("calc_pop_rate_dynamics", (self.pop_combined, (0.0, 1.0)), {"dt_bin": 0.1, "ncells": 2, "tau_smooth": 0.2}),
            ("calc_net_rate_dynamics", (self.net_combined, (0.0, 1.0)), {"dt_bin": 0.1, "ncells": self.net_size}),
        ]
        for name, args, kwargs in call_map:
            expected = getattr(self.a1, name)(*args, **kwargs)
            actual = getattr(collected, name)(*args, **kwargs)
            _assert_equal_nested(self, actual, expected)

    def test_combined_rate_dynamics_matches_sim_res_analyzer_core(self):
        spike_times = self.pop_combined[0]
        expected = self.sim_res_analyzer.calc_rate_dynamics(
            spike_times, (0.0, 1.0), 0.1, pop_sz=2
        )
        actual = collected.calc_pop_rate_dynamics(
            [spike_times], (0.0, 1.0), dt_bin=0.1, ncells=2
        )
        _assert_equal_nested(self, actual, expected)


if __name__ == "__main__":
    unittest.main()
