import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import xarray as xr

from sim_data_analyzer import xr_adapters as collected
from sim_data_analyzer.tests.test_netpyne_res_parse_utils import _make_sim_result


XR_ADAPTER_NAMES = [
    "get_trace_xr",
    "get_voltages_xr",
    "get_lfp_xr",
    "get_pop_lfps_xr",
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


def _assert_xr_dict_equal(testcase: unittest.TestCase, left, right):
    testcase.assertEqual(set(left), set(right))
    for key in left:
        if left[key] is None:
            testcase.assertIsNone(right[key])
        else:
            testcase.assertIsInstance(right[key], xr.DataArray)
            xr.testing.assert_identical(left[key], right[key])


def _make_xr_sim_result():
    sim_result = _make_sim_result()
    sim_result["net"]["pops"]["SOM2"] = {
        "cellGids": [],
        "tags": {"ynormRange": [0.3, 0.4]},
    }
    sim_result["net"]["params"]["popParams"]["SOM2"] = {"cellType": "SOM"}
    return sim_result


class TestCollectedXRAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.a1 = _load_module(
            "a1_parse_utils_xr",
            str(repo_root / "A1-OUinp/analysis/ou_tuning/netpyne_res_parse_utils.py"),
        )
        sim_res_analyzer_dir = repo_root / "sim_res_analyzer/code"
        cls.sim_res_analyzer = _load_module(
            "sim_res_analyzer_parser_xr",
            str(sim_res_analyzer_dir / "sim_res_parser.py"),
            prepend_sys_path=str(sim_res_analyzer_dir),
        )
        cls.sim_result = _make_xr_sim_result()

    def test_smoke_exports(self):
        for name in XR_ADAPTER_NAMES:
            self.assertTrue(hasattr(collected, name), name)

    def test_signatures_match_a1(self):
        for name in ["get_trace_xr", "get_voltages_xr"]:
            collected_sig = inspect.signature(getattr(collected, name))
            a1_sig = inspect.signature(getattr(self.a1, name))
            self.assertEqual(str(collected_sig), str(a1_sig), name)

    def test_signatures_match_sim_res_analyzer_lfp_helpers(self):
        sig_map = {
            "get_lfp_xr": "_sim_res_to_xr_LFP",
            "get_pop_lfps_xr": "_sim_res_to_xr_pop_LFPs",
        }
        for name, src_name in sig_map.items():
            collected_sig = inspect.signature(getattr(collected, name))
            src_sig = inspect.signature(getattr(self.sim_res_analyzer, src_name))
            self.assertEqual(len(collected_sig.parameters), len(src_sig.parameters), name)
            self.assertEqual(
                [p.kind for p in collected_sig.parameters.values()],
                [p.kind for p in src_sig.parameters.values()],
                name,
            )

    def test_get_trace_xr_equivalence(self):
        calls = [
            ((self.sim_result, "V_soma"), {}),
            ((self.sim_result, "V_soma"), {"t_limits": (1.0, 3.0)}),
            ((self.sim_result, "V_soma"), {"ms": False}),
        ]
        for args, kwargs in calls:
            expected = self.a1.get_trace_xr(*args, **kwargs)
            actual = collected.get_trace_xr(*args, **kwargs)
            _assert_xr_dict_equal(self, actual, expected)

    def test_get_voltages_xr_equivalence(self):
        calls = [
            ((self.sim_result,), {}),
            ((self.sim_result,), {"t_limits": (1.0, 3.0)}),
            ((self.sim_result,), {"ms": False}),
        ]
        for args, kwargs in calls:
            expected = self.a1.get_voltages_xr(*args, **kwargs)
            actual = collected.get_voltages_xr(*args, **kwargs)
            _assert_xr_dict_equal(self, actual, expected)

    def test_get_lfp_xr_equivalence(self):
        expected = self.sim_res_analyzer._sim_res_to_xr_LFP(self.sim_result)
        actual = collected.get_lfp_xr(self.sim_result)
        xr.testing.assert_identical(actual, expected)

    def test_get_pop_lfps_xr_equivalence(self):
        expected = self.sim_res_analyzer._sim_res_to_xr_pop_LFPs(self.sim_result)
        actual = collected.get_pop_lfps_xr(self.sim_result)
        xr.testing.assert_identical(actual, expected)

    def test_empty_population_returns_none(self):
        traces = collected.get_voltages_xr(self.sim_result)
        self.assertIn("SOM2", traces)
        self.assertIsNone(traces["SOM2"])


if __name__ == "__main__":
    unittest.main()
