import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_spect as collected

try:
    import dask.array as da
except ImportError:
    da = None


XR_SPECT_NAMES = [
    "calc_xr_welch",
    "calc_xr_cpsd",
    "calc_xr_tf",
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


def _make_signal_xr():
    fs = 100.0
    tt = np.arange(0.0, 2.0, 1.0 / fs)
    x1 = np.sin(2 * np.pi * 10 * tt) + 0.25 * np.sin(2 * np.pi * 20 * tt)
    x2 = np.cos(2 * np.pi * 10 * tt)
    X1 = xr.DataArray(x1, dims=["time"], coords={"time": tt})
    X2 = xr.DataArray(x2, dims=["time"], coords={"time": tt})
    return X1, X2


def _make_multi_signal_xr():
    fs = 100.0
    tt = np.arange(0.0, 2.0, 1.0 / fs)
    data = np.vstack([
        np.sin(2 * np.pi * 10 * tt),
        np.cos(2 * np.pi * 20 * tt),
    ])
    return xr.DataArray(
        data,
        dims=["chan", "time"],
        coords={"chan": ["a", "b"], "time": tt},
    )


def _assert_dataarray_equal(left, right):
    xr.testing.assert_allclose(left, right)
    assert left.dims == right.dims
    for coord_name in left.coords:
        xr.testing.assert_allclose(left.coords[coord_name], right.coords[coord_name])


class TestCollectedXRSpect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.xr_utils_neuro = _load_module(
            "xr_utils_neuro_spect",
            str(repo_root / "xr_utils_neuro/xr_spect.py"),
        )
        cls.a1 = _load_module(
            "a1_xr_spect",
            str(repo_root / "A1-OUinp/analysis/xr_proc/xr_spect.py"),
        )
        sim_res_analyzer_dir = repo_root / "sim_res_analyzer/code/xr_proc"
        cls.sim_res_analyzer = _load_module(
            "sim_res_analyzer_xr_spect",
            str(sim_res_analyzer_dir / "xr_spect.py"),
            prepend_sys_path=str(sim_res_analyzer_dir),
        )
        cls.X1, cls.X2 = _make_signal_xr()
        cls.Xmulti = _make_multi_signal_xr()

    def test_smoke_exports(self):
        for name in XR_SPECT_NAMES:
            self.assertTrue(hasattr(collected, name), name)

    def test_signature_shape(self):
        for name in XR_SPECT_NAMES:
            collected_sig = inspect.signature(getattr(collected, name))
            src_sig = inspect.signature(getattr(self.xr_utils_neuro, name))
            src_params = list(src_sig.parameters)
            collected_params = list(collected_sig.parameters)
            self.assertEqual(collected_params[:len(src_params)], src_params, name)
            self.assertEqual(collected_params[-2:], ["compute", "store_proc_info"], name)

    def test_calc_xr_welch_matches_xr_utils_neuro(self):
        calls = [
            ((self.X1,), {}),
            ((self.X1,), {"fmax": 30, "win_len": 0.4, "win_overlap": 0.25}),
            ((self.Xmulti,), {"fmax": 30}),
            ((self.X1,), {"fs": 100.0, "fmax": 30}),
        ]
        for args, kwargs in calls:
            expected = self.xr_utils_neuro.calc_xr_welch(*args, **kwargs)
            actual = collected.calc_xr_welch(*args, **kwargs)
            _assert_dataarray_equal(actual, expected)

    def test_calc_xr_cpsd_matches_xr_utils_neuro(self):
        expected = self.xr_utils_neuro.calc_xr_cpsd(self.X1, self.X2, fmax=30)
        actual = collected.calc_xr_cpsd(self.X1, self.X2, fmax=30)
        _assert_dataarray_equal(actual, expected)

    def test_calc_xr_tf_matches_xr_utils_neuro(self):
        calls = [
            ((self.X1,), {}),
            ((self.X1,), {"fmax": 30, "win_len": 0.4, "win_overlap": 0.25}),
            ((self.Xmulti,), {"fmax": 30}),
        ]
        for args, kwargs in calls:
            expected = self.xr_utils_neuro.calc_xr_tf(*args, **kwargs)
            actual = collected.calc_xr_tf(*args, **kwargs)
            _assert_dataarray_equal(actual, expected)

    def test_calc_xr_welch_matches_sim_res_analyzer_overlap(self):
        expected = self.sim_res_analyzer.calc_xr_welch(self.X1, fmax=30)
        actual = collected.calc_xr_welch(self.X1, fmax=30)
        _assert_dataarray_equal(actual, expected)

    def test_calc_xr_tf_matches_a1_overlap(self):
        expected = self.a1.calc_xr_tf(self.X1, fmax=30)
        actual = collected.calc_xr_tf(self.X1, fmax=30)
        _assert_dataarray_equal(actual, expected)

    def test_invalid_time_dim_position_raises(self):
        Xbad = self.Xmulti.transpose("time", "chan")
        with self.assertRaises(ValueError):
            collected.calc_xr_welch(Xbad)
        with self.assertRaises(ValueError):
            collected.calc_xr_cpsd(Xbad, Xbad)
        with self.assertRaises(ValueError):
            collected.calc_xr_tf(Xbad)

    def test_compute_false_leaves_non_dask_input_realized(self):
        out = collected.calc_xr_welch(self.X1, compute=False)
        self.assertFalse(hasattr(out.data, "compute"))

    @unittest.skipIf(da is None, "dask is not installed")
    def test_compute_false_preserves_deferred_behavior(self):
        Xchunk = self.X1.chunk({"time": 50})
        out = collected.calc_xr_welch(Xchunk, compute=False)
        self.assertTrue(hasattr(out.data, "compute"))

    @unittest.skipIf(da is None, "dask is not installed")
    def test_compute_true_returns_realized_result(self):
        Xchunk = self.X1.chunk({"time": 50})
        out = collected.calc_xr_welch(Xchunk, compute=True)
        self.assertFalse(hasattr(out.data, "chunks"))

    def test_store_proc_info_false_preserves_attrs(self):
        Xin = self.X1.assign_attrs({"source": "test"})
        out = collected.calc_xr_welch(Xin, store_proc_info=False)
        self.assertEqual(out.attrs.get("source"), "test")
        self.assertNotIn("proc_steps", out.attrs)

    def test_store_proc_info_true_writes_proc_steps(self):
        out = collected.calc_xr_welch(self.X1, fmax=30, store_proc_info=True)
        self.assertIn("proc_steps", out.attrs)
        self.assertIsInstance(out.attrs["proc_steps"], list)
        self.assertEqual(len(out.attrs["proc_steps"]), 1)
        step = out.attrs["proc_steps"][0]
        self.assertEqual(step["name"], "calc_xr_welch")
        self.assertEqual(step["params"]["fmax"], 30)
        self.assertEqual(step["params"]["time_dim"], "time")
        self.assertEqual(step["params"]["fs"], 100.0)

    def test_store_proc_info_appends_existing_steps(self):
        Xin = self.X1.assign_attrs({"proc_steps": [{"name": "seed", "params": {"a": 1}}]})
        out = collected.calc_xr_tf(Xin, store_proc_info=True)
        self.assertEqual(len(out.attrs["proc_steps"]), 2)
        self.assertEqual(out.attrs["proc_steps"][0]["name"], "seed")
        self.assertEqual(out.attrs["proc_steps"][1]["name"], "calc_xr_tf")


if __name__ == "__main__":
    unittest.main()
