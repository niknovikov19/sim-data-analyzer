import importlib.util
import inspect
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_diff as collected

try:
    import dask.array as da
except ImportError:
    da = None


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_signal_xr():
    tt = np.arange(0.0, 1.0, 0.1)
    yy = np.array([100.0, 200.0, 300.0, 400.0])
    data = np.array([
        np.sin(2 * np.pi * tt),
        np.cos(2 * np.pi * tt),
        np.sin(4 * np.pi * tt),
        np.cos(4 * np.pi * tt),
    ])
    return xr.DataArray(
        data,
        dims=['y', 'time'],
        coords={'y': yy, 'time': tt},
        attrs={'source': 'test'},
    )


class TestCollectedXRDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.a1 = _load_module(
            'a1_xr_diff',
            str(repo_root / 'A1-OUinp/analysis/xr_proc/xr_diff.py'),
        )
        cls.sim_res = _load_module(
            'sim_res_xr_diff',
            str(repo_root / 'sim_res_analyzer/code/xr_proc/xr_diff.py'),
        )
        cls.X = _make_signal_xr()

    def test_smoke_exports(self):
        for name in ['calc_xr_diff', 'calc_xr_bipolar', 'calc_xr_csd']:
            self.assertTrue(hasattr(collected, name), name)

    def test_signature_shape(self):
        src_sig = inspect.signature(self.a1.calc_xr_diff)
        collected_sig = inspect.signature(collected.calc_xr_diff)
        src_params = list(src_sig.parameters)
        collected_params = list(collected_sig.parameters)
        self.assertEqual(collected_params[:len(src_params)], src_params)
        self.assertEqual(collected_params[-1], 'store_proc_info')

    def test_calc_xr_diff_matches_old_lineages(self):
        for n in [1, 2]:
            expected_a1 = self.a1.calc_xr_diff(self.X, n=n)
            expected_sr = self.sim_res.calc_xr_diff(self.X, n=n)
            actual = collected.calc_xr_diff(self.X, n=n)
            xr.testing.assert_identical(expected_a1, expected_sr)
            xr.testing.assert_allclose(actual, expected_a1)
            self.assertEqual(actual.dims, expected_a1.dims)
            for coord_name in actual.coords:
                xr.testing.assert_identical(actual.coords[coord_name], expected_a1.coords[coord_name])
            self.assertEqual(actual.attrs.get('source'), 'test')

    def test_bipolar_and_csd_wrappers_match_diff(self):
        xr.testing.assert_allclose(
            collected.calc_xr_bipolar(self.X),
            collected.calc_xr_diff(self.X, n=1),
        )
        xr.testing.assert_allclose(
            collected.calc_xr_csd(self.X),
            collected.calc_xr_diff(self.X, n=2),
        )

    def test_store_proc_info_true_writes_proc_steps(self):
        out = collected.calc_xr_diff(self.X, n=2, store_proc_info=True)
        self.assertIn('proc_steps', out.attrs)
        self.assertEqual(out.attrs['proc_steps'][-1]['name'], 'calc_xr_diff')
        self.assertEqual(out.attrs['proc_steps'][-1]['params']['n'], 2)
        self.assertEqual(out.attrs['proc_steps'][-1]['params']['ydim'], 'y')

    def test_store_proc_info_appends_existing_steps(self):
        X = self.X.assign_attrs({'proc_steps': [{'name': 'seed', 'params': {'a': 1}}]})
        out = collected.calc_xr_bipolar(X, store_proc_info=True)
        self.assertEqual(len(out.attrs['proc_steps']), 2)
        self.assertEqual(out.attrs['proc_steps'][0]['name'], 'seed')
        self.assertEqual(out.attrs['proc_steps'][1]['name'], 'calc_xr_bipolar')

    def test_store_proc_info_false_preserves_attrs(self):
        out = collected.calc_xr_csd(self.X, store_proc_info=False)
        self.assertEqual(out.attrs.get('source'), 'test')
        self.assertNotIn('proc_steps', out.attrs)

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_compute_false_preserves_deferred_behavior(self):
        X = self.X.chunk({'time': 5})
        out = collected.calc_xr_diff(X, n=1, compute=False)
        self.assertTrue(hasattr(out.data, 'compute'))

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_compute_true_returns_realized_result(self):
        X = self.X.chunk({'time': 5})
        out = collected.calc_xr_diff(X, n=1, compute=True)
        self.assertFalse(hasattr(out.data, 'chunks'))


if __name__ == '__main__':
    unittest.main()
