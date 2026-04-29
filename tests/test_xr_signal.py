import unittest

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_signal as collected
from sim_data_analyzer.signal_filters import filter_signal

try:
    import dask.array as da
except ImportError:
    da = None


class TestCollectedXRSignal(unittest.TestCase):
    def test_smoke_exports(self):
        self.assertTrue(hasattr(collected, 'interp_time_outliers'))
        self.assertTrue(hasattr(collected, 'filter_xr_signal'))

    def test_interp_time_outliers_fixes_isolated_outlier(self):
        tt = np.arange(7, dtype=float)
        X = xr.DataArray(
            np.array([1.0, 1.0, 1.0, -5.0, 1.0, 1.0, 1.0]),
            dims=['time'],
            coords={'time': tt},
        )
        out = collected.interp_time_outliers(X)
        expected = xr.DataArray(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dims=['time'],
            coords={'time': tt},
        )
        xr.testing.assert_identical(out, expected.assign_attrs(out.attrs))

    def test_interp_time_outliers_preserves_clean_signal(self):
        tt = np.linspace(0.0, 1.0, 11)
        X = xr.DataArray(np.sin(tt), dims=['time'], coords={'time': tt})
        out = collected.interp_time_outliers(X)
        xr.testing.assert_allclose(out, X.astype(float))

    def test_interp_time_outliers_handles_multichannel_input(self):
        tt = np.arange(7, dtype=float)
        data = np.array([
            [1.0, 1.0, 1.0, -5.0, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        ])
        X = xr.DataArray(
            data,
            dims=['chan', 'time'],
            coords={'chan': ['a', 'b'], 'time': tt},
            attrs={'source': 'test'},
        )
        out = collected.interp_time_outliers(X)
        self.assertEqual(out.attrs['source'], 'test')
        self.assertEqual(out.dims, X.dims)
        np.testing.assert_allclose(out.sel(chan='a').values, np.ones(7))
        np.testing.assert_allclose(out.sel(chan='b').values, data[1])

    def test_adjacent_outliers_are_left_unchanged(self):
        tt = np.arange(7, dtype=float)
        values = np.array([1.0, 1.0, -5.0, -5.0, 1.0, 1.0, 1.0])
        X = xr.DataArray(values, dims=['time'], coords={'time': tt})
        out = collected.interp_time_outliers(X)
        np.testing.assert_allclose(out.values, values)

    def test_missing_time_dim_raises(self):
        X = xr.DataArray(np.arange(5), dims=['sample'])
        with self.assertRaises(ValueError):
            collected.interp_time_outliers(X)

    def test_filter_xr_signal_preserves_dims_and_coords(self):
        tt = np.arange(0.0, 1.0, 0.01)
        xx = np.sin(2 * np.pi * 10 * tt) + 0.2 * np.sin(2 * np.pi * 40 * tt)
        X = xr.DataArray(xx, dims=['time'], coords={'time': tt})
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0))
        self.assertEqual(out.dims, X.dims)
        xr.testing.assert_allclose(out.coords['time'], X.coords['time'])

    def test_filter_xr_signal_handles_multichannel_input(self):
        tt = np.arange(0.0, 1.0, 0.01)
        data = np.vstack([
            np.sin(2 * np.pi * 10 * tt) + 0.1 * np.sin(2 * np.pi * 40 * tt),
            np.cos(2 * np.pi * 10 * tt) + 0.1 * np.cos(2 * np.pi * 40 * tt),
        ])
        X = xr.DataArray(
            data,
            dims=['chan', 'time'],
            coords={'chan': ['a', 'b'], 'time': tt},
        )
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0))
        self.assertEqual(out.dims, X.dims)
        self.assertEqual(out.shape, X.shape)

    def test_filter_xr_signal_matches_core_row_by_row(self):
        tt = np.arange(0.0, 1.0, 0.01)
        data = np.vstack([
            np.sin(2 * np.pi * 10 * tt) + 0.1 * np.sin(2 * np.pi * 40 * tt),
            np.cos(2 * np.pi * 10 * tt) + 0.1 * np.cos(2 * np.pi * 40 * tt),
        ])
        X = xr.DataArray(
            data,
            dims=['chan', 'time'],
            coords={'chan': ['a', 'b'], 'time': tt},
        )
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0), order=3)
        expected = np.vstack([
            filter_signal(row, t=tt, fband=(8.0, 12.0), order=3, btype='bandpass')
            for row in data
        ])
        np.testing.assert_allclose(out.values, expected)

    def test_filter_xr_signal_invalid_time_position_raises(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(
            np.vstack([np.sin(2 * np.pi * 10 * tt), np.cos(2 * np.pi * 10 * tt)]).T,
            dims=['time', 'chan'],
            coords={'time': tt, 'chan': ['a', 'b']},
        )
        with self.assertRaises(ValueError):
            collected.filter_xr_signal(X, fband=(8.0, 12.0))

    def test_filter_xr_signal_compute_false_leaves_non_dask_input_realized(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt})
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0), compute=False)
        self.assertFalse(hasattr(out.data, 'compute'))

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_filter_xr_signal_compute_false_preserves_deferred_behavior(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt}).chunk({'time': 50})
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0), compute=False)
        self.assertTrue(hasattr(out.data, 'compute'))

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_filter_xr_signal_compute_true_returns_realized_result(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt}).chunk({'time': 50})
        out = collected.filter_xr_signal(X, fband=(8.0, 12.0), compute=True)
        self.assertFalse(hasattr(out.data, 'chunks'))

    def test_filter_xr_signal_store_proc_info_true_writes_proc_steps(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt})
        out = collected.filter_xr_signal(
            X, fband=(8.0, 12.0), btype='bandpass', store_proc_info=True
        )
        self.assertIn('proc_steps', out.attrs)
        self.assertEqual(out.attrs['proc_steps'][-1]['name'], 'filter_xr_signal')
        self.assertEqual(out.attrs['proc_steps'][-1]['params']['btype'], 'bandpass')
        self.assertEqual(out.attrs['proc_steps'][-1]['params']['fband'], [8.0, 12.0])

    def test_filter_xr_signal_store_proc_info_appends_existing_steps(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(
            np.sin(2 * np.pi * 10 * tt),
            dims=['time'],
            coords={'time': tt},
            attrs={'proc_steps': [{'name': 'seed', 'params': {'a': 1}}]},
        )
        out = collected.filter_xr_signal(X, fband=20.0, btype='lowpass', store_proc_info=True)
        self.assertEqual(len(out.attrs['proc_steps']), 2)
        self.assertEqual(out.attrs['proc_steps'][0]['name'], 'seed')
        self.assertEqual(out.attrs['proc_steps'][1]['name'], 'filter_xr_signal')


if __name__ == '__main__':
    unittest.main()
