import unittest

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_signal as collected
from sim_data_analyzer.signal_filters import filter_signal

try:
    import dask.array as da
except ImportError:
    da = None


def _reference_crosscorr(values1, values2, sample_lags, subtract_mean=False, normalize=False):
    x1 = np.asarray(values1, dtype=float).copy()
    x2 = np.asarray(values2, dtype=float).copy()
    if subtract_mean:
        x1 -= x1.mean()
        x2 -= x2.mean()

    if normalize:
        denom = np.sqrt(np.sum(x1 * x1) * np.sum(x2 * x2))
        if denom <= np.finfo(float).eps or not np.isfinite(denom):
            return np.full(len(sample_lags), np.nan, dtype=float)
    else:
        denom = float(len(x1))

    out = []
    for lag in sample_lags:
        if lag == 0:
            numer = np.sum(x1 * x2)
        elif lag > 0:
            numer = np.sum(x2[:-lag] * x1[lag:])
        else:
            numer = np.sum(x2[-lag:] * x1[:lag])
        out.append(numer / denom)
    return np.asarray(out, dtype=float)


class TestCollectedXRSignal(unittest.TestCase):
    def test_smoke_exports(self):
        self.assertTrue(hasattr(collected, 'interp_time_outliers'))
        self.assertTrue(hasattr(collected, 'filter_xr_signal'))
        self.assertTrue(hasattr(collected, 'calc_xr_crosscorr'))

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

    def test_calc_xr_crosscorr_full_matches_numpy_reference(self):
        tt = np.arange(0.0, 0.5, 0.1)
        x1 = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        x2 = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        X1 = xr.DataArray(x1, dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(x2, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X2)

        sample_lags = np.arange(-(len(tt) - 1), len(tt))
        expected = _reference_crosscorr(x1, x2, sample_lags)
        np.testing.assert_allclose(out.values, expected)
        np.testing.assert_allclose(out.coords['lag'].values, sample_lags * (tt[1] - tt[0]))

    def test_calc_xr_crosscorr_lag_window_avoids_full_range(self):
        tt = np.arange(0.0, 0.6, 0.1)
        x1 = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
        x2 = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 0.0])
        X1 = xr.DataArray(x1, dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(x2, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X2, lag_window=(-0.15, 0.25))

        sample_lags = np.array([-1, 0, 1, 2])
        expected = _reference_crosscorr(x1, x2, sample_lags)
        np.testing.assert_allclose(out.values, expected)
        np.testing.assert_allclose(out.coords['lag'].values, sample_lags * 0.1)

    def test_calc_xr_crosscorr_clips_lag_window_to_valid_range(self):
        tt = np.arange(0.0, 0.4, 0.1)
        x1 = np.array([1.0, 2.0, 3.0, 4.0])
        x2 = np.array([4.0, 3.0, 2.0, 1.0])
        X1 = xr.DataArray(x1, dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(x2, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X2, lag_window=(-10.0, 0.15))

        sample_lags = np.array([-3, -2, -1, 0, 1])
        expected = _reference_crosscorr(x1, x2, sample_lags)
        np.testing.assert_allclose(out.values, expected)
        np.testing.assert_allclose(out.coords['lag'].values, sample_lags * 0.1)

    def test_calc_xr_crosscorr_subtract_mean_matches_reference(self):
        tt = np.arange(0.0, 0.5, 0.1)
        x1 = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
        x2 = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        X1 = xr.DataArray(x1, dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(x2, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X2, subtract_mean=True, lag_window=(-0.2, 0.2))

        sample_lags = np.arange(-2, 3)
        expected = _reference_crosscorr(x1, x2, sample_lags, subtract_mean=True)
        np.testing.assert_allclose(out.values, expected)

    def test_calc_xr_crosscorr_normalize_returns_coefficient_like_output(self):
        tt = np.arange(0.0, 0.4, 0.1)
        x1 = np.array([1.0, 2.0, 3.0, 4.0])
        X1 = xr.DataArray(x1, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X1, normalize=True, lag_window=(0.0, 0.0))

        np.testing.assert_allclose(out.values, np.array([1.0]))

    def test_calc_xr_crosscorr_broadcasts_over_non_time_dims(self):
        tt = np.arange(0.0, 0.5, 0.1)
        data = np.vstack([
            np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
            np.array([1.0, 0.0, -1.0, 0.0, 1.0]),
        ])
        ref = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        X1 = xr.DataArray(data, dims=['pop', 'time'], coords={'pop': ['a', 'b'], 'time': tt})
        X2 = xr.DataArray(ref, dims=['time'], coords={'time': tt})

        out = collected.calc_xr_crosscorr(X1, X2, lag_window=(-0.1, 0.1))

        self.assertEqual(out.dims, ('pop', 'lag'))
        expected_a = _reference_crosscorr(data[0], ref, np.array([-1, 0, 1]))
        expected_b = _reference_crosscorr(data[1], ref, np.array([-1, 0, 1]))
        np.testing.assert_allclose(out.sel(pop='a').values, expected_a)
        np.testing.assert_allclose(out.sel(pop='b').values, expected_b)

    def test_calc_xr_crosscorr_invalid_time_position_raises(self):
        tt = np.arange(0.0, 0.5, 0.1)
        X1 = xr.DataArray(np.ones((5, 2)), dims=['time', 'chan'], coords={'time': tt, 'chan': ['a', 'b']})
        X2 = xr.DataArray(np.ones(5), dims=['time'], coords={'time': tt})
        with self.assertRaises(ValueError):
            collected.calc_xr_crosscorr(X1, X2)

    def test_calc_xr_crosscorr_non_monotonic_time_raises(self):
        tt = np.array([0.0, 0.1, 0.3, 0.2, 0.4])
        X1 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        with self.assertRaises(ValueError):
            collected.calc_xr_crosscorr(X1, X2)

    def test_calc_xr_crosscorr_irregular_time_raises(self):
        tt = np.array([0.0, 0.1, 0.21, 0.31, 0.41])
        X1 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        with self.assertRaises(ValueError):
            collected.calc_xr_crosscorr(X1, X2)

    def test_calc_xr_crosscorr_mismatched_time_coords_raises(self):
        tt1 = np.arange(0.0, 0.5, 0.1)
        tt2 = np.arange(0.05, 0.55, 0.1)
        X1 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt1})
        X2 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt2})
        with self.assertRaises(ValueError):
            collected.calc_xr_crosscorr(X1, X2)

    def test_calc_xr_crosscorr_invalid_lag_window_raises(self):
        tt = np.arange(0.0, 0.5, 0.1)
        X1 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        X2 = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        with self.assertRaises(ValueError):
            collected.calc_xr_crosscorr(X1, X2, lag_window=(0.2, -0.1))

    def test_calc_xr_crosscorr_compute_false_leaves_non_dask_input_realized(self):
        tt = np.arange(0.0, 0.5, 0.1)
        X = xr.DataArray(np.arange(5, dtype=float), dims=['time'], coords={'time': tt})
        out = collected.calc_xr_crosscorr(X, X, compute=False, lag_window=(-0.1, 0.1))
        self.assertFalse(hasattr(out.data, 'compute'))

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_calc_xr_crosscorr_compute_false_preserves_deferred_behavior(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt}).chunk({'time': 50})
        out = collected.calc_xr_crosscorr(X, X, compute=False, lag_window=(-0.05, 0.05))
        self.assertTrue(hasattr(out.data, 'compute'))

    @unittest.skipIf(da is None, 'dask is not installed')
    def test_calc_xr_crosscorr_compute_true_returns_realized_result(self):
        tt = np.arange(0.0, 1.0, 0.01)
        X = xr.DataArray(np.sin(2 * np.pi * 10 * tt), dims=['time'], coords={'time': tt}).chunk({'time': 50})
        out = collected.calc_xr_crosscorr(X, X, compute=True, lag_window=(-0.05, 0.05))
        self.assertFalse(hasattr(out.data, 'chunks'))

    def test_calc_xr_crosscorr_store_proc_info_true_writes_proc_steps(self):
        tt = np.arange(0.0, 0.5, 0.1)
        X = xr.DataArray(
            np.arange(5, dtype=float),
            dims=['time'],
            coords={'time': tt},
            attrs={'source': 'test'},
        )
        out = collected.calc_xr_crosscorr(
            X,
            X,
            lag_window=(-0.1, 0.1),
            subtract_mean=True,
            normalize=True,
            store_proc_info=True,
        )
        self.assertEqual(out.attrs['source'], 'test')
        self.assertEqual(out.attrs['proc_steps'][-1]['name'], 'calc_xr_crosscorr')
        self.assertEqual(out.attrs['proc_steps'][-1]['params']['lag_window'], [-0.1, 0.1])
        self.assertTrue(out.attrs['proc_steps'][-1]['params']['subtract_mean'])
        self.assertTrue(out.attrs['proc_steps'][-1]['params']['normalize'])


if __name__ == '__main__':
    unittest.main()
