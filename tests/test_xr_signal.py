import unittest

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_signal as collected


class TestCollectedXRSignal(unittest.TestCase):
    def test_smoke_exports(self):
        self.assertTrue(hasattr(collected, 'interp_time_outliers'))

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


if __name__ == '__main__':
    unittest.main()
