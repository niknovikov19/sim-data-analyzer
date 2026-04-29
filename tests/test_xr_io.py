import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_io as collected


def _make_dataarray():
    tt = np.arange(0.0, 1.0, 0.1)
    yy = np.array([100.0, 300.0])
    data = np.vstack([np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)])
    return xr.DataArray(
        data,
        dims=['y', 'time'],
        coords={'y': yy, 'time': tt},
        attrs={'source': 'test'},
    )


def _make_dataset():
    X = _make_dataarray()
    return xr.Dataset({
        'lfp': X,
        'lfp_abs': abs(X),
    }, attrs={'kind': 'test-dataset'})


class TestCollectedXRIO(unittest.TestCase):
    def test_smoke_exports(self):
        for name in [
                'save_xr_dataarray',
                'load_xr_dataarray',
                'save_xr_dataset',
                'load_xr_dataset',
                'save_xr',
                'load_xr']:
            self.assertTrue(hasattr(collected, name), name)

    def test_save_and_load_dataarray(self):
        X = _make_dataarray()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'nested' / 'signal.nc'
            collected.save_xr_dataarray(X, fpath)
            Y = collected.load_xr_dataarray(fpath, load=True)
            xr.testing.assert_identical(X, Y)

    def test_save_and_load_dataset(self):
        X = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'nested' / 'signal.nc'
            collected.save_xr_dataset(X, fpath)
            Y = collected.load_xr_dataset(fpath, load=True)
            xr.testing.assert_identical(X, Y)

    def test_generic_save_and_load_dataarray(self):
        X = _make_dataarray()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'signal.nc'
            collected.save_xr(X, fpath)
            Y = collected.load_xr(fpath, data_type='dataarray', load=True)
            xr.testing.assert_identical(X, Y)

    def test_generic_save_and_load_dataset(self):
        X = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'signal.nc'
            collected.save_xr(X, fpath)
            Y = collected.load_xr(fpath, data_type='dataset', load=True)
            xr.testing.assert_identical(X, Y)

    def test_invalid_input_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'signal.nc'
            with self.assertRaises(TypeError):
                collected.save_xr(np.arange(3), fpath)

    def test_invalid_data_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'signal.nc'
            with self.assertRaises(ValueError):
                collected.load_xr(fpath, data_type='not-supported')


if __name__ == '__main__':
    unittest.main()
