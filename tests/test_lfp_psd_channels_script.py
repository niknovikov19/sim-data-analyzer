import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_signal_y():
    tt = np.arange(0.0, 10.0, 0.1)
    yy = np.array([100.0, 300.0])
    values = np.vstack([
        np.sin(2 * np.pi * 5.0 * tt),
        np.sin(2 * np.pi * 8.0 * tt),
    ])
    return xr.DataArray(
        values,
        dims=['y', 'time'],
        coords={'y': yy, 'time': tt},
        attrs={'source': 'test-signal'},
    )


class TestLfpPsdChannelsScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.collected = _load_module(
            'lfp_psd_channels_script',
            repo_root / 'dev_scratch' / 'analysis' / 'lfp_psd_channels.py',
        )

    def test_results_dirname_includes_params_and_log_flags(self):
        actual = self.collected._get_results_dirname(
            'csd', 4.0, 0.5, 'median', True, False, t_limits_s=(5.0, 30.0)
        )
        self.assertEqual(actual, 'csd__t_5_30__win_4__over_0p5__avg_median__logx_1__logy_0')

    def test_cache_path_includes_signal_type(self):
        lfp_path = self.collected._get_psd_cache_path(
            Path('/tmp/proc'),
            'lfp',
            4.0,
            0.5,
            2.0,
            100.0,
            'median',
            t_limits_s=(5.0, 30.0),
        )
        csd_path = self.collected._get_psd_cache_path(
            Path('/tmp/proc'),
            'csd',
            4.0,
            0.5,
            2.0,
            100.0,
            'median',
            t_limits_s=(5.0, 30.0),
        )
        self.assertNotEqual(lfp_path, csd_path)

    def test_psd_cache_path_is_under_data_proc(self):
        actual = self.collected._get_psd_cache_path(
            Path('/tmp/proc'),
            'csd',
            4.0,
            0.5,
            2.0,
            100.0,
            'median',
            t_limits_s=(5.0, 30.0),
        )
        self.assertEqual(
            actual,
            Path('/tmp/proc') / 'psd_cache' / 'csd__t_5_30__win_4__over_0p5__avg_median__f_2_100.nc',
        )

    def test_load_or_compute_psd_cache_cache_miss_computes_and_saves(self):
        signal = _make_signal_y()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'psd_cache' / 'run1.nc'
            psd, cache_hit = self.collected._load_or_compute_psd_cache(
                signal,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
            )
            self.assertFalse(cache_hit)
            self.assertTrue(cache_path.exists())
            self.assertEqual(psd.dims, ('y', 'freq'))
            self.assertTrue(np.isfinite(psd.values).any())

    def test_load_or_compute_psd_cache_cache_hit_reuses_existing_file(self):
        signal = _make_signal_y()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'psd_cache' / 'run1.nc'
            psd_first, cache_hit_first = self.collected._load_or_compute_psd_cache(
                signal,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
            )
            psd_second, cache_hit_second = self.collected._load_or_compute_psd_cache(
                signal,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
            )
            self.assertFalse(cache_hit_first)
            self.assertTrue(cache_hit_second)
            np.testing.assert_allclose(psd_first.values, psd_second.values)

    def test_plot_channel_psd_writes_png(self):
        signal = _make_signal_y()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'psd_cache' / 'run1.nc'
            psd, _cache_hit = self.collected._load_or_compute_psd_cache(
                signal,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
            )
            fpath_out = Path(tmpdir) / 'psd_y_100.png'
            self.collected._plot_channel_psd(psd, 100.0, fpath_out, signal_type='lfp', logx=True, logy=True)
            self.assertTrue(fpath_out.exists())


if __name__ == '__main__':
    unittest.main()
