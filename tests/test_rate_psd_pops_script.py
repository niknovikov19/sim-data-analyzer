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


def _make_rates():
    tt = np.arange(0.0, 10.0, 0.05)
    pops = np.array(['IT2', 'PV3', 'IT3_frz'])
    values = np.vstack([
        np.sin(2 * np.pi * 6.0 * tt),
        np.sin(2 * np.pi * 10.0 * tt),
        np.sin(2 * np.pi * 4.0 * tt),
    ])
    return xr.DataArray(
        values,
        dims=['pop', 'time'],
        coords={'pop': pops, 'time': tt},
        attrs={'source': 'test-rates'},
    )


class TestRatePsdPopsScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.collected = _load_module(
            'rate_psd_pops_script',
            repo_root / 'dev_scratch' / 'analysis' / 'rate_psd_pops.py',
        )

    def test_resolve_pop_names_filters_frozen_pops(self):
        actual = self.collected._resolve_pop_names(['IT2', 'IT3_frz', 'PV3'], requested_pop_names='all')
        self.assertEqual(actual, ['IT2', 'PV3'])

    def test_resolve_pop_groups_keeps_only_present_members(self):
        actual = self.collected._resolve_pop_groups(
            ['IT2', 'PV3', 'TC'],
            pop_groups={
                'IT': ['IT2', 'IT3'],
                'PV': ['PV3', 'PV4'],
                'THAL': ['TC', 'IRE'],
            },
        )
        self.assertEqual(actual, {'IT': ['IT2'], 'PV': ['PV3'], 'THAL': ['TC']})

    def test_results_dirname_includes_rate_params_and_log_flags(self):
        actual = self.collected._get_results_dirname(
            5e-3,
            4.0,
            0.75,
            'median',
            True,
            (5.0, 30.0),
            3,
            True,
            False,
            t_limits_s=None,
            plot_f_limits=(2.0, 50.0),
        )
        self.assertEqual(
            actual,
            't_full__dt_0p005__win_4__over_0p75__avg_median__norm_1__smooth_3__nband_5_30__plotf_2_50__logx_1__logy_0',
        )

    def test_psd_cache_path_is_under_data_proc(self):
        actual = self.collected._get_psd_cache_path(
            Path('/tmp/proc'),
            5e-3,
            4.0,
            0.75,
            2.0,
            30.0,
            'median',
            True,
            (5.0, 30.0),
            3,
            t_limits_s=None,
            plot_f_limits=(2.0, 50.0),
        )
        self.assertEqual(
            actual,
            Path('/tmp/proc') / 'rate_psd_cache' / 't_full__dt_0p005__win_4__over_0p75__avg_median__norm_1__smooth_3__nband_5_30__plotf_2_50__f_2_30.nc',
        )

    def test_postprocess_rate_psd_normalizes_and_smooths(self):
        psd = xr.DataArray(
            np.array([
                [2.0, 4.0, 6.0, 8.0],
                [1.0, 3.0, 5.0, 7.0],
            ]),
            dims=['pop', 'freq'],
            coords={'pop': ['IT2', 'PV3'], 'freq': [5.0, 10.0, 20.0, 30.0]},
        )
        actual = self.collected._postprocess_rate_psd(
            psd,
            normalize=True,
            normalize_f_band=(5.0, 30.0),
            smooth_freq_bins=3,
            plot_f_limits=(5.0, 20.0),
        )
        self.assertEqual(actual.dims, ('pop', 'freq'))
        self.assertEqual(actual.sizes['freq'], 3)
        self.assertTrue(np.isfinite(actual.values).all())

    def test_load_or_compute_psd_cache_cache_miss_computes_and_saves(self):
        rates = _make_rates().sel(pop=['IT2', 'PV3'])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'rate_psd_cache' / 'run1.nc'
            psd, cache_hit = self.collected._load_or_compute_psd_cache(
                rates,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
                normalize=True,
                normalize_f_band=(5.0, 20.0),
                smooth_freq_bins=3,
                plot_f_limits=(2.0, 20.0),
            )
            self.assertFalse(cache_hit)
            self.assertTrue(cache_path.exists())
            self.assertEqual(psd.dims, ('pop', 'freq'))
            self.assertTrue(np.isfinite(psd.values).any())

    def test_load_or_compute_psd_cache_cache_hit_reuses_existing_file(self):
        rates = _make_rates().sel(pop=['IT2', 'PV3'])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'rate_psd_cache' / 'run1.nc'
            psd_first, cache_hit_first = self.collected._load_or_compute_psd_cache(
                rates,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
                normalize=True,
                normalize_f_band=(5.0, 20.0),
                smooth_freq_bins=3,
                plot_f_limits=(2.0, 20.0),
            )
            psd_second, cache_hit_second = self.collected._load_or_compute_psd_cache(
                rates,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
                normalize=True,
                normalize_f_band=(5.0, 20.0),
                smooth_freq_bins=3,
                plot_f_limits=(2.0, 20.0),
            )
            self.assertFalse(cache_hit_first)
            self.assertTrue(cache_hit_second)
            np.testing.assert_allclose(psd_first.values, psd_second.values)

    def test_plot_pop_psd_writes_png(self):
        rates = _make_rates().sel(pop=['IT2', 'PV3'])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'rate_psd_cache' / 'run1.nc'
            psd, _cache_hit = self.collected._load_or_compute_psd_cache(
                rates,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
                normalize=True,
                normalize_f_band=(5.0, 20.0),
                smooth_freq_bins=3,
                plot_f_limits=(2.0, 20.0),
            )
            fpath_out = Path(tmpdir) / 'psd_IT2.png'
            self.collected._plot_pop_psd(psd, 'IT2', fpath_out, logx=True, logy=True)
            self.assertTrue(fpath_out.exists())

    def test_plot_group_psd_writes_png(self):
        rates = _make_rates().sel(pop=['IT2', 'PV3'])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'proc' / 'rate_psd_cache' / 'run1.nc'
            psd, _cache_hit = self.collected._load_or_compute_psd_cache(
                rates,
                cache_path,
                win_len=2.0,
                win_overlap=0.5,
                fmin=2.0,
                fmax=20.0,
                average='median',
                normalize=True,
                normalize_f_band=(5.0, 20.0),
                smooth_freq_bins=3,
                plot_f_limits=(2.0, 20.0),
            )
            fpath_out = Path(tmpdir) / 'psd_group_test.png'
            self.collected._plot_group_psd(psd, 'test', ['IT2', 'PV3'], fpath_out, logx=True, logy=True)
            self.assertTrue(fpath_out.exists())


if __name__ == '__main__':
    unittest.main()
