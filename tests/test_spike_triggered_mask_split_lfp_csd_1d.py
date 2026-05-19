import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer.spike_data import SpikeData, _SpikeMeta


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_signal_y():
    tt = np.arange(0.0, 1.0, 0.1)
    yy = np.array([100.0, 300.0])
    values = np.vstack([
        np.arange(len(tt), dtype=float),
        np.arange(len(tt), dtype=float) + 100.0,
    ])
    return xr.DataArray(
        values,
        dims=['y', 'time'],
        coords={'y': yy, 'time': tt},
        attrs={'source': 'test-signal'},
    )


def _make_mask_y():
    tt = np.arange(0.0, 1.0, 0.1)
    yy = np.array([100.0, 300.0])
    values = np.zeros((2, len(tt)), dtype=float)
    values[0, 2] = 1.0
    values[1, 5] = 1.0
    return xr.DataArray(
        values,
        dims=['y', 'time'],
        coords={'y': yy, 'time': tt},
        name='mask',
    )


def _make_trigger_spikes(pop_name='IT2'):
    return SpikeData(
        {str(pop_name): [np.array([200.0, 500.0, 800.0])]},
        meta=_SpikeMeta(
            combine=True, t0=0.0, tmax=1000.0, subtract_t0=False, ms=True, ndigits=3
        ),
        pop_sizes={str(pop_name): 1},
    )


class TestSpikeTriggeredMaskSplitLFPCSDScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.collected = _load_module(
            'spike_triggered_mask_split_lfp_csd_1d_script',
            repo_root / 'dev_scratch' / 'analysis' / 'spike_triggered_mask_split_lfp_csd_1d.py',
        )

    def test_split_raw_spikes_by_mask_uses_same_channel_mask(self):
        mask = _make_mask_y().sel(y=100.0)
        split = self.collected._split_raw_spikes_by_mask(mask, np.array([200.0, 500.0, 800.0]), 1e-3)
        np.testing.assert_allclose(split['mask1'], np.array([200.0]))
        np.testing.assert_allclose(split['mask0'], np.array([500.0, 800.0]))

    def test_build_mask_split_sta_dataset_shared_csd_keeps_common_signal_offset(self):
        signal = _make_signal_y()
        signal = signal - signal.mean(dim='time')
        mask = _make_mask_y()
        spikes = _make_trigger_spikes('IT2')
        ds = self.collected._build_mask_split_sta_dataset(
            signal,
            mask,
            spikes,
            'IT2',
            (-0.1, 0.2),
            'csd',
            'shared',
        )
        row_mask1 = ds['sta_avg'].sel(state='mask1', y=100.0).values
        row_mask0 = ds['sta_avg'].sel(state='mask0', y=100.0).values
        np.testing.assert_allclose(row_mask1, np.array([-3.5, -2.5, -1.5, -0.5]))
        np.testing.assert_allclose(row_mask0, np.array([-0.5, 0.5, 1.5, 2.5]))

    def test_build_mask_split_sta_dataset_separate_csd_centers_each_state(self):
        signal = _make_signal_y()
        mask = _make_mask_y()
        spikes = _make_trigger_spikes('IT2')
        ds = self.collected._build_mask_split_sta_dataset(
            signal,
            mask,
            spikes,
            'IT2',
            (-0.1, 0.2),
            'csd',
            'separate',
        )
        row_mask1 = ds['sta_avg'].sel(state='mask1', y=100.0).values
        row_mask0 = ds['sta_avg'].sel(state='mask0', y=100.0).values
        np.testing.assert_allclose(row_mask1, np.array([-1.5, -0.5, 0.5, 1.5]))
        np.testing.assert_allclose(row_mask0, np.array([-1.5, -0.5, 0.5, 1.5]))

    def test_load_or_compute_sta_cache_cache_miss_and_hit(self):
        signal = _make_signal_y()
        mask = _make_mask_y()
        spikes = _make_trigger_spikes('IT2')
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'proc' / 'cache1'
            ds_first, cache_path, hit_first = self.collected._load_or_compute_sta_cache(
                signal,
                mask,
                spikes,
                'IT2',
                'lfp',
                (-100.0, 200.0),
                'shared',
                cache_dir,
            )
            ds_second, cache_path_second, hit_second = self.collected._load_or_compute_sta_cache(
                signal,
                mask,
                spikes,
                'IT2',
                'lfp',
                (-100.0, 200.0),
                'shared',
                cache_dir,
            )
            self.assertFalse(hit_first)
            self.assertTrue(hit_second)
            self.assertEqual(cache_path, cache_path_second)
            self.assertTrue(cache_path.exists())
            np.testing.assert_allclose(ds_first['sta_avg'].values, ds_second['sta_avg'].values)

    def test_plot_mask_split_sta_1d_writes_png(self):
        signal = _make_signal_y()
        mask = _make_mask_y()
        spikes = _make_trigger_spikes('IT2')
        ds = self.collected._build_mask_split_sta_dataset(
            signal,
            mask,
            spikes,
            'IT2',
            (-0.1, 0.2),
            'lfp',
            'shared',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_out = Path(tmpdir) / 'sta_masksplit_y_100.png'
            self.collected._plot_mask_split_sta_1d(
                ds,
                100.0,
                fpath_out,
                'IT2',
                'lfp',
                show_zero_line=True,
                zero_line_alpha=0.3,
            )
            self.assertTrue(fpath_out.exists())


if __name__ == '__main__':
    unittest.main()
