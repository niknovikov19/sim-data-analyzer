import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer.spike_data import SpikeData, _SpikeMeta
from sim_data_analyzer.xr_io import load_xr


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


def _make_combined_spikes():
    return SpikeData(
        {'IT2': [np.array([200.0, 500.0, 800.0])]},
        meta=_SpikeMeta(
            combine=True, t0=0.0, tmax=1000.0, subtract_t0=True, ms=True, ndigits=3
        ),
        pop_sizes={'IT2': 1},
    )


def _make_layer_config():
    return {
        'y_size_um': 1000.0,
        'layers': [
            {'name': 'L2', 'y_norm': [0.0, 0.2]},
            {'name': 'L3', 'y_norm': [0.2, 0.5]},
            {'name': 'L4', 'y_norm': [0.5, 0.7]},
            {'name': 'THAL', 'y_norm': [1.2, 1.4]},
        ],
    }


class TestSpikeTriggeredLFPCSDScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.collected = _load_module(
            'spike_triggered_lfp_csd_script',
            repo_root / 'dev_scratch' / 'analysis' / 'spike_triggered_lfp_csd.py',
        )

    def test_output_dirname_rounds_ms_window(self):
        actual = self.collected._get_sta_output_dirname('ITS4', 'csd', (-99.6, 100.4))
        self.assertEqual(actual, 'ITS4_csd_-100_100')

    def test_resolve_plot_depths_defaults_to_all_channels(self):
        resolved = self.collected._resolve_plot_depths(np.array([100.0, 300.0, 600.0]), None)
        self.assertEqual(resolved, [100.0, 300.0, 600.0])

    def test_resolve_plot_depths_maps_to_nearest_unique_channels(self):
        resolved = self.collected._resolve_plot_depths(
            np.array([100.0, 300.0, 600.0]),
            selected_y=[280.0, 590.0, 310.0],
        )
        self.assertEqual(resolved, [300.0, 600.0])

    def test_maybe_subtract_channel_mean_operates_per_channel(self):
        signal = xr.DataArray(
            np.array([
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
            ]),
            dims=['y', 'time'],
            coords={'y': [100.0, 300.0], 'time': [0.0, 0.1, 0.2]},
            attrs={'source': 'test-signal'},
        )
        centered = self.collected._maybe_subtract_channel_mean(signal, enabled=True)
        np.testing.assert_allclose(centered.mean(dim='time').values, np.array([0.0, 0.0]))
        self.assertTrue(centered.attrs['channel_mean_subtracted'])
        self.assertEqual(centered.attrs['channel_mean_subtract_time_dim'], 'time')

    def test_encode_dataset_attr_converts_bool_to_int(self):
        self.assertEqual(self.collected._encode_dataset_attr(True), 1)
        self.assertEqual(self.collected._encode_dataset_attr(False), 0)

    def test_decode_int_list_attr_accepts_json_text_from_cache(self):
        decoded = self.collected._decode_int_list_attr('[1, 2, 3]')
        self.assertEqual(decoded, [1, 2, 3])

    def test_get_layer_spans_um_converts_norm_to_absolute_depths(self):
        spans = self.collected._get_layer_spans_um(_make_layer_config())
        self.assertEqual(
            spans,
            [
                {'name': 'L2', 'y0_um': 0.0, 'y1_um': 200.0},
                {'name': 'L3', 'y0_um': 200.0, 'y1_um': 500.0},
                {'name': 'L4', 'y0_um': 500.0, 'y1_um': 700.0},
                {'name': 'THAL', 'y0_um': 1200.0, 'y1_um': 1400.0},
            ],
        )

    def test_get_visible_layer_spans_filters_to_plotted_range(self):
        spans = self.collected._get_layer_spans_um(_make_layer_config())
        visible = self.collected._get_visible_layer_spans(
            spans,
            np.array([0.0, 100.0, 300.0, 600.0]),
        )
        self.assertEqual(
            visible,
            [
                {'name': 'L2', 'y0_um': 0.0, 'y1_um': 200.0},
                {'name': 'L3', 'y0_um': 200.0, 'y1_um': 500.0},
                {'name': 'L4', 'y0_um': 500.0, 'y1_um': 700.0},
            ],
        )

    def test_build_sta_avg_2d_streams_one_row_per_channel(self):
        signal = _make_signal_y()
        spikes = _make_combined_spikes()
        avg_sta = self.collected._build_sta_avg_2d(signal, spikes, 'IT2', (-0.1, 0.2))
        self.assertEqual(avg_sta.dims, ('y', 'time_rel'))
        np.testing.assert_allclose(avg_sta.coords['y'].values, np.array([100.0, 300.0]))
        np.testing.assert_allclose(
            avg_sta.values,
            np.array([
                [2.5, 3.5, 4.5, 5.5],
                [102.5, 103.5, 104.5, 105.5],
            ]),
        )
        self.assertEqual(avg_sta.attrs['n_spikes_used_by_channel'], [2, 2])

    def test_save_sta_avg_cache_writes_only_shared_cache_and_manifest(self):
        avg_sta = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            dims=['y', 'time_rel'],
            coords={'y': [100.0, 300.0], 'time_rel': [-0.1, 0.1]},
            attrs={
                'n_spikes_used_by_channel': [5, 5],
                'outlier_interp': {
                    'name': 'interp_time_outliers',
                    'params': {'time_dim': 'time', 'z_thresh': 8.0},
                },
            },
            name='sta_avg',
        )
        manifest = {
            'analysis': 'spike_triggered_sta',
            'trigger_pop': 'IT2',
            'signal_type': 'lfp',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cache_dir = tmpdir / 'proc' / 'spike_triggered_cache' / 'run1'
            result_dir = tmpdir / 'results' / 'run1'
            cache_nc, result_nc, manifest_path = self.collected._save_sta_avg_cache(
                avg_sta,
                cache_dir,
                result_dir,
                manifest,
            )

            self.assertTrue(cache_nc.exists())
            self.assertTrue(result_nc.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                sorted(path.name for path in cache_dir.iterdir()),
                ['avg_2d.nc', 'manifest.json'],
            )

            loaded = load_xr(cache_nc, load=True)
            self.assertEqual(loaded.dims, avg_sta.dims)
            np.testing.assert_allclose(loaded.coords['y'].values, avg_sta.coords['y'].values)
            np.testing.assert_allclose(
                loaded.coords['time_rel'].values,
                avg_sta.coords['time_rel'].values,
            )
            np.testing.assert_allclose(loaded.values, avg_sta.values)
            self.assertIsInstance(loaded.attrs['outlier_interp'], str)
            self.assertIn('interp_time_outliers', loaded.attrs['outlier_interp'])
            loaded_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(loaded_manifest, manifest)

    def test_is_valid_sta_cache_rejects_all_nan_arrays(self):
        avg_sta = xr.DataArray(
            np.full((2, 3), np.nan, dtype=float),
            dims=['y', 'time_rel'],
            coords={'y': [100.0, 300.0], 'time_rel': [-0.1, 0.0, 0.1]},
        )
        self.assertFalse(self.collected._is_valid_sta_cache(avg_sta))

    def test_show_zero_line_defaults_to_off(self):
        self.assertFalse(self.collected.SHOW_ZERO_LINE)

    def test_load_or_compute_sta_avg_cache_cache_miss_computes_and_saves(self):
        signal = _make_signal_y()
        spikes = _make_combined_spikes()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cache_dir = tmpdir / 'proc' / 'spike_triggered_cache' / 'run1'
            result_dir = tmpdir / 'results' / 'run1'
            avg_sta, cache_nc, result_nc, manifest_path, cache_hit = (
                self.collected._load_or_compute_sta_avg_cache(
                    signal,
                    spikes,
                    'IT2',
                    'lfp',
                    (0.0, 1.0),
                    (-100.0, 200.0),
                    True,
                    cache_dir,
                    result_dir,
                    tmpdir / 'sim.pkl',
                    tmpdir / 'lfp.nc',
                    tmpdir / 'spikes.npz',
                )
            )
            self.assertFalse(cache_hit)
            self.assertTrue(cache_nc.exists())
            self.assertTrue(result_nc.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(avg_sta.dims, ('y', 'time_rel'))
            self.assertTrue(np.isfinite(avg_sta.values).any())

    def test_load_or_compute_sta_avg_cache_cache_hit_reuses_existing_file(self):
        signal = _make_signal_y()
        spikes = _make_combined_spikes()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cache_dir = tmpdir / 'proc' / 'spike_triggered_cache' / 'run1'
            result_dir = tmpdir / 'results' / 'run1'
            cache_dir.mkdir(parents=True, exist_ok=True)
            avg_sta_in = xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                dims=['y', 'time_rel'],
                coords={'y': [100.0, 300.0], 'time_rel': [-0.1, 0.1]},
                attrs={'n_spikes_used_by_channel': [7, 7]},
                name='sta_avg',
            )
            manifest = {
                'analysis': 'spike_triggered_sta',
                'trigger_pop': 'IT2',
                'signal_type': 'lfp',
            }
            self.collected._save_sta_avg_cache(avg_sta_in, cache_dir, result_dir, manifest)

            avg_sta, cache_nc, result_nc, manifest_path, cache_hit = (
                self.collected._load_or_compute_sta_avg_cache(
                    signal,
                    spikes,
                    'IT2',
                    'lfp',
                    (0.0, 1.0),
                    (-100.0, 200.0),
                    True,
                    cache_dir,
                    result_dir,
                    tmpdir / 'sim.pkl',
                    tmpdir / 'lfp.nc',
                    tmpdir / 'spikes.npz',
                )
            )
            self.assertTrue(cache_hit)
            self.assertTrue(cache_nc.exists())
            self.assertTrue(result_nc.exists())
            self.assertTrue(manifest_path.exists())
            np.testing.assert_allclose(avg_sta.values, avg_sta_in.values)


if __name__ == '__main__':
    unittest.main()
