import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer.xr_io import load_xr, save_xr


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRateCrosscorrHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.module = _load_module(
            'rate_crosscorr_analysis',
            str(repo_root / 'sim_data_analyzer' / 'dev_scratch' / 'analysis' / 'rate_crosscorr.py'),
        )

    def test_resolve_analysis_pop_names_filters_frozen_pops(self):
        pop_names = ['IT2', 'IT3_frz', 'ITP4', 'SOM5A']
        actual = self.module._resolve_analysis_pop_names(pop_names)
        self.assertEqual(actual, ['IT2', 'ITP4', 'SOM5A'])

    def test_resolve_analysis_pop_names_respects_allowlist_order(self):
        pop_names = ['IT2', 'ITP4', 'SOM5A']
        actual = self.module._resolve_analysis_pop_names(pop_names, ['SOM5A', 'IT2'])
        self.assertEqual(actual, ['SOM5A', 'IT2'])

    def test_resolve_analysis_pop_names_rejects_unknown_allowlist_items(self):
        pop_names = ['IT2', 'IT3_frz', 'ITP4']
        with self.assertRaisesRegex(ValueError, 'Requested populations'):
            self.module._resolve_analysis_pop_names(pop_names, ['IT2', 'IT3_frz'])

    def test_get_filter_tag_formats_filtered_and_unfiltered_runs(self):
        self.assertEqual(self.module._get_filter_tag(None), 'nofilt')
        self.assertEqual(self.module._get_filter_tag((4.0, 12.5)), 'bp_4_12p5')

    def test_get_output_dir_uses_analysis_label_and_filter_tag(self):
        actual = self.module._get_output_dir(
            Path('/tmp/results'),
            'exp1',
            'rate_crosscorr',
            (8.0, 30.0),
        )
        self.assertEqual(actual, Path('/tmp/results/exp1/rate_crosscorr__bp_8_30'))

    def test_extract_peak_metrics_uses_largest_absolute_peak(self):
        corr = xr.DataArray(
            np.array([0.2, -0.7, 0.5]),
            dims=['lag'],
            coords={'lag': np.array([-0.1, 0.0, 0.1])},
        )
        peak_val, peak_lag = self.module._extract_peak_metrics(corr)
        self.assertAlmostEqual(peak_val, -0.7)
        self.assertAlmostEqual(peak_lag, 0.0)

    def test_fill_symmetric_pair_populates_both_halves(self):
        pop_names = ['IT2', 'ITP4', 'SOM5A']
        table = self.module._init_metric_table(pop_names)
        pop_index = {pop_name: idx for idx, pop_name in enumerate(pop_names)}
        self.module._fill_symmetric_pair(table, pop_index, 'IT2', 'SOM5A', 0.25)
        self.assertAlmostEqual(table[0, 2], 0.25)
        self.assertAlmostEqual(table[2, 0], 0.25)
        self.assertTrue(np.isnan(table[1, 1]))

    def test_normalize_round_digits_accepts_none_and_nonnegative_int(self):
        self.assertIsNone(self.module._normalize_round_digits(None))
        self.assertEqual(self.module._normalize_round_digits(3), 3)

    def test_normalize_round_digits_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'CSV_ROUND_DIGITS'):
            self.module._normalize_round_digits(-1)
        with self.assertRaisesRegex(ValueError, 'CSV_ROUND_DIGITS'):
            self.module._normalize_round_digits(1.5)

    def test_normalize_matrix_threshold_accepts_none_and_unit_interval(self):
        self.assertIsNone(self.module._normalize_matrix_threshold(None))
        self.assertAlmostEqual(self.module._normalize_matrix_threshold(0.25), 0.25)

    def test_normalize_matrix_threshold_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'MATRIX_THRESHOLD'):
            self.module._normalize_matrix_threshold(-0.1)
        with self.assertRaisesRegex(ValueError, 'MATRIX_THRESHOLD'):
            self.module._normalize_matrix_threshold(1.1)

    def test_normalize_plot_amp_threshold_accepts_none_and_unit_interval(self):
        self.assertIsNone(self.module._normalize_plot_amp_threshold(None))
        self.assertAlmostEqual(self.module._normalize_plot_amp_threshold(0.25), 0.25)

    def test_normalize_plot_amp_threshold_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'PLOT_AMP_THRESHOLD'):
            self.module._normalize_plot_amp_threshold(-0.1)
        with self.assertRaisesRegex(ValueError, 'PLOT_AMP_THRESHOLD'):
            self.module._normalize_plot_amp_threshold(1.1)

    def test_csv_cell_applies_fixed_decimal_rounding(self):
        self.assertEqual(self.module._csv_cell(0.12345, round_digits=3), '0.123')
        self.assertEqual(self.module._csv_cell(-0.2, round_digits=3), '-0.200')

    def test_get_matrix_png_name_adds_threshold_tag_when_present(self):
        self.assertEqual(
            self.module._get_matrix_png_name('rate_crosscorr', None, masked=False),
            'rate_crosscorr__matrices.png',
        )
        self.assertEqual(
            self.module._get_matrix_png_name('rate_crosscorr', 0.25, masked=True),
            'rate_crosscorr__matrices__thr_0p25.png',
        )

    def test_get_matrix_png_names_returns_plain_and_masked_versions(self):
        self.assertEqual(
            self.module._get_matrix_png_names('rate_crosscorr', None),
            ['rate_crosscorr__matrices.png'],
        )
        self.assertEqual(
            self.module._get_matrix_png_names('rate_crosscorr', 0.25),
            ['rate_crosscorr__matrices.png', 'rate_crosscorr__matrices__thr_0p25.png'],
        )

    def test_get_pair_png_dirname_uses_threshold_tag_when_present(self):
        self.assertEqual(self.module._get_pair_png_dirname(None), 'pair_pngs')
        self.assertEqual(self.module._get_pair_png_dirname(0.25), 'pair_pngs__thr_0p25')

    def test_get_crosscorr_cache_dir_and_path_group_under_proc_subfolder(self):
        dirpath_proc = Path('/tmp/proc/exp1')
        cache_dir = self.module._get_crosscorr_cache_dir(dirpath_proc)
        cache_path = self.module._get_crosscorr_cache_path(
            dirpath_proc,
            'rate_crosscorr',
            ['IT2', 'ITP4'],
            0.005,
            (10.0, 30.0),
            (-0.5, 0.5),
            (8.0, 14.0),
        )
        self.assertEqual(cache_dir, Path('/tmp/proc/exp1/crosscorr_cache'))
        self.assertTrue(str(cache_path).startswith('/tmp/proc/exp1/crosscorr_cache/'))
        self.assertIn('rate_crosscorr', cache_path.name)
        self.assertIn('bp_8_14', cache_path.name)
        self.assertTrue(cache_path.name.endswith('.nc'))

    def test_pair_passes_plot_threshold_uses_absolute_amplitude(self):
        self.assertTrue(self.module._pair_passes_plot_threshold(-0.3, 0.25))
        self.assertFalse(self.module._pair_passes_plot_threshold(0.2, 0.25))
        self.assertTrue(self.module._pair_passes_plot_threshold(0.2, None))

    def test_pair_is_self_detects_self_and_cross_pairs(self):
        corr_ds = xr.Dataset(
            data_vars={
                'normalized_corr': (['pair', 'lag'], np.array([[1.0], [0.4]])),
            },
            coords={
                'pair': ['IT2__IT2', 'IT2__ITP4'],
                'lag': np.array([0.0]),
                'pop_i': ('pair', ['IT2', 'IT2']),
                'pop_j': ('pair', ['IT2', 'ITP4']),
            },
        )
        self.assertTrue(self.module._pair_is_self('IT2__IT2', corr_ds))
        self.assertFalse(self.module._pair_is_self('IT2__ITP4', corr_ds))

    def test_write_metric_csv_writes_square_table_with_labels(self):
        pop_names = ['IT2', 'ITP4']
        table = np.array([[1.0, -0.2], [-0.2, 0.5]])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_csv = Path(tmpdir) / 'amp.csv'
            self.module._write_metric_csv(fpath_csv, pop_names, table)
            actual = fpath_csv.read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(actual[0], 'pop,IT2,ITP4')
        self.assertEqual(actual[1], 'IT2,1,-0.2')
        self.assertEqual(actual[2], 'ITP4,-0.2,0.5')

    def test_write_metric_csv_respects_round_digits(self):
        pop_names = ['IT2', 'ITP4']
        table = np.array([[1.0, -0.23456], [-0.23456, 0.5]])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_csv = Path(tmpdir) / 'amp.csv'
            self.module._write_metric_csv(fpath_csv, pop_names, table, round_digits=2)
            actual = fpath_csv.read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(actual[1], 'IT2,1.00,-0.23')
        self.assertEqual(actual[2], 'ITP4,-0.23,0.50')

    def test_make_matrix_plot_writes_png(self):
        pop_names = ['IT2', 'ITP4']
        amp_table = np.array([[1.0, -0.2], [-0.2, 1.0]])
        lag_table = np.array([[0.0, 0.05], [0.05, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_png = Path(tmpdir) / 'matrices.png'
            self.module._make_matrix_plot(
                fpath_png,
                pop_names,
                amp_table,
                lag_table,
                analysis_label='rate_crosscorr',
                filter_fband=(8.0, 14.0),
                matrix_threshold=None,
                use_mask=False,
            )
            self.assertTrue(fpath_png.exists())
            self.assertGreater(fpath_png.stat().st_size, 0)

    def test_make_matrix_plot_writes_masked_png(self):
        pop_names = ['IT2', 'ITP4']
        amp_table = np.array([[1.0, 0.2], [0.2, 1.0]])
        lag_table = np.array([[0.0, 0.05], [0.05, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_png = Path(tmpdir) / 'matrices_masked.png'
            self.module._make_matrix_plot(
                fpath_png,
                pop_names,
                amp_table,
                lag_table,
                analysis_label='rate_crosscorr',
                filter_fband=(8.0, 14.0),
                matrix_threshold=0.3,
                use_mask=True,
            )
            self.assertTrue(fpath_png.exists())
            self.assertGreater(fpath_png.stat().st_size, 0)

    def test_prepare_matrix_tables_for_plot_returns_weak_mask(self):
        amp_table = np.array([[1.0, 0.2], [0.2, 0.9]])
        lag_table = np.array([[0.0, 0.05], [0.05, 0.0]])
        amp_plot, lag_plot, weak_mask = self.module._prepare_matrix_tables_for_plot(
            amp_table,
            lag_table,
            matrix_threshold=0.3,
        )
        self.assertAlmostEqual(amp_plot[0, 1], 0.2)
        self.assertAlmostEqual(amp_plot[1, 0], 0.2)
        self.assertAlmostEqual(lag_plot[0, 1], 0.05)
        self.assertAlmostEqual(lag_plot[1, 0], 0.05)
        self.assertAlmostEqual(amp_plot[0, 0], 1.0)
        np.testing.assert_array_equal(
            weak_mask,
            np.array([[False, True], [True, False]]),
        )

    def test_get_symmetric_plot_limit_uses_observed_abs_max(self):
        values = np.array([[0.0, -0.03], [0.01, 0.02]])
        actual = self.module._get_symmetric_plot_limit(values, fallback=0.5)
        self.assertAlmostEqual(actual, 0.03)

    def test_get_symmetric_plot_limit_falls_back_for_zero_or_nan(self):
        self.assertAlmostEqual(
            self.module._get_symmetric_plot_limit(np.array([[0.0, 0.0]]), fallback=0.5),
            0.5,
        )
        self.assertAlmostEqual(
            self.module._get_symmetric_plot_limit(np.array([[np.nan]]), fallback=0.5),
            0.5,
        )

    def test_get_lag_plot_limit_ignores_masked_cells_in_masked_view(self):
        lag_plot = np.array([[0.02, 0.4], [0.03, -0.01]])
        weak_mask = np.array([[False, True], [False, False]])
        actual = self.module._get_lag_plot_limit(
            lag_plot,
            weak_mask,
            use_mask=True,
            fallback=0.5,
        )
        self.assertAlmostEqual(actual, 0.03)

    def test_get_lag_plot_limit_uses_all_cells_in_plain_view(self):
        lag_plot = np.array([[0.02, 0.4], [0.03, -0.01]])
        weak_mask = np.array([[False, True], [False, False]])
        actual = self.module._get_lag_plot_limit(
            lag_plot,
            weak_mask,
            use_mask=False,
            fallback=0.5,
        )
        self.assertAlmostEqual(actual, 0.4)

    def test_crosscorr_cache_dataset_round_trip_preserves_dataset_shape(self):
        corr_ds = xr.Dataset(
            data_vars={
                'raw_corr': (['pair', 'lag'], np.array([[1.0, 0.5], [0.2, 0.1]])),
                'demeaned_corr': (['pair', 'lag'], np.array([[0.7, 0.2], [0.1, 0.0]])),
                'normalized_corr': (['pair', 'lag'], np.array([[1.0, 0.3], [0.4, 0.2]])),
            },
            coords={
                'pair': ['IT2__IT2', 'IT2__ITP4'],
                'lag': np.array([0.0, 0.1]),
                'pop_i': ('pair', ['IT2', 'IT2']),
                'pop_j': ('pair', ['IT2', 'ITP4']),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_cache = Path(tmpdir) / 'crosscorr_cache' / 'cache.nc'
            save_xr(corr_ds, fpath_cache)
            loaded = load_xr(fpath_cache, data_type='dataset', load=True)
        xr.testing.assert_identical(loaded, corr_ds)


if __name__ == '__main__':
    unittest.main()
