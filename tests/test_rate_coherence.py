import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_data_analyzer.xr_io import load_xr, save_xr


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRateCoherenceHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.module = _load_module(
            'rate_coherence_analysis',
            str(repo_root / 'sim_data_analyzer' / 'dev_scratch' / 'analysis' / 'rate_coherence.py'),
        )

    def test_get_coherence_cache_dir_and_path_group_under_proc_subfolder(self):
        dirpath_proc = Path('/tmp/proc/exp1')
        cache_dir = self.module._get_coherence_cache_dir(dirpath_proc)
        cache_path = self.module._get_coherence_cache_path(
            dirpath_proc,
            'rate_coherence',
            ['IT2', 'PV3'],
            0.001,
            (10.0, 30.0),
            1.0,
            0.5,
            100.0,
        )
        self.assertEqual(cache_dir, Path('/tmp/proc/exp1/coherence_cache'))
        self.assertTrue(str(cache_path).startswith('/tmp/proc/exp1/coherence_cache/'))
        self.assertIn('rate_coherence', cache_path.name)
        self.assertTrue(cache_path.name.endswith('.nc'))

    def test_get_output_dirname_uses_win_overlap_and_fband(self):
        actual = self.module._get_output_dirname('rate_coher_allpops', 2.0, 0.5, (8.0, 14.0))
        self.assertEqual(actual, 'rate_coher_allpops__win_2_over_0p5_fband_8_14')

    def test_get_output_dir_uses_grouped_rate_coher_root(self):
        actual = self.module._get_output_dir(
            Path('/tmp/results'),
            'exp1',
            'rate_coher',
            'rate_coher_allpops',
            2.0,
            0.5,
            (8.0, 14.0),
        )
        self.assertEqual(
            actual,
            Path('/tmp/results/exp1/rate_coher/rate_coher_allpops__win_2_over_0p5_fband_8_14'),
        )

    def test_normalize_coherence_threshold_accepts_none_and_unit_interval(self):
        self.assertIsNone(self.module._normalize_coherence_threshold(None))
        self.assertAlmostEqual(self.module._normalize_coherence_threshold(0.25), 0.25)

    def test_normalize_coherence_threshold_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'COHERENCE_THRESHOLD'):
            self.module._normalize_coherence_threshold(-0.1)
        with self.assertRaisesRegex(ValueError, 'COHERENCE_THRESHOLD'):
            self.module._normalize_coherence_threshold(1.1)

    def test_get_matrix_png_names_returns_plain_and_masked_versions(self):
        self.assertEqual(
            self.module._get_matrix_png_names(None),
            ['matrices.png'],
        )
        self.assertEqual(
            self.module._get_matrix_png_names(0.4),
            [
                'matrices.png',
                'matrices__thr_0p4.png',
            ],
        )

    def test_compute_band_mean_tables_uses_complex_mean_and_symmetry(self):
        coh_ds = xr.Dataset(
            data_vars={
                'coherence': (
                    ['pair', 'freq'],
                    np.array([
                        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                        [0.1 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j, 0.2 + 0.0j],
                        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                    ], dtype=np.complex128),
                ),
            },
            coords={
                'pair': ['IT2__IT2', 'IT2__PV3', 'PV3__PV3'],
                'freq': np.array([5.0, 10.0, 12.0, 20.0]),
                'pop_i': ('pair', ['IT2', 'IT2', 'PV3']),
                'pop_j': ('pair', ['IT2', 'PV3', 'PV3']),
            },
        )
        old_fband = self.module.FBAND
        self.module.FBAND = (8.0, 14.0)
        try:
            coherence_table, phase_table, complex_by_pair = self.module._compute_band_mean_tables_from_cache(
                coh_ds,
                ['IT2', 'PV3'],
            )
        finally:
            self.module.FBAND = old_fband

        expected_band_mean = 0.5 + 0.5j
        self.assertAlmostEqual(complex_by_pair['IT2__PV3'].real, expected_band_mean.real)
        self.assertAlmostEqual(complex_by_pair['IT2__PV3'].imag, expected_band_mean.imag)
        self.assertAlmostEqual(coherence_table[0, 1], np.abs(expected_band_mean))
        self.assertAlmostEqual(coherence_table[1, 0], np.abs(expected_band_mean))
        self.assertAlmostEqual(phase_table[0, 1], np.pi / 4)
        self.assertAlmostEqual(phase_table[1, 0], -np.pi / 4)
        self.assertAlmostEqual(coherence_table[0, 0], 1.0)
        self.assertAlmostEqual(phase_table[0, 0], 0.0)

    def test_prepare_matrix_tables_for_plot_returns_weak_and_diag_masks(self):
        coherence_table = np.array([[1.0, 0.2], [0.2, 1.0]])
        phase_table = np.array([[0.0, 0.6], [-0.6, 0.0]])
        coherence_plot, phase_plot, weak_mask, diag_mask = self.module._prepare_matrix_tables_for_plot(
            coherence_table,
            phase_table,
            coherence_threshold=0.3,
        )
        np.testing.assert_array_equal(weak_mask, np.array([[False, True], [True, False]]))
        np.testing.assert_array_equal(diag_mask, np.array([[True, False], [False, True]]))
        self.assertAlmostEqual(coherence_plot[0, 1], 0.2)
        self.assertAlmostEqual(phase_plot[1, 0], -0.6)

    def test_get_phase_plot_limit_ignores_masked_values(self):
        phase_plot = np.array([[0.0, 0.1], [-2.0, 0.0]])
        plot_mask = np.array([[True, False], [True, True]])
        actual = self.module._get_phase_plot_limit(phase_plot, plot_mask, fallback=np.pi)
        self.assertAlmostEqual(actual, 0.1)

    def test_make_matrix_plot_writes_plain_and_masked_pngs(self):
        pop_names = ['IT2', 'PV3']
        coherence_table = np.array([[1.0, 0.4], [0.4, 1.0]])
        phase_table = np.array([[0.0, 0.5], [-0.5, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            plain_png = Path(tmpdir) / 'plain.png'
            masked_png = Path(tmpdir) / 'masked.png'
            self.module._make_matrix_plot(
                plain_png,
                pop_names,
                coherence_table,
                phase_table,
                analysis_label='rate_coherence',
                fband=(8.0, 14.0),
                coherence_threshold=0.5,
                use_mask=False,
            )
            self.module._make_matrix_plot(
                masked_png,
                pop_names,
                coherence_table,
                phase_table,
                analysis_label='rate_coherence',
                fband=(8.0, 14.0),
                coherence_threshold=0.5,
                use_mask=True,
            )
            self.assertTrue(plain_png.exists())
            self.assertGreater(plain_png.stat().st_size, 0)
            self.assertTrue(masked_png.exists())
            self.assertGreater(masked_png.stat().st_size, 0)

    def test_load_or_compute_coherence_cache_loads_existing_dataset(self):
        coh_ds = xr.Dataset(
            data_vars={
                'cpsd': (['pair', 'freq'], np.array([[1.0 + 1.0j]], dtype=np.complex128)),
                'coherence': (['pair', 'freq'], np.array([[0.5 + 0.25j]], dtype=np.complex128)),
            },
            coords={
                'pair': ['IT2__PV3'],
                'freq': np.array([10.0]),
                'pop_i': ('pair', ['IT2']),
                'pop_j': ('pair', ['PV3']),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_cache = Path(tmpdir) / 'coh.nc'
            save_xr(coh_ds, fpath_cache)
            actual = self.module._load_or_compute_coherence_cache(None, ['IT2', 'PV3'], [('IT2', 'PV3')], fpath_cache)
        np.testing.assert_allclose(actual['cpsd'].values, coh_ds['cpsd'].values)
        np.testing.assert_allclose(actual['coherence'].values, coh_ds['coherence'].values)

    def test_coherence_cache_dataset_round_trip_preserves_complex_vars(self):
        coh_ds = xr.Dataset(
            data_vars={
                'cpsd': (['pair', 'freq'], np.array([[1.0 + 1.0j, 2.0 + 0.0j]], dtype=np.complex128)),
                'coherence': (
                    ['pair', 'freq'],
                    np.array([[0.5 + 0.25j, 0.25 - 0.5j]], dtype=np.complex128),
                ),
            },
            coords={
                'pair': ['IT2__PV3'],
                'freq': np.array([10.0, 12.0]),
                'pop_i': ('pair', ['IT2']),
                'pop_j': ('pair', ['PV3']),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_cache = Path(tmpdir) / 'coh.nc'
            save_xr(coh_ds, fpath_cache)
            actual = load_xr(fpath_cache, data_type='dataset', load=True)
        self.assertEqual(actual.sizes['pair'], 1)
        self.assertEqual(actual.sizes['freq'], 2)
        np.testing.assert_allclose(actual['cpsd'].values, coh_ds['cpsd'].values)
        np.testing.assert_allclose(actual['coherence'].values, coh_ds['coherence'].values)
