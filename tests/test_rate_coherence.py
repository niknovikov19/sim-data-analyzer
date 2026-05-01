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

    def test_resolve_basis_pop_requires_member_of_analyzed_set(self):
        self.assertEqual(self.module._resolve_basis_pop(['IT2', 'PV3'], 'IT2'), 'IT2')
        with self.assertRaisesRegex(ValueError, 'BASIS_POP'):
            self.module._resolve_basis_pop(['IT2', 'PV3'], 'SOM3')

    def test_normalize_vector_color_scheme_accepts_supported_values(self):
        self.assertEqual(self.module._normalize_vector_color_scheme('cell_type'), 'cell_type')
        self.assertEqual(self.module._normalize_vector_color_scheme('Layer'), 'layer')

    def test_normalize_vector_color_scheme_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'VECTOR_COLOR_SCHEME'):
            self.module._normalize_vector_color_scheme('pop')

    def test_get_pop_cell_type_group_maps_supported_prefixes(self):
        self.assertEqual(self.module._get_pop_cell_type_group('IT3'), 'PYR')
        self.assertEqual(self.module._get_pop_cell_type_group('CT6'), 'PYR')
        self.assertEqual(self.module._get_pop_cell_type_group('PT5B'), 'PYR')
        self.assertEqual(self.module._get_pop_cell_type_group('PV3'), 'PV')
        self.assertEqual(self.module._get_pop_cell_type_group('SOM5A'), 'SOM')
        self.assertEqual(self.module._get_pop_cell_type_group('VIP2'), 'VIP')
        self.assertEqual(self.module._get_pop_cell_type_group('NGF5A'), 'NGF')
        self.assertEqual(self.module._get_pop_cell_type_group('TC'), 'TC')
        self.assertEqual(self.module._get_pop_cell_type_group('HTC'), 'TC')
        self.assertEqual(self.module._get_pop_cell_type_group('TCM'), 'TC')
        self.assertEqual(self.module._get_pop_cell_type_group('TI'), 'TI')
        self.assertEqual(self.module._get_pop_cell_type_group('TIM'), 'TI')
        self.assertEqual(self.module._get_pop_cell_type_group('IRE'), 'IRE')
        self.assertEqual(self.module._get_pop_cell_type_group('IREM'), 'IRE')

    def test_get_pop_layer_group_maps_cortical_and_thalamic_names(self):
        self.assertEqual(self.module._get_pop_layer_group('IT2'), 'L2')
        self.assertEqual(self.module._get_pop_layer_group('PV3'), 'L3')
        self.assertEqual(self.module._get_pop_layer_group('SOM4'), 'L4')
        self.assertEqual(self.module._get_pop_layer_group('IT5A'), 'L5A')
        self.assertEqual(self.module._get_pop_layer_group('PV5B'), 'L5B')
        self.assertEqual(self.module._get_pop_layer_group('CT6'), 'L6')
        self.assertEqual(self.module._get_pop_layer_group('TC'), 'THAL')
        self.assertEqual(self.module._get_pop_layer_group('IREM'), 'THAL')

    def test_get_vector_style_returns_group_and_color(self):
        group, color = self.module._get_vector_style('PV3', 'cell_type')
        self.assertEqual(group, 'PV')
        self.assertEqual(color, self.module.CELL_TYPE_COLORS['PV'])
        group, color = self.module._get_vector_style('IT5A', 'layer')
        self.assertEqual(group, 'L5A')
        self.assertEqual(color, self.module.LAYER_COLORS['L5A'])

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

    def test_get_vector_png_name_uses_basis_and_optional_threshold(self):
        self.assertEqual(
            self.module._get_vector_png_name('IT2', None),
            'vectors__basis_IT2.png',
        )
        self.assertEqual(
            self.module._get_vector_png_name('IT2', 0.4),
            'vectors__basis_IT2__thr_0p4.png',
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

    def test_build_basis_vectors_uses_basis_row_and_threshold(self):
        pop_names = ['IT2', 'PV3', 'SOM3']
        coherence_table = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ])
        phase_table = np.array([
            [0.0, np.pi / 2, -np.pi / 4],
            [-np.pi / 2, 0.0, 0.2],
            [np.pi / 4, -0.2, 0.0],
        ])
        vectors = self.module._build_basis_vectors(
            pop_names,
            coherence_table,
            phase_table,
            basis_pop='IT2',
            coherence_threshold=0.5,
        )
        self.assertEqual([item['pop'] for item in vectors], ['IT2', 'PV3'])
        self.assertAlmostEqual(vectors[0]['endpoint'].real, 1.0)
        self.assertAlmostEqual(vectors[0]['endpoint'].imag, 0.0)
        self.assertAlmostEqual(vectors[1]['endpoint'].real, 0.0, places=12)
        self.assertAlmostEqual(vectors[1]['endpoint'].imag, 0.8)

    def test_get_phase_plot_limit_ignores_masked_values(self):
        phase_plot = np.array([[0.0, 0.1], [-2.0, 0.0]])
        plot_mask = np.array([[True, False], [True, True]])
        actual = self.module._get_phase_plot_limit(phase_plot, plot_mask, fallback=np.pi)
        self.assertAlmostEqual(actual, 0.1)

    def test_get_vector_axis_limits_focus_on_visible_cluster_and_origin(self):
        endpoints = np.array([0.9 + 0.0j, 0.6 - 0.4j, 0.7 - 0.1j], dtype=np.complex128)
        xlim, ylim = self.module._get_vector_axis_limits(endpoints)
        self.assertLess(xlim[0], 0.0)
        self.assertGreater(xlim[1], 0.9)
        self.assertLess(ylim[0], -0.4)
        self.assertGreater(ylim[1], 0.0)
        self.assertNotAlmostEqual(abs(xlim[0]), abs(xlim[1]))

    def test_get_vector_label_offset_returns_directional_offsets(self):
        dx, dy, ha, va = self.module._get_vector_label_offset(0.8 + 0.0j, 0)
        self.assertGreater(dx, 0.0)
        self.assertIn(ha, {'left', 'right'})
        self.assertIn(va, {'bottom', 'top'})
        dx2, dy2, _, _ = self.module._get_vector_label_offset(0.8 + 0.0j, 1)
        self.assertNotEqual(dy, dy2)

    def test_relax_vector_labels_separates_overlapping_positions(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        vectors = [
            {'pop': 'A', 'endpoint': complex(0.8, 0.0), 'coherence': 0.8},
            {'pop': 'B', 'endpoint': complex(0.8, 0.0), 'coherence': 0.8},
        ]
        annotations = []
        for idx, item in enumerate(vectors):
            dx, dy, ha, va = self.module._get_vector_label_offset(item['endpoint'], idx)
            ann = ax.annotate(
                item['pop'],
                xy=(item['endpoint'].real, item['endpoint'].imag),
                xytext=(10.0, 4.0),
                textcoords='offset points',
                ha=ha,
                va=va,
            )
            annotations.append(ann)
        initial_positions = [tuple(ann.get_position()) for ann in annotations]
        self.module._relax_vector_labels(fig, ax, vectors, annotations, n_iter=20)
        final_positions = [tuple(ann.get_position()) for ann in annotations]
        plt.close(fig)
        self.assertNotEqual(initial_positions, final_positions)

    def test_expand_vector_limits_for_annotations_includes_label_box(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-0.1, 1.0)
        ax.set_ylim(-0.2, 0.2)
        ann = ax.annotate(
            'LONG_LABEL',
            xy=(1.0, 0.0),
            xytext=(18.0, 0.0),
            textcoords='offset points',
            ha='left',
            va='center',
        )
        xlim2, ylim2 = self.module._expand_vector_limits_for_annotations(
            fig,
            ax,
            (-0.1, 1.0),
            (-0.2, 0.2),
            [ann],
        )
        plt.close(fig)
        self.assertGreater(xlim2[1], 1.0)
        self.assertLessEqual(xlim2[0], -0.1)
        self.assertLessEqual(ylim2[0], -0.2)
        self.assertGreaterEqual(ylim2[1], 0.2)

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

    def test_make_vector_plot_writes_png(self):
        vectors = [
            {'pop': 'IT2', 'endpoint': complex(1.0, 0.0), 'coherence': 1.0},
            {'pop': 'PV3', 'endpoint': complex(0.2, 0.6), 'coherence': 0.632},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_png = Path(tmpdir) / 'vectors.png'
            self.module._make_vector_plot(
                fpath_png,
                vectors,
                basis_pop='IT2',
                fband=(8.0, 14.0),
                coherence_threshold=0.5,
                color_scheme='cell_type',
            )
            self.assertTrue(fpath_png.exists())
            self.assertGreater(fpath_png.stat().st_size, 0)

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
