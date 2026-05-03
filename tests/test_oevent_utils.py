import contextlib
import io
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sim_data_analyzer.oevent_utils import (
    OEventAnalyzer,
    OEventDetectionParams,
    OEventSpectrogramParams,
    build_band_event_result_dataset,
    event_table_from_dataset,
    normalize_band_event_table,
    prepare_csv_event_table,
    resolve_xr_channel_selection,
)
from sim_data_analyzer.xr_io import load_xr, save_xr


def _build_signal_for_replay():
    sampr = 500.0
    t = np.arange(0, 6.0, 1 / sampr)
    rng = np.random.default_rng(0)
    sig = 0.02 * rng.standard_normal(t.shape)
    mask_strong = (t >= 1.0) & (t < 2.5)
    mask_weak = (t >= 3.2) & (t < 4.8)
    sig[mask_strong] += 3.0 * np.sin(2 * np.pi * 10 * t[mask_strong])
    sig[mask_weak] += 2.0 * np.sin(2 * np.pi * 8 * t[mask_weak])
    return sampr, sig


def _build_signal_for_band_override():
    sampr = 500.0
    t = np.arange(0, 6.0, 1 / sampr)
    rng = np.random.default_rng(1)
    sig = 0.02 * rng.standard_normal(t.shape)
    mask = (t >= 1.0) & (t < 2.8)
    sig[mask] += 3.2 * np.sin(2 * np.pi * 8 * t[mask])
    return sampr, sig


def _make_spectrogram_params(sampr: float) -> OEventSpectrogramParams:
    return OEventSpectrogramParams(
        winsz=6.0,
        sampr=sampr,
        freqmin=4.0,
        freqmax=20.0,
        freqstep=0.5,
        getphase=True,
        useloglfreq=False,
        mspecwidth=7.0,
        noiseamp=9999.0,
        normop_name='mednorm',
    )


def _run_detection(analyzer, signal, bundle, detection_params):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with contextlib.redirect_stdout(io.StringIO()):
            dout = analyzer.detect_from_bundle(bundle, signal, detection_params)
            dframe = analyzer.to_dataframe(dout, signal, haveMUA=False)
    return dout, dframe


def _make_mock_event_frame():
    return pd.DataFrame({
        'band': ['theta', 'alpha', 'alpha', 'beta'],
        'chan': [0, 0, 0, 0],
        'ncycle': [2.0, 4.0, 2.5, 5.0],
        'Foct': [0.2, 0.4, 1.7, 0.5],
        'filtsigcor': [0.5, 0.4, 0.1, 0.6],
        'absPeakT': [100.0, 200.0, 350.0, 450.0],
        'absminT': [80.0, 180.0, 300.0, 420.0],
        'absmaxT': [120.0, 240.0, 390.0, 510.0],
        'dur': [40.0, 60.0, 90.0, 90.0],
        'peakF': [7.5, 10.5, 12.0, 18.0],
        'avgpow': [1.0, 2.0, 3.0, 4.0],
        'avgpowevent': [1.5, 2.5, 3.5, 4.5],
        'OSCscore': [2.0, 5.0, 1.0, 6.0],
        'minF': [6.5, 9.0, 10.0, 16.0],
        'maxF': [8.0, 12.0, 14.0, 20.0],
        'CSDwvf': [np.array([1, 2]), np.array([3, 4]), np.array([5]), np.array([6])],
        'filtsig': [np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])],
    })


class TestOEventUtils(unittest.TestCase):
    def test_detect_from_bundle_matches_detect_from_signal(self):
        sampr, signal = _build_signal_for_replay()
        analyzer = OEventAnalyzer(_make_spectrogram_params(sampr))
        detection_params = OEventDetectionParams(
            medthresh=1.2,
            overlapth=0.5,
            band_overrides={'alpha': (7.0, 15.0)},
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = analyzer.build_bundle(signal)
                dout_from_signal = analyzer.detect_from_signal(signal, detection_params)
                dout_from_bundle = analyzer.detect_from_bundle(bundle, signal, detection_params)
                df_from_signal = analyzer.to_dataframe(dout_from_signal, signal, haveMUA=False)
                df_from_bundle = analyzer.to_dataframe(dout_from_bundle, signal, haveMUA=False)

        cols = ['band', 'peakF', 'minF', 'maxF', 'left', 'right', 'bottom', 'top', 'ncycle']
        actual = df_from_bundle[cols].sort_values(cols).reset_index(drop=True)
        expected = df_from_signal[cols].sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_bundle_pickle_round_trip_preserves_replay(self):
        sampr, signal = _build_signal_for_replay()
        analyzer = OEventAnalyzer(_make_spectrogram_params(sampr))
        detection_params = OEventDetectionParams(
            medthresh=1.2,
            overlapth=0.5,
            band_overrides={'alpha': (7.0, 15.0)},
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = analyzer.build_bundle(signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'bundle.pkl'
            with fpath.open('wb') as fobj:
                pickle.dump(bundle, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            with fpath.open('rb') as fobj:
                loaded = pickle.load(fobj)

        self.assertEqual(bundle.spectrogram_params, loaded.spectrogram_params)
        np.testing.assert_allclose(bundle.stacked_tfr(), loaded.stacked_tfr())

        _, df_original = _run_detection(analyzer, signal, bundle, detection_params)
        _, df_loaded = _run_detection(analyzer, signal, loaded, detection_params)
        cols = ['band', 'peakF', 'minF', 'maxF', 'left', 'right', 'bottom', 'top', 'ncycle']
        actual = df_loaded[cols].sort_values(cols).reset_index(drop=True)
        expected = df_original[cols].sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_threshold_change_reuses_same_bundle_and_changes_selected_band_counts(self):
        sampr, signal = _build_signal_for_replay()
        analyzer = OEventAnalyzer(_make_spectrogram_params(sampr))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = analyzer.build_bundle(signal)
        lsidx_before = list(bundle.lsidx)
        leidx_before = list(bundle.leidx)
        tfr_before = bundle.stacked_tfr().copy()

        low_threshold = OEventDetectionParams(
            medthresh=1.0,
            overlapth=0.5,
            band_overrides={'alpha': (7.0, 15.0)},
        )
        high_threshold = OEventDetectionParams(
            medthresh=1.2,
            overlapth=0.5,
            band_overrides={'alpha': (7.0, 15.0)},
        )

        _, df_low = _run_detection(analyzer, signal, bundle, low_threshold)
        _, df_high = _run_detection(analyzer, signal, bundle, high_threshold)

        count_low = int((df_low['band'] == 'alpha').sum())
        count_high = int((df_high['band'] == 'alpha').sum())
        self.assertEqual(bundle.spectrogram_params, analyzer.spectrogram_params)
        self.assertEqual(bundle.lsidx, lsidx_before)
        self.assertEqual(bundle.leidx, leidx_before)
        np.testing.assert_allclose(bundle.stacked_tfr(), tfr_before)
        self.assertLess(count_low, count_high)

    def test_band_override_changes_grouping_without_rebuilding_bundle(self):
        sampr, signal = _build_signal_for_band_override()
        analyzer = OEventAnalyzer(_make_spectrogram_params(sampr))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = analyzer.build_bundle(signal)

        default_alpha = OEventDetectionParams(medthresh=1.2, overlapth=0.5)
        widened_alpha = OEventDetectionParams(
            medthresh=1.2,
            overlapth=0.5,
            band_overrides={'alpha': (7.0, 15.0)},
        )

        _, df_default = _run_detection(analyzer, signal, bundle, default_alpha)
        _, df_widened = _run_detection(analyzer, signal, bundle, widened_alpha)

        count_default = int((df_default['band'] == 'alpha').sum())
        count_widened = int((df_widened['band'] == 'alpha').sum())
        self.assertLess(count_default, count_widened)
        self.assertIn(8.5, df_widened.loc[df_widened['band'] == 'alpha', 'peakF'].tolist())

    def test_resolve_xr_channel_selection_single_and_multi(self):
        signal_raw = xr.DataArray(
            np.arange(12, dtype=float).reshape(3, 4),
            dims=('y', 'time'),
            coords={'y': [0.0, 600.0, 1600.0], 'time': [0.0, 0.1, 0.2, 0.3]},
        )
        signal_proc = signal_raw - signal_raw.mean(dim='time')

        raw_single, proc_single = resolve_xr_channel_selection(
            signal_raw,
            signal_proc,
            channel_mode='single',
            y=550.0,
        )
        self.assertEqual(raw_single.shape, (1, 4))
        self.assertEqual(float(raw_single.coords['channel_y'].item()), 600.0)
        np.testing.assert_allclose(proc_single.values.mean(axis=1), 0.0)

        raw_multi, proc_multi = resolve_xr_channel_selection(
            signal_raw,
            signal_proc,
            channel_mode='multi',
            y_values=[550.0, 1550.0],
        )
        np.testing.assert_allclose(raw_multi.coords['channel_y'].values, [600.0, 1600.0])
        self.assertEqual(proc_multi.shape, (2, 4))

        raw_all, _ = resolve_xr_channel_selection(
            signal_raw,
            signal_proc,
            channel_mode='multi',
        )
        np.testing.assert_allclose(raw_all.coords['channel_y'].values, [0.0, 600.0, 1600.0])

    def test_lightweight_result_dataset_round_trip(self):
        mock_events = _make_mock_event_frame()
        selected_events, _passed_events = normalize_band_event_table(
            mock_events,
            bands_of_interest=['theta', 'alpha'],
            sampr=1000.0,
            time_offset_s=5.0,
            resolved_y=600.0,
            channel_index=1,
            min_ncycle=3.0,
            max_foct=1.5,
            min_filtsigcor=0.2,
        )
        signal_raw = np.arange(12, dtype=float).reshape(2, 6)
        signal_proc = signal_raw - signal_raw.mean(axis=1, keepdims=True)
        spectrogram_norm = np.arange(2 * 5 * 4, dtype=float).reshape(2, 5, 4)
        time_s = np.linspace(5.0, 5.5, 6)
        spec_time_s = np.linspace(5.0, 5.3, 4)
        freq_hz = np.linspace(4.0, 12.0, 5)
        channel_y = [600.0, 1600.0]

        dataset = build_band_event_result_dataset(
            signal_raw=signal_raw,
            signal_proc=signal_proc,
            spectrogram_norm=spectrogram_norm,
            time_s=time_s,
            spec_time_s=spec_time_s,
            freq_hz=freq_hz,
            channel_y=channel_y,
            event_table=selected_events,
            attrs={'bands_of_interest': ['theta', 'alpha'], 'result_cache_version': 'v1'},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / 'result.nc'
            save_xr(dataset, fpath)
            loaded = load_xr(fpath, data_type='dataset', load=True)

        recovered = event_table_from_dataset(loaded)
        self.assertEqual(loaded['signal_raw'].shape, (2, 6))
        self.assertEqual(loaded['spectrogram_norm'].shape, (2, 5, 4))
        np.testing.assert_allclose(loaded.coords['channel_y'].values, channel_y)
        self.assertEqual(loaded.attrs['result_cache_version'], 'v1')
        self.assertEqual(int(recovered['event_channel'].iloc[0]), 1)
        self.assertEqual(float(recovered['event_y'].iloc[0]), 600.0)
        self.assertEqual(recovered['event_band'].tolist(), ['theta', 'alpha', 'alpha'])
        self.assertEqual(recovered['event_passed'].astype(bool).tolist(), [False, True, False])

    def test_prepare_csv_event_table_keeps_scalar_fields_and_drops_duplicate_status(self):
        mock_events = _make_mock_event_frame()
        selected_events, _passed_events = normalize_band_event_table(
            mock_events,
            bands_of_interest=['theta', 'alpha'],
            sampr=1000.0,
            time_offset_s=5.0,
            resolved_y=600.0,
            channel_index=1,
            min_ncycle=3.0,
            max_foct=1.5,
            min_filtsigcor=0.2,
        )
        csv_table = prepare_csv_event_table(selected_events, round_digits=3)
        self.assertIn('event_passed', csv_table.columns)
        self.assertNotIn('selection_status', csv_table.columns)
        self.assertNotIn('CSDwvf', csv_table.columns)
        self.assertNotIn('filtsig', csv_table.columns)
        self.assertFalse(any('MUA' in col for col in csv_table.columns))
        self.assertEqual(csv_table['event_passed'].astype(bool).tolist(), [False, True, False])
        self.assertAlmostEqual(float(csv_table['peak_time_s'].iloc[1]), 5.2, places=3)


if __name__ == '__main__':
    unittest.main()
