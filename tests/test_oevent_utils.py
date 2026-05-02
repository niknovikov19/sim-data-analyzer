import contextlib
import io
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sim_data_analyzer.oevent_utils import (
    OEventAnalyzer,
    OEventDetectionParams,
    OEventSpectrogramParams,
)


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

    def test_threshold_change_reuses_same_bundle_and_changes_alpha_counts(self):
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

        alpha_low = int((df_low['band'] == 'alpha').sum())
        alpha_high = int((df_high['band'] == 'alpha').sum())
        self.assertEqual(bundle.spectrogram_params, analyzer.spectrogram_params)
        self.assertEqual(bundle.lsidx, lsidx_before)
        self.assertEqual(bundle.leidx, leidx_before)
        np.testing.assert_allclose(bundle.stacked_tfr(), tfr_before)
        self.assertLess(alpha_low, alpha_high)

    def test_alpha_band_override_changes_grouping_without_rebuilding_bundle(self):
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

        alpha_default = int((df_default['band'] == 'alpha').sum())
        alpha_widened = int((df_widened['band'] == 'alpha').sum())
        self.assertLess(alpha_default, alpha_widened)
        self.assertIn(8.5, df_widened.loc[df_widened['band'] == 'alpha', 'peakF'].tolist())


if __name__ == '__main__':
    unittest.main()
