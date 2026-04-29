import importlib.util
import unittest
from pathlib import Path

import numpy as np

from sim_data_analyzer import signal_filters as collected


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_signal():
    fs = 100.0
    tt = np.arange(0.0, 2.0, 1.0 / fs)
    xx = (
        np.sin(2 * np.pi * 10 * tt)
        + 0.5 * np.sin(2 * np.pi * 30 * tt)
        + 0.1 * np.sin(2 * np.pi * 2 * tt)
    )
    return tt, xx


class TestCollectedSignalFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.sim_res_common = _load_module(
            'sim_res_common',
            str(repo_root / 'sim_res_analyzer/code/common.py'),
        )
        cls.tt, cls.xx = _make_signal()

    def test_smoke_exports(self):
        self.assertTrue(hasattr(collected, 'filter_signal'))
        self.assertTrue(hasattr(collected, 'filter_signal_bandpass'))

    def test_bandpass_matches_sim_res_analyzer_overlap(self):
        expected = self.sim_res_common.filter_signal(
            self.xx, self.tt, (8.0, 12.0), order=3
        )
        actual = collected.filter_signal(
            self.xx, t=self.tt, fband=(8.0, 12.0), order=3, btype='bandpass'
        )
        np.testing.assert_allclose(actual, expected)

    def test_bandpass_wrapper_matches_generalized_helper(self):
        expected = collected.filter_signal(
            self.xx, t=self.tt, fband=(8.0, 12.0), order=3, btype='bandpass'
        )
        actual = collected.filter_signal_bandpass(
            self.xx, self.tt, (8.0, 12.0), order=3
        )
        np.testing.assert_allclose(actual, expected)

    def test_generalized_modes(self):
        for btype, fband in [
                ('bandpass', (8.0, 12.0)),
                ('lowpass', 20.0),
                ('highpass', 5.0),
                ('bandstop', (25.0, 35.0))]:
            yy = collected.filter_signal(
                self.xx, t=self.tt, fband=fband, order=3, btype=btype
            )
            self.assertEqual(yy.shape, self.xx.shape)
            self.assertTrue(np.isfinite(yy).all())

    def test_fs_inference_matches_explicit_fs(self):
        expected = collected.filter_signal(
            self.xx, t=self.tt, fband=(8.0, 12.0), order=3, btype='bandpass'
        )
        actual = collected.filter_signal(
            self.xx, fs=100.0, fband=(8.0, 12.0), order=3, btype='bandpass'
        )
        np.testing.assert_allclose(actual, expected)

    def test_invalid_argument_combinations_raise(self):
        with self.assertRaises(ValueError):
            collected.filter_signal(self.xx, fband=(8.0, 12.0))
        with self.assertRaises(ValueError):
            collected.filter_signal(self.xx, t=self.tt, fband=10.0, btype='bandpass')
        with self.assertRaises(ValueError):
            collected.filter_signal(self.xx, t=self.tt, fband=(8.0, 12.0), btype='lowpass')
        with self.assertRaises(ValueError):
            collected.filter_signal(self.xx, t=self.tt, fband=(8.0, 12.0), btype='not-a-filter')


if __name__ == '__main__':
    unittest.main()
