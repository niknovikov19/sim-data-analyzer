import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer import xr_cache as collected
from sim_data_analyzer.xr_spect import calc_xr_welch


def _make_signal_xr(with_nested_attr: bool = False):
    tt = np.arange(0.0, 2.0, 0.05)
    X = xr.DataArray(
        np.sin(2 * np.pi * 6.0 * tt),
        dims=["time"],
        coords={"time": tt},
        attrs={"source": "test-signal"},
    )
    if with_nested_attr:
        X.attrs["nested"] = {"bands": [5.0, 12.0], "flag": True}
    return X


class TestCollectedXRCache(unittest.TestCase):
    def test_smoke_exports(self):
        for name in [
                "encode_xr_attrs_json",
                "decode_xr_attrs_json",
                "normalize_cache_params",
                "make_cache_info",
                "stamp_xr_cache_info",
                "infer_source_fingerprint",
                "load_or_run_xr"]:
            self.assertTrue(hasattr(collected, name), name)

    def test_attr_json_encode_decode_round_trip(self):
        X = _make_signal_xr(with_nested_attr=True)
        encoded = collected.encode_xr_attrs_json(X)
        self.assertIsInstance(encoded.attrs["nested"], str)
        decoded = collected.decode_xr_attrs_json(encoded)
        self.assertEqual(decoded.attrs["nested"], X.attrs["nested"])

    def test_load_or_run_xr_cache_miss_then_hit(self):
        calls = {"count": 0}

        def build_signal(*, scale=1.0):
            calls["count"] += 1
            return _make_signal_xr(with_nested_attr=True) * scale

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_cache = Path(tmpdir) / "signal.nc"
            X1, cache_hit1 = collected.load_or_run_xr(
                fpath_cache,
                build_signal,
                scale=2.0,
            )
            X2, cache_hit2 = collected.load_or_run_xr(
                fpath_cache,
                build_signal,
                scale=2.0,
                load=True,
            )

            self.assertFalse(cache_hit1)
            self.assertTrue(cache_hit2)
            self.assertEqual(calls["count"], 1)
            self.assertIn("cache_info", X1.attrs)
            self.assertIn("proc_steps", X1.attrs)
            self.assertEqual(X2.attrs["cache_info"]["step"], "build_signal")
            xr.testing.assert_allclose(X1, X2)

    def test_load_or_run_xr_param_mismatch_recomputes(self):
        calls = {"count": 0}

        def build_signal(*, scale=1.0):
            calls["count"] += 1
            return _make_signal_xr() * scale

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_cache = Path(tmpdir) / "signal.nc"
            X1, cache_hit1 = collected.load_or_run_xr(fpath_cache, build_signal, scale=1.0)
            X2, cache_hit2 = collected.load_or_run_xr(fpath_cache, build_signal, scale=3.0)

            self.assertFalse(cache_hit1)
            self.assertFalse(cache_hit2)
            self.assertEqual(calls["count"], 2)
            self.assertFalse(np.allclose(X1.values, X2.values))

    def test_load_or_run_xr_tracks_upstream_cached_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            fpath_signal = dirpath / "signal.nc"
            fpath_psd = dirpath / "psd.nc"

            signal, _cache_hit_signal = collected.load_or_run_xr(
                fpath_signal,
                _make_signal_xr,
            )
            psd, _cache_hit_psd = collected.load_or_run_xr(
                fpath_psd,
                calc_xr_welch,
                signal,
                win_len=0.5,
                win_overlap=0.5,
                fmax=20.0,
                compute=False,
            )

            source = psd.attrs["cache_info"]["source"]
            self.assertEqual(source["kind"], "xr_cache")
            self.assertEqual(
                source["cache_id"],
                signal.attrs["cache_info"]["cache_id"],
            )


if __name__ == "__main__":
    unittest.main()
