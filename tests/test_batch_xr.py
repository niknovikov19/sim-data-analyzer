import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from sim_data_analyzer import batch_xr as collected
from sim_data_analyzer.tests.test_netpyne_res_parse_utils import _make_sim_result
from sim_data_analyzer.xr_io import save_xr


def _make_job_sim_result(extra_it2_spikes: int):
    sim_result = copy.deepcopy(_make_sim_result())
    spkid = list(sim_result["simData"]["spkid"])
    spkt = list(sim_result["simData"]["spkt"])
    for idx in range(extra_it2_spikes):
        spkid.append(0)
        spkt.append(1.5 + 0.5 * idx)
    order = np.argsort(np.asarray(spkt, dtype=float))
    sim_result["simData"]["spkid"] = [spkid[idx] for idx in order]
    sim_result["simData"]["spkt"] = [spkt[idx] for idx in order]
    return sim_result


def _write_cfg(dirpath_cfg: Path, job_id: int, rx: float, wx: float):
    payload = {
        "simConfig": {
            "rx": rx,
            "wx": wx,
        }
    }
    fpath_cfg = dirpath_cfg / f"grid_{job_id:05d}_cfg.json"
    fpath_cfg.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_job_idx_xr(dirpath_cfg: Path):
    return collected.extract_batch_params_to_xr(
        dirpath_cfg,
        cfg_param_fields={"rx": "rx", "wx": "wx"},
        fname_cfg_templ="grid_*_cfg.json",
        job_pos_in_fname=-2,
    )


def _make_rate_batch(root: Path):
    dirpath_cfg = root / "cfg"
    dirpath_data = root / "data"
    dirpath_cfg.mkdir()
    dirpath_data.mkdir()

    for job_id, rx in enumerate([10.0, 20.0]):
        _write_cfg(dirpath_cfg, job_id, rx=rx, wx=0.1)
        sim_result = _make_job_sim_result(extra_it2_spikes=job_id)
        with (dirpath_data / f"grid_{job_id:05d}_data.pkl").open("wb") as fobj:
            pickle.dump(sim_result, fobj)

    return _build_job_idx_xr(dirpath_cfg), dirpath_data


def _make_json_batch(root: Path):
    dirpath_cfg = root / "cfg"
    dirpath_json = root / "results"
    dirpath_cfg.mkdir()
    dirpath_json.mkdir()

    for job_id, rx in enumerate([10.0, 20.0]):
        _write_cfg(dirpath_cfg, job_id, rx=rx, wx=0.1)
        payload = {
            "rates": {"IT2": 1.0 + job_id, "PV2": 2.0 + job_id},
            "cvs": {"IT2": 0.1 + job_id, "PV2": 0.2 + job_id},
        }
        (dirpath_json / f"result_{job_id:05d}_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    return _build_job_idx_xr(dirpath_cfg), dirpath_json


def _make_xr_batch(root: Path):
    dirpath_cfg = root / "cfg"
    dirpath_xr = root / "rates_xr"
    dirpath_cfg.mkdir()
    dirpath_xr.mkdir()

    freq = np.array([5.0, 10.0, 20.0])
    for job_id, rx in enumerate([10.0, 20.0]):
        _write_cfg(dirpath_cfg, job_id, rx=rx, wx=0.1)
        X = xr.DataArray(
            np.array([1.0, 2.0, 3.0]) + job_id,
            dims=["freq"],
            coords={"freq": freq},
            attrs={"kind": "psd"},
        )
        save_xr(X, dirpath_xr / f"summary_{job_id:05d}.nc")

    return _build_job_idx_xr(dirpath_cfg), dirpath_xr


def _make_xr_set_batch(root: Path):
    dirpath_cfg = root / "cfg"
    dirpath_xr = root / "rates_xr"
    dirpath_cfg.mkdir()
    dirpath_xr.mkdir()

    for job_id, rx in enumerate([10.0, 20.0]):
        _write_cfg(dirpath_cfg, job_id, rx=rx, wx=0.1)
        for pop_name, base in [("IT2", 1.0), ("PV2", 10.0)]:
            X = xr.DataArray(
                np.array([base, base + 1.0]) + job_id,
                dims=["feat"],
                coords={"feat": ["a", "b"]},
            )
            save_xr(X, dirpath_xr / f"job_{job_id:05d}_{pop_name}.nc")

    return _build_job_idx_xr(dirpath_cfg), dirpath_xr


class TestCollectedBatchXR(unittest.TestCase):
    def _assert_same_values(self, X1, X2):
        if isinstance(X1, xr.DataArray):
            np.testing.assert_allclose(np.asarray(X1.values), np.asarray(X2.values))
            self.assertEqual(X1.dims, X2.dims)
            return

        self.assertEqual(set(X1.data_vars), set(X2.data_vars))
        for var_name in X1.data_vars:
            np.testing.assert_allclose(
                np.asarray(X1[var_name].values),
                np.asarray(X2[var_name].values),
            )
            self.assertEqual(X1[var_name].dims, X2[var_name].dims)

    def _assert_cache_behavior(self, call_fn, cache_path: Path, mismatch_kwargs: dict[str, object]):
        X_cached = call_fn(cache_path=cache_path)
        self.assertTrue(cache_path.exists())
        self.assertIn("cache_info", X_cached.attrs)

        X_reused = call_fn(cache_path=cache_path)
        self.assertIn("cache_info", X_reused.attrs)
        self._assert_same_values(X_cached, X_reused)

        with self.assertRaises(ValueError):
            call_fn(cache_path=cache_path, **mismatch_kwargs)

        with self.assertRaises(ValueError):
            call_fn(lazy=True)

        fpath_lazy = cache_path.with_name(cache_path.stem + "_lazy" + cache_path.suffix)
        X_lazy = call_fn(cache_path=fpath_lazy, lazy=True)
        self.assertTrue(fpath_lazy.exists())
        self.assertIn("cache_info", X_lazy.attrs)
        self._assert_same_values(X_cached, X_lazy)

    def test_smoke_exports(self):
        for name in [
                "extract_batch_params_to_xr",
                "iter_batch_jobs",
                "collect_batch_xr",
                "collect_batch_xr_set",
                "collect_batch_json",
                "collect_batch_rates_from_pkl",
                "collect_batch_lfp_from_pkl"]:
            self.assertTrue(hasattr(collected, name), name)

        for name in [
                "write_batch_netcdf",
                "write_batch_xr_netcdf",
                "write_batch_xr_set_netcdf",
                "write_batch_json_netcdf",
                "write_batch_rates_from_pkl_netcdf",
                "write_batch_lfp_from_pkl_netcdf",
                "collect_batch_rates_from_spike_data",
                "write_batch_rates_from_spike_data_netcdf"]:
            self.assertFalse(hasattr(collected, name), name)

    def test_extract_batch_params_to_xr_builds_expected_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath_cfg = Path(tmpdir) / "cfg"
            dirpath_cfg.mkdir(parents=True, exist_ok=True)
            _write_cfg(dirpath_cfg, 0, rx=10.0, wx=0.1)
            _write_cfg(dirpath_cfg, 1, rx=20.0, wx=0.1)

            job_idx_xr = _build_job_idx_xr(dirpath_cfg)

            self.assertEqual(job_idx_xr.dims, ("rx", "wx"))
            self.assertEqual(job_idx_xr.sel(rx=10.0, wx=0.1).item(), 0)
            self.assertEqual(job_idx_xr.sel(rx=20.0, wx=0.1).item(), 1)

    def test_collect_batch_rates_from_pkl_eager_and_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_idx_xr, dirpath_data = _make_rate_batch(root)

            rates = collected.collect_batch_rates_from_pkl(
                job_idx_xr,
                dirpath_data,
                dt_bin=1e-3,
            )

            self.assertEqual(rates.dims, ("rx", "wx", "pop", "time"))
            self.assertIn("job_id", rates.coords)
            self.assertEqual(rates.coords["job_id"].sel(rx=10.0, wx=0.1).item(), 0)
            self.assertEqual(rates.coords["job_id"].sel(rx=20.0, wx=0.1).item(), 1)
            self.assertGreater(
                float(rates.sel(rx=20.0, wx=0.1, pop="IT2").sum()),
                float(rates.sel(rx=10.0, wx=0.1, pop="IT2").sum()),
            )

            self._assert_cache_behavior(
                lambda **kwargs: collected.collect_batch_rates_from_pkl(
                    job_idx_xr,
                    dirpath_data,
                    dt_bin=1e-3,
                    chunks={"rx": 1, "time": 3},
                    **kwargs,
                ),
                root / "cache" / "batch_rates.nc",
                {"fname_templ": "grid_{job:05d}_other.pkl"},
            )

    def test_collect_batch_lfp_from_pkl_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_idx_xr, dirpath_data = _make_rate_batch(root)

            lfp = collected.collect_batch_lfp_from_pkl(
                job_idx_xr,
                dirpath_data,
            )

            self.assertEqual(lfp.dims[:2], ("rx", "wx"))
            self.assertIn("job_id", lfp.coords)

            self._assert_cache_behavior(
                lambda **kwargs: collected.collect_batch_lfp_from_pkl(
                    job_idx_xr,
                    dirpath_data,
                    chunks={"rx": 1, "time": 3},
                    **kwargs,
                ),
                root / "cache" / "batch_lfp.nc",
                {"fname_templ": "grid_{job:05d}_other.pkl"},
            )

    def test_collect_batch_json_adds_population_dimension_and_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_idx_xr, dirpath_json = _make_json_batch(root)

            X = collected.collect_batch_json(
                job_idx_xr,
                dirpath_json,
                var_mappings={"rate": "rates", "cv": "cvs"},
                dict_dims={"pop": ["IT2", "PV2"]},
            )

            self.assertEqual(X["rate"].dims, ("rx", "wx", "pop"))
            self.assertAlmostEqual(X["rate"].sel(rx=10.0, wx=0.1, pop="PV2").item(), 2.0)
            self.assertAlmostEqual(X["cv"].sel(rx=20.0, wx=0.1, pop="IT2").item(), 1.1)

            self._assert_cache_behavior(
                lambda **kwargs: collected.collect_batch_json(
                    job_idx_xr,
                    dirpath_json,
                    var_mappings={"rate": "rates", "cv": "cvs"},
                    dict_dims={"pop": ["IT2", "PV2"]},
                    chunks={"rx": 1},
                    **kwargs,
                ),
                root / "cache" / "batch_json.nc",
                {"fname_templ": "result_{job:05d}_other.json"},
            )

    def test_collect_batch_xr_carries_native_dims_and_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_idx_xr, dirpath_xr = _make_xr_batch(root)

            X = collected.collect_batch_xr(
                job_idx_xr,
                dirpath_xr,
                fname_templ="summary_{job:05d}.nc",
            )

            self.assertEqual(X.dims, ("rx", "wx", "freq"))
            self.assertEqual(X.attrs["kind"], "psd")
            np.testing.assert_allclose(X.coords["freq"].values, np.array([5.0, 10.0, 20.0]))

            self._assert_cache_behavior(
                lambda **kwargs: collected.collect_batch_xr(
                    job_idx_xr,
                    dirpath_xr,
                    fname_templ="summary_{job:05d}.nc",
                    **kwargs,
                ),
                root / "cache" / "batch_xr.nc",
                {"variable": "other"},
            )

    def test_collect_batch_xr_set_concat_builds_new_pop_dim_and_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            job_idx_xr, dirpath_xr = _make_xr_set_batch(root)

            X = collected.collect_batch_xr_set(
                job_idx_xr,
                dirpath_xr,
                fname_templ="job_{job:05d}_{label}.nc",
                labels=["IT2", "PV2"],
                combine="concat",
                concat_dim="pop",
            )

            self.assertEqual(X.dims, ("rx", "wx", "pop", "feat"))
            self.assertEqual(list(X.coords["pop"].values), ["IT2", "PV2"])
            self.assertAlmostEqual(X.sel(rx=10.0, wx=0.1, pop="PV2", feat="a").item(), 10.0)

            self._assert_cache_behavior(
                lambda **kwargs: collected.collect_batch_xr_set(
                    job_idx_xr,
                    dirpath_xr,
                    fname_templ="job_{job:05d}_{label}.nc",
                    labels=["IT2", "PV2"],
                    combine="concat",
                    concat_dim="pop",
                    **kwargs,
                ),
                root / "cache" / "batch_xr_set.nc",
                {"select_label": "IT2"},
            )


if __name__ == "__main__":
    unittest.main()
