import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_data_analyzer.tests.test_netpyne_res_parse_utils import _make_sim_result


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_job_sim_result(extra_it2_spikes: int):
    sim_result = _make_sim_result()
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
    payload = {"simConfig": {"rx": rx, "wx": wx}}
    (dirpath_cfg / f"grid_{job_id:05d}_cfg.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


class TestBatchRatePsdChainScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.collected = _load_module(
            "batch_rate_psd_chain_script",
            repo_root / "dev_scratch" / "demo" / "batch" / "batch_rate_psd_chain.py",
        )

    def test_run_batch_rate_psd_chain_computes_then_reuses_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            dirpath_cfg = dirpath / "cfg"
            dirpath_data = dirpath / "data"
            dirpath_cache = dirpath / "cache"
            dirpath_cfg.mkdir()
            dirpath_data.mkdir()
            dirpath_cache.mkdir()

            for job_id, rx in enumerate([10.0, 20.0]):
                _write_cfg(dirpath_cfg, job_id, rx=rx, wx=0.1)
                sim_result = _make_job_sim_result(extra_it2_spikes=job_id)
                with (dirpath_data / f"grid_{job_id:05d}_data.pkl").open("wb") as fobj:
                    pickle.dump(sim_result, fobj)

            kwargs = {
                "dirpath_cfg": dirpath_cfg,
                "dirpath_batch_data": dirpath_data,
                "dirpath_cache": dirpath_cache,
                "cfg_param_fields": {"rx": "rx", "wx": "wx"},
                "fname_cfg_templ": "grid_*_cfg.json",
                "job_pos_in_fname": -2,
                "fname_data_templ": "grid_{job:05d}_data.pkl",
                "dt_bin": 1e-3,
                "win_len": 2e-3,
                "win_overlap": 0.5,
                "fmin": 2.0,
                "fmax": 20.0,
                "average": "median",
                "compute_psd": False,
                "rate_chunks": {"rx": 1, "time": 3},
                "rate_open_kwargs": {"chunks": {"rx": 1, "time": 3}},
                "psd_open_kwargs": {"chunks": {"rx": 1}},
            }

            first = self.collected.run_batch_rate_psd_chain(**kwargs)
            second = self.collected.run_batch_rate_psd_chain(**kwargs)

            self.assertTrue(first["rates_cache_path"].exists())
            self.assertTrue(first["psd_cache_path"].exists())
            self.assertEqual(first["rates_xr"].dims, ("rx", "wx", "pop", "time"))
            self.assertEqual(first["psd_xr"].dims, ("rx", "wx", "pop", "freq"))
            self.assertEqual(second["rates_xr"].dims, ("rx", "wx", "pop", "time"))
            self.assertEqual(second["psd_xr"].dims, ("rx", "wx", "pop", "freq"))
            self.assertIn("cache_info", first["psd_xr"].attrs)


if __name__ == "__main__":
    unittest.main()
