import tempfile
import unittest
import pickle
from pathlib import Path

import xarray as xr

from sim_data_analyzer import scratch_data as collected
from sim_data_analyzer.tests.test_netpyne_res_parse_utils import _make_sim_result
from sim_data_analyzer.xr_adapters import get_lfp_xr, get_net_rate_dynamics_xr
from sim_data_analyzer.xr_io import save_xr


class TestScratchData(unittest.TestCase):
    def test_smoke_exports(self):
        for name in [
                'get_exp_label',
                'get_proc_dir',
                'get_lfp_cache_path',
                'get_rates_cache_path',
                'load_sim_result',
                'load_or_extract_lfp',
                'load_or_extract_rates']:
            self.assertTrue(hasattr(collected, name), name)

    def test_exp_label_derivation(self):
        fpath_sim_result = Path('/tmp/a1_lfp_15s/data_00000_seed_1000.pkl')
        self.assertEqual(collected.get_exp_label(fpath_sim_result), 'a1_lfp_15s_0')

    def test_cache_path_construction(self):
        fpath_sim_result = Path('/tmp/a1_lfp_15s/data_00000_seed_1000.pkl')
        dirpath_proc_root = Path('/cache/root')
        self.assertEqual(
            collected.get_proc_dir(fpath_sim_result, dirpath_proc_root),
            dirpath_proc_root / 'a1_lfp_15s_0',
        )
        self.assertEqual(
            collected.get_lfp_cache_path(fpath_sim_result, dirpath_proc_root),
            dirpath_proc_root / 'a1_lfp_15s_0' / 'a1_lfp_15s_0_lfp.nc',
        )
        self.assertEqual(
            collected.get_rates_cache_path(fpath_sim_result, dirpath_proc_root, 5e-3),
            dirpath_proc_root / 'a1_lfp_15s_0' / 'a1_lfp_15s_0_rates_dt_0.005.nc',
        )

    def test_load_sim_result_round_trip(self):
        sim_result = {'simData': {'t': [0.0, 1.0]}, 'meta': {'seed': 1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_sim_result = Path(tmpdir) / 'src' / 'data_00000_seed_1000.pkl'
            fpath_sim_result.parent.mkdir(parents=True, exist_ok=True)
            with fpath_sim_result.open('wb') as fobj:
                pickle.dump(sim_result, fobj)
            self.assertEqual(collected.load_sim_result(fpath_sim_result), sim_result)

    def test_load_or_extract_lfp_loads_existing_cache_without_raw_result(self):
        sim_result = _make_sim_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_sim_result = Path(tmpdir) / 'src' / 'data_00000_seed_1000.pkl'
            fpath_sim_result.parent.mkdir(parents=True, exist_ok=True)
            fpath_sim_result.write_bytes(b'placeholder')
            dirpath_proc_root = Path(tmpdir) / 'proc'

            expected = get_lfp_xr(sim_result)
            cache_path = collected.get_lfp_cache_path(fpath_sim_result, dirpath_proc_root)
            save_xr(expected, cache_path)

            actual = collected.load_or_extract_lfp(None, fpath_sim_result, dirpath_proc_root)
            xr.testing.assert_identical(actual, expected)

    def test_load_or_extract_rates_loads_existing_cache_without_raw_result(self):
        sim_result = _make_sim_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_sim_result = Path(tmpdir) / 'src' / 'data_00000_seed_1000.pkl'
            fpath_sim_result.parent.mkdir(parents=True, exist_ok=True)
            fpath_sim_result.write_bytes(b'placeholder')
            dirpath_proc_root = Path(tmpdir) / 'proc'

            expected = get_net_rate_dynamics_xr(sim_result, dt_bin=5e-3, avg_cells=True)
            cache_path = collected.get_rates_cache_path(fpath_sim_result, dirpath_proc_root, 5e-3)
            save_xr(expected, cache_path)

            actual = collected.load_or_extract_rates(
                None, fpath_sim_result, dirpath_proc_root, rate_dt=5e-3
            )
            xr.testing.assert_identical(actual, expected)

    def test_load_or_extract_lfp_missing_cache_requires_raw_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_sim_result = Path(tmpdir) / 'src' / 'data_00000_seed_1000.pkl'
            dirpath_proc_root = Path(tmpdir) / 'proc'
            with self.assertRaises(ValueError):
                collected.load_or_extract_lfp(None, fpath_sim_result, dirpath_proc_root)

    def test_load_or_extract_rates_missing_cache_requires_raw_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_sim_result = Path(tmpdir) / 'src' / 'data_00000_seed_1000.pkl'
            dirpath_proc_root = Path(tmpdir) / 'proc'
            with self.assertRaises(ValueError):
                collected.load_or_extract_rates(None, fpath_sim_result, dirpath_proc_root, rate_dt=5e-3)


if __name__ == '__main__':
    unittest.main()
