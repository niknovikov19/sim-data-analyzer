import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_data_analyzer import netpyne_res_parse_utils as parse_utils
from sim_data_analyzer.spike_data import SpikeData


def _make_sim_result():
    return {
        "simConfig": {
            "recordLFP": [[0, 100], [0, 300]],
            "recordStep": 1.0,
            "dt": 0.1,
            "duration": 6.0,
            "sizeY": 1000,
        },
        "net": {
            "pops": {
                "IT2": {"cellGids": [0, 1], "tags": {"ynormRange": [0.0, 0.2]}},
                "PV2": {"cellGids": [2], "tags": {"ynormRange": [0.2, 0.3]}},
                "SILENT": {"cellGids": [3], "tags": {"ynormRange": [0.3, 0.4]}},
                "EMPTY": {"cellGids": [], "tags": {"ynormRange": [0.4, 0.5]}},
            },
            "cells": [
                {"tags": {"pop": "IT2"}},
                {"tags": {"pop": "IT2"}},
                {"tags": {"pop": "PV2"}},
                {"tags": {"pop": "SILENT"}},
            ],
            "params": {
                "popParams": {
                    "IT2": {"cellType": "IT"},
                    "PV2": {"cellType": "PV"},
                    "SILENT": {"cellType": "IT"},
                    "EMPTY": {"cellType": "IT"},
                }
            },
        },
        "simData": {
            "LFP": [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            "LFPPops": {},
            "spkid": [0, 1, 2, 0, 2],
            "spkt": [1.0, 2.0, 3.0, 4.5, 5.5],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "V_soma": {
                "cell_0": [-65.0, -64.0, -63.0, -62.0, -61.0],
                "cell_1": [-66.0, -65.0, -64.0, -63.0, -62.0],
                "cell_2": [-60.0, -59.0, -58.0, -57.0, -56.0],
                "cell_3": [-70.0, -70.0, -70.0, -70.0, -70.0],
            },
        },
    }


def _assert_spike_dict_equal(testcase, left, right):
    testcase.assertEqual(set(left), set(right))
    for pop_name in left:
        testcase.assertEqual(len(left[pop_name]), len(right[pop_name]))
        for spikes_left, spikes_right in zip(left[pop_name], right[pop_name]):
            np.testing.assert_array_equal(spikes_left, spikes_right)


class TestSpikeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim_result = _make_sim_result()

    def test_from_sim_result_combined_matches_parser(self):
        spike_data = SpikeData.from_sim_result(
            self.sim_result,
            combine=True,
            t0=0.001,
            tmax=0.005,
            subtract_t0=False,
            ms=True,
            ndigits=4,
        )
        expected = parse_utils.get_net_spikes(
            self.sim_result,
            combine_cells=True,
            t0=0.001,
            tmax=0.005,
            subtract_t0=False,
            ms=True,
            ndigits=4,
        )
        _assert_spike_dict_equal(self, spike_data.get_net_spikes(), expected)
        self.assertTrue(spike_data.combine_mode)

    def test_from_sim_result_per_cell_matches_parser(self):
        spike_data = SpikeData.from_sim_result(
            self.sim_result,
            combine=False,
            ms=True,
        )
        expected = parse_utils.get_net_spikes(
            self.sim_result,
            combine_cells=False,
            ms=True,
        )
        _assert_spike_dict_equal(self, spike_data.get_net_spikes(), expected)
        np.testing.assert_array_equal(
            spike_data.get_pop_cell_gids("IT2"),
            np.array([0, 1]),
        )
        self.assertFalse(spike_data.combine_mode)

    def test_subset_pop_names(self):
        spike_data = SpikeData.from_sim_result(
            self.sim_result,
            pop_names=["PV2", "SILENT"],
            combine=True,
        )
        self.assertEqual(spike_data.get_pop_names(), ["PV2", "SILENT"])

    def test_combine_per_cell_matches_direct_combined_extraction(self):
        per_cell = SpikeData.from_sim_result(self.sim_result, combine=False, ms=True)
        combined = per_cell.combine()
        expected = parse_utils.get_net_spikes(
            self.sim_result,
            combine_cells=True,
            ms=True,
        )
        _assert_spike_dict_equal(self, combined.get_net_spikes(), expected)
        self.assertTrue(combined.combine_mode)

    def test_combine_on_combined_data_raises(self):
        combined = SpikeData.from_sim_result(self.sim_result, combine=True)
        with self.assertRaises(ValueError):
            combined.combine()

    def test_roundtrip_save_load_combined(self):
        spike_data = SpikeData.from_sim_result(
            self.sim_result,
            combine=True,
            ms=True,
            t0=0.001,
            tmax=0.005,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "combined_spikes.npz"
            spike_data.save(fpath)
            with np.load(fpath, allow_pickle=False) as data:
                self.assertIn("pop_names", data.files)
            loaded = SpikeData.load(fpath)
        _assert_spike_dict_equal(self, loaded.get_net_spikes(), spike_data.get_net_spikes())
        self.assertEqual(loaded.metadata, spike_data.metadata)

    def test_roundtrip_save_load_per_cell(self):
        spike_data = SpikeData.from_sim_result(
            self.sim_result,
            combine=False,
            ms=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "cell_spikes.npz"
            spike_data.save(fpath)
            with np.load(fpath, allow_pickle=False) as data:
                self.assertIn("pop_0__cell_offsets", data.files)
            loaded = SpikeData.load(fpath)
        _assert_spike_dict_equal(self, loaded.get_net_spikes(), spike_data.get_net_spikes())
        np.testing.assert_array_equal(
            loaded.get_pop_cell_gids("IT2"),
            spike_data.get_pop_cell_gids("IT2"),
        )

    def test_empty_and_silent_populations(self):
        combined = SpikeData.from_sim_result(self.sim_result, combine=True)
        per_cell = SpikeData.from_sim_result(self.sim_result, combine=False)

        self.assertEqual(len(combined.get_pop_spikes("EMPTY")), 1)
        np.testing.assert_array_equal(combined.get_pop_spikes("EMPTY")[0], np.array([]))
        self.assertEqual(len(per_cell.get_pop_spikes("EMPTY")), 0)

        self.assertEqual(len(combined.get_pop_spikes("SILENT")), 1)
        np.testing.assert_array_equal(combined.get_pop_spikes("SILENT")[0], np.array([]))
        self.assertEqual(len(per_cell.get_pop_spikes("SILENT")), 1)
        np.testing.assert_array_equal(per_cell.get_pop_spikes("SILENT")[0], np.array([]))

    def test_combined_data_has_no_cell_gids(self):
        combined = SpikeData.from_sim_result(self.sim_result, combine=True)
        with self.assertRaises(ValueError):
            combined.get_pop_cell_gids("IT2")


if __name__ == "__main__":
    unittest.main()
