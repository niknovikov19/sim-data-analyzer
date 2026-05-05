import unittest
from unittest import mock

import numpy as np
import xarray as xr

from sim_data_analyzer.mask_utils import (
    extract_mask_intervals,
    sample_control_intervals,
    sample_control_intervals_by_channel,
)


class _FakeRng:
    def __init__(self, integers_values=(), uniform_values=()):
        self._integers_values = list(integers_values)
        self._uniform_values = list(uniform_values)

    def integers(self, high):
        if not self._integers_values:
            raise AssertionError('Unexpected integers() call')
        value = self._integers_values.pop(0)
        return int(value % high)

    def uniform(self, low, high):
        if not self._uniform_values:
            raise AssertionError('Unexpected uniform() call')
        value = float(self._uniform_values.pop(0))
        if value < low or value > high:
            raise AssertionError(f'Uniform value {value} outside [{low}, {high}]')
        return value


class TestMaskUtils(unittest.TestCase):
    def test_extract_mask_intervals_single_channel(self):
        mask = xr.DataArray(
            np.array([0, 0, 1, 1, 0, 1, 0], dtype=np.uint8),
            dims=('time',),
            coords={'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        )

        intervals = extract_mask_intervals(mask)

        self.assertEqual(intervals['mask_1'], [(2.0, 3.0), (5.0, 5.0)])
        self.assertEqual(intervals['mask_0'], [(0.0, 1.0), (4.0, 4.0), (6.0, 6.0)])

    def test_extract_mask_intervals_multi_channel(self):
        mask = xr.DataArray(
            np.array([
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ], dtype=np.uint8),
            dims=('y', 'time'),
            coords={'y': [0.0, 200.0], 'time': [10.0, 11.0, 12.0, 13.0]},
        )

        intervals = extract_mask_intervals(mask)

        self.assertEqual(set(intervals['mask_1']), {'y_0', 'y_200'})
        self.assertEqual(intervals['mask_1']['y_0'], [(11.0, 12.0)])
        self.assertEqual(intervals['mask_0']['y_0'], [(10.0, 10.0), (13.0, 13.0)])
        self.assertEqual(intervals['mask_1']['y_200'], [(10.0, 10.0), (13.0, 13.0)])
        self.assertEqual(intervals['mask_0']['y_200'], [(11.0, 12.0)])

    def test_sample_control_intervals_matches_duration_and_no_overlap(self):
        mask1_intervals = [(1.0, 2.5), (4.0, 4.5)]
        mask0_intervals = [(10.0, 14.0), (20.0, 23.0)]

        sampled, successful_seed = sample_control_intervals(
            mask1_intervals,
            mask0_intervals,
            seed=0,
            max_seed_tries=10,
        )

        self.assertGreaterEqual(successful_seed, 0)
        self.assertEqual(len(sampled), len(mask1_intervals))
        for source, control in zip(mask1_intervals, sampled):
            self.assertAlmostEqual(control[1] - control[0], source[1] - source[0])
        for interval in sampled:
            self.assertTrue(
                any(interval[0] >= lo and interval[1] <= hi for lo, hi in mask0_intervals),
                interval,
            )
        for left, right in zip(sampled[:-1], sampled[1:]):
            self.assertFalse(max(left[0], right[0]) <= min(left[1], right[1]))

    def test_sample_control_intervals_retries_next_seed(self):
        mask1_intervals = [(0.0, 6.0), (0.0, 3.0)]
        mask0_intervals = [(0.0, 10.0)]
        rng_seed0 = _FakeRng(integers_values=[0], uniform_values=[2.0])
        rng_seed1 = _FakeRng(integers_values=[0, 0], uniform_values=[0.5, 6.75])

        with mock.patch(
                'sim_data_analyzer.mask_utils.np.random.default_rng',
                side_effect=[rng_seed0, rng_seed1],
        ):
            sampled, successful_seed = sample_control_intervals(
                mask1_intervals,
                mask0_intervals,
                seed=0,
                max_seed_tries=2,
            )

        self.assertEqual(successful_seed, 1)
        self.assertEqual(len(sampled), 2)
        self.assertAlmostEqual(sampled[0][1] - sampled[0][0], 6.0)
        self.assertAlmostEqual(sampled[1][1] - sampled[1][0], 3.0)

    def test_sample_control_intervals_raises_when_impossible(self):
        with self.assertRaises(RuntimeError):
            sample_control_intervals(
                [(0.0, 5.0), (0.0, 4.0)],
                [(10.0, 16.0)],
                seed=0,
                max_seed_tries=5,
            )

    def test_sample_control_intervals_by_channel_uses_shared_seed(self):
        mask1_by_channel = {
            'y_0': [(0.0, 2.0)],
            'y_200': [(0.0, 1.0), (0.0, 1.5)],
        }
        mask0_by_channel = {
            'y_0': [(10.0, 20.0)],
            'y_200': [(30.0, 40.0)],
        }

        sampled, successful_seed = sample_control_intervals_by_channel(
            mask1_by_channel,
            mask0_by_channel,
            seed=3,
            max_seed_tries=10,
        )

        self.assertGreaterEqual(successful_seed, 3)
        self.assertEqual(list(sampled.keys()), ['y_0', 'y_200'])
        self.assertEqual(len(sampled['y_0']), 1)
        self.assertEqual(len(sampled['y_200']), 2)
        for key, source_intervals in mask1_by_channel.items():
            for source, control in zip(source_intervals, sampled[key]):
                self.assertAlmostEqual(control[1] - control[0], source[1] - source[0])

    def test_extract_mask_intervals_accepts_time_and_y_time_xarray(self):
        single = xr.DataArray(
            np.array([1, 1, 0], dtype=np.uint8),
            dims=('time',),
            coords={'time': [0.0, 0.5, 1.0]},
        )
        multi = xr.DataArray(
            np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8),
            dims=('y', 'time'),
            coords={'y': [100.0, 300.0], 'time': [0.0, 0.5, 1.0]},
        )

        single_intervals = extract_mask_intervals(single)
        multi_intervals = extract_mask_intervals(multi)

        self.assertEqual(single_intervals['mask_1'], [(0.0, 0.5)])
        self.assertEqual(multi_intervals['mask_1']['y_100'], [(0.0, 0.0)])
        self.assertEqual(multi_intervals['mask_1']['y_300'], [(0.5, 1.0)])


if __name__ == '__main__':
    unittest.main()
