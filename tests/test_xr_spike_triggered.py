import unittest

import numpy as np
import xarray as xr

from sim_data_analyzer.spike_data import SpikeData, _SpikeMeta
from sim_data_analyzer.xr_spike_triggered import (
    calc_xr_sta,
    extract_xr_spike_triggered_epochs,
)


def _make_combined_spikes():
    spikes_by_pop = {
        'IT2': [np.array([0.2, 0.5, 0.8])],
        'PV2': [np.array([0.3, 0.7])],
    }
    return SpikeData(
        spikes_by_pop,
        meta=_SpikeMeta(
            combine=True, t0=0.0, tmax=1.0, subtract_t0=True, ms=False, ndigits=6
        ),
        pop_sizes={'IT2': 2, 'PV2': 1},
    )


def _make_per_cell_spikes():
    spikes_by_pop = {
        'IT2': [np.array([0.2, 0.8]), np.array([0.5])],
        'PV2': [np.array([0.3, 0.7])],
    }
    cell_gids_by_pop = {
        'IT2': np.array([10, 11]),
        'PV2': np.array([20]),
    }
    return SpikeData(
        spikes_by_pop,
        meta=_SpikeMeta(
            combine=False, t0=0.0, tmax=1.0, subtract_t0=True, ms=False, ndigits=6
        ),
        cell_gids_by_pop=cell_gids_by_pop,
        pop_sizes={'IT2': 2, 'PV2': 1},
    )


def _make_signal_1d():
    tt = np.arange(0.0, 1.0, 0.1)
    values = np.arange(len(tt), dtype=float)
    return xr.DataArray(values, dims=['time'], coords={'time': tt}, attrs={'source': 'test'})


def _make_signal_pop():
    tt = np.arange(0.0, 1.0, 0.1)
    values = np.vstack([
        np.arange(len(tt), dtype=float),
        np.arange(len(tt), dtype=float) * 10.0,
    ])
    return xr.DataArray(
        values,
        dims=['pop', 'time'],
        coords={'pop': ['IT2', 'PV2'], 'time': tt},
        attrs={'source': 'test'},
    )


def _make_signal_cell():
    tt = np.arange(0.0, 1.0, 0.1)
    values = np.stack([
        np.arange(len(tt), dtype=float),
        np.arange(len(tt), dtype=float) + 100.0,
    ])
    return xr.DataArray(
        values,
        dims=['cell_gid', 'time'],
        coords={'cell_gid': [10, 11], 'time': tt},
        attrs={'source': 'test'},
    )


def _make_signal_pop_cell():
    tt = np.arange(0.0, 1.0, 0.1)
    values = np.array([
        [
            np.arange(len(tt), dtype=float),
            np.arange(len(tt), dtype=float) + 100.0,
        ],
        [
            np.arange(len(tt), dtype=float) * 10.0,
            np.arange(len(tt), dtype=float) * 10.0 + 1000.0,
        ],
    ])
    return xr.DataArray(
        values,
        dims=['pop', 'cell_gid', 'time'],
        coords={'pop': ['IT2', 'PV2'], 'cell_gid': [10, 11], 'time': tt},
        attrs={'source': 'test'},
    )


class TestXRSpikeTriggered(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.combined_spikes = _make_combined_spikes()
        cls.per_cell_spikes = _make_per_cell_spikes()
        cls.X1 = _make_signal_1d()
        cls.X_pop = _make_signal_pop()
        cls.X_cell = _make_signal_cell()
        cls.X_pop_cell = _make_signal_pop_cell()
        cls.time_win = (-0.1, 0.2)

    def test_combined_avg_on_1d_signal(self):
        out = calc_xr_sta(
            self.X1, self.combined_spikes, self.time_win,
            pop_name='IT2', return_mode='avg'
        )
        expected = np.array([2.5, 3.5, 4.5, 5.5])
        np.testing.assert_allclose(out.values, expected)
        self.assertEqual(out.dims, ('time_rel',))

    def test_per_cell_spikes_are_pooled_for_1d_signal(self):
        out = calc_xr_sta(
            self.X1, self.per_cell_spikes, self.time_win,
            pop_name='IT2', return_mode='avg'
        )
        expected = np.array([2.5, 3.5, 4.5, 5.5])
        np.testing.assert_allclose(out.values, expected)

    def test_broadcast_over_cell_dimension(self):
        out = calc_xr_sta(
            self.X_cell, self.combined_spikes, self.time_win,
            pop_name='IT2', return_mode='avg'
        )
        expected = np.array([
            [2.5, 3.5, 4.5, 5.5],
            [102.5, 103.5, 104.5, 105.5],
        ])
        np.testing.assert_allclose(out.values, expected)
        self.assertEqual(out.dims, ('cell_gid', 'time_rel'))

    def test_auto_pop_mapping_returns_pop_dimension(self):
        out = calc_xr_sta(
            self.X_pop, self.combined_spikes, self.time_win,
            return_mode='avg'
        )
        self.assertEqual(out.dims, ('pop', 'time_rel'))
        np.testing.assert_allclose(out.sel(pop='IT2').values, np.array([2.5, 3.5, 4.5, 5.5]))
        np.testing.assert_allclose(out.sel(pop='PV2').values, np.array([40.0, 50.0, 60.0, 70.0]))

    def test_epochs_return_dict_for_multi_pop(self):
        out = extract_xr_spike_triggered_epochs(
            self.X_pop, self.combined_spikes, self.time_win
        )
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {'IT2', 'PV2'})
        self.assertEqual(out['IT2'].dims, ('spike', 'time_rel'))

    def test_return_mode_both(self):
        out = calc_xr_sta(
            self.X_pop, self.combined_spikes, self.time_win,
            return_mode='both'
        )
        self.assertEqual(set(out), {'avg', 'epochs'})
        self.assertIsInstance(out['epochs'], dict)
        self.assertEqual(out['avg'].dims, ('pop', 'time_rel'))

    def test_time_units_conversion_from_ms_spikes(self):
        spikes_ms = SpikeData(
            {'IT2': [np.array([200.0, 500.0, 800.0])]},
            meta=_SpikeMeta(
                combine=True, t0=0.0, tmax=1000.0, subtract_t0=True, ms=True, ndigits=3
            ),
            pop_sizes={'IT2': 1},
        )
        out = calc_xr_sta(
            self.X1, spikes_ms, self.time_win,
            pop_name='IT2', time_units='s', return_mode='avg'
        )
        np.testing.assert_allclose(out.values, np.array([2.5, 3.5, 4.5, 5.5]))

    def test_dropped_boundary_spikes(self):
        spikes = SpikeData(
            {'IT2': [np.array([0.05, 0.2, 0.5, 0.95])]},
            meta=_SpikeMeta(
                combine=True, t0=0.0, tmax=1.0, subtract_t0=True, ms=False, ndigits=6
            ),
            pop_sizes={'IT2': 1},
        )
        epochs = extract_xr_spike_triggered_epochs(
            self.X1, spikes, self.time_win, pop_name='IT2'
        )
        self.assertEqual(epochs.sizes['spike'], 2)

    def test_empty_epochs_behavior(self):
        spikes = SpikeData(
            {'IT2': [np.array([])]},
            meta=_SpikeMeta(
                combine=True, t0=0.0, tmax=1.0, subtract_t0=True, ms=False, ndigits=6
            ),
            pop_sizes={'IT2': 1},
        )
        epochs = extract_xr_spike_triggered_epochs(
            self.X1, spikes, self.time_win, pop_name='IT2'
        )
        self.assertEqual(epochs.sizes['spike'], 0)
        avg = calc_xr_sta(self.X1, spikes, self.time_win, pop_name='IT2', return_mode='avg')
        self.assertTrue(np.isnan(avg.values).all())

    def test_store_proc_info_true(self):
        out = calc_xr_sta(
            self.X1, self.combined_spikes, self.time_win,
            pop_name='IT2', return_mode='avg', store_proc_info=True
        )
        self.assertIn('proc_steps', out.attrs)
        self.assertEqual(out.attrs['proc_steps'][-1]['name'], 'calc_xr_sta')

    def test_missing_pop_name_without_pop_dim_raises(self):
        with self.assertRaises(ValueError):
            calc_xr_sta(self.X1, self.combined_spikes, self.time_win)

    def test_unknown_pop_name_raises(self):
        with self.assertRaises(ValueError):
            calc_xr_sta(self.X1, self.combined_spikes, self.time_win, pop_name='NOPE')

    def test_unsupported_time_units_raises(self):
        with self.assertRaises(ValueError):
            calc_xr_sta(
                self.X1, self.combined_spikes, self.time_win,
                pop_name='IT2', time_units='sec'
            )

    def test_non_monotonic_time_raises(self):
        X = self.X1.assign_coords(time=np.array([0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        with self.assertRaises(ValueError):
            calc_xr_sta(X, self.combined_spikes, self.time_win, pop_name='IT2')

    def test_irregular_time_raises(self):
        X = self.X1.assign_coords(time=np.array([0.0, 0.1, 0.2, 0.31, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        with self.assertRaises(ValueError):
            calc_xr_sta(X, self.combined_spikes, self.time_win, pop_name='IT2')


if __name__ == '__main__':
    unittest.main()
