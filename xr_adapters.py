"""
Xarray adapters for NetPyNE result traces.

These helpers are collected separately from the parser core because they define
an xarray-facing representation layer on top of the raw NetPyNE result dict.
The current pass combines the A1 trace adapters with the sim_res_analyzer LFP
conversion helpers.
"""

from typing import Dict, Tuple

import numpy as np
import xarray as xr

from sim_data_analyzer.netpyne_res_parse_utils import (
    get_lfp,
    get_net_size,
    get_pop_cell_gids,
    get_pop_lfps,
    get_pop_names,
    get_pop_size,
    get_pop_spikes,
    get_sim_duration,
)
from sim_data_analyzer.data_proc_utils import (
    calc_net_rate_dynamics,
    calc_pop_rate_dynamics,
)


def _make_rate_tvec(time_range: Tuple[float, float], dt_bin: float) -> np.ndarray:
    """Create a time vector aligned with calc_pop_rate_dynamics(). """
    t1, t2 = time_range
    nbins = int((t2 - t1) / dt_bin)
    return np.arange(nbins, dtype=np.float64) * dt_bin + t1


def get_trace_xr(
        sim_result: Dict,
        trace_name: str,
        t_limits: Tuple[float, float] | None = None,
        ms: bool = True
        ) -> Dict[str, xr.Dataset]:
    """Returns a dict: {pop: X (cells x time)}, where X is a recorded trace. """

    pop_names = get_pop_names(sim_result)

    # Time bins and limits
    tvec = np.array(sim_result['simData']['t'])
    if not ms:
        tvec /= 1000
    if t_limits is not None:
        tmask = (tvec >= t_limits[0]) & (tvec <= t_limits[1])
        tvec = tvec[tmask]
    else:
        tmask = np.ones_like(tvec, dtype=bool)

    # Extract traces and the corresponding cell gids
    X_data = {pop: [] for pop in pop_names}
    cell_gids = {pop: [] for pop in pop_names}
    for cell, X_vec in sim_result['simData'][trace_name].items():
        gid = int(cell.split('_')[-1])
        pop = sim_result['net']['cells'][gid]['tags']['pop']
        X_data[pop].append(np.array(X_vec)[tmask])
        cell_gids[pop].append(gid)

    # Convert to xarray
    for pop, X_ in X_data.items():
        if len(X_) == 0:
            X_data[pop] = None
            continue
        X_data[pop] = xr.DataArray(
            np.array(X_),
            dims=['cell_gid', 'time'],
            coords={
                'cell_gid': cell_gids[pop],
                'time': tvec
            }
        )

    return X_data


def get_voltages_xr(
        sim_result: Dict,
        t_limits: Tuple[float, float] | None = None,
        ms: bool = True
        ) -> Dict[str, xr.Dataset]:
    """Returns a dict: {pop: Vmat (cells x time)}. """

    return get_trace_xr(sim_result, 'V_soma', t_limits, ms)


def get_lfp_xr(sim_result: Dict) -> xr.DataArray:
    """Convert NetPyNE LFP output to an xarray DataArray."""

    lfp, tt, lfp_coords = get_lfp(sim_result)
    dims = ['y', 'time']
    coords = {'y': lfp_coords[:, 1], 'time': tt / 1000}
    return xr.DataArray(lfp, dims=dims, coords=coords)


def get_pop_lfps_xr(sim_result: Dict) -> xr.DataArray:
    """Convert NetPyNE population LFP output to an xarray DataArray."""

    lfp, tt, lfp_coords = get_pop_lfps(sim_result)
    dims = ['pop', 'y', 'time']
    pop_names = list(lfp.keys())
    coords = {'pop': pop_names, 'y': lfp_coords[:, 1], 'time': tt / 1000}
    sz = [len(coord) for coord in coords.values()]
    X = xr.DataArray(np.full(sz, np.nan), dims=dims, coords=coords)
    for pop in pop_names:
        X.loc[{'pop': pop}] = lfp[pop]
    return X


def get_pop_rate_dynamics_xr(
        sim_result: Dict,
        pop_name: str,
        t_limits: Tuple[float, float | None] = (0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        avg_cells: bool = True
        ) -> xr.DataArray | None:
    """Convert population rate dynamics to an xarray DataArray."""

    t_limits = list(t_limits)
    if t_limits[1] is None:
        t_limits[1] = get_sim_duration(sim_result)

    if avg_cells:
        pop_spikes = get_pop_spikes(
            sim_result,
            pop_name,
            combine_cells=True,
            t0=t_limits[0],
            tmax=t_limits[1],
            subtract_t0=False,
            ms=False,
        )
        pop_size = get_pop_size(sim_result, pop_name)
        if pop_size == 0:
            tvec = _make_rate_tvec(tuple(t_limits), dt_bin)
            values = np.full(len(tvec), np.nan)
        else:
            tvec, values = calc_pop_rate_dynamics(
                pop_spikes, tuple(t_limits), dt_bin, tau_smooth, ncells=pop_size
            )
        return xr.DataArray(values, dims=['time'], coords={'time': tvec})

    pop_spikes = get_pop_spikes(
        sim_result,
        pop_name,
        combine_cells=False,
        t0=t_limits[0],
        tmax=t_limits[1],
        subtract_t0=False,
        ms=False,
    )
    cell_gids = get_pop_cell_gids(sim_result, pop_name)
    if len(cell_gids) == 0:
        return None

    tvec, rvecs = calc_pop_rate_dynamics(
        pop_spikes, tuple(t_limits), dt_bin, tau_smooth
    )
    return xr.DataArray(
        np.array(rvecs),
        dims=['cell_gid', 'time'],
        coords={'cell_gid': cell_gids, 'time': tvec},
    )


def get_net_rate_dynamics_xr(
        sim_result: Dict,
        t_limits: Tuple[float, float | None] = (0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        avg_cells: bool = True,
        pop_names: list[str] | tuple[str, ...] | None = None,
        ) -> xr.DataArray:
    """Convert network population rate dynamics to an xarray DataArray."""

    if not avg_cells:
        raise ValueError('Per-cell network rate dynamics are not supported')

    t_limits = list(t_limits)
    if t_limits[1] is None:
        t_limits[1] = get_sim_duration(sim_result)

    pop_names = get_pop_names(sim_result) if pop_names is None else [str(pop_name) for pop_name in pop_names]
    ncells = get_net_size(sim_result)
    net_spikes = {
        pop_name: get_pop_spikes(
            sim_result,
            pop_name,
            combine_cells=True,
            t0=t_limits[0],
            tmax=t_limits[1],
            subtract_t0=False,
            ms=False,
        )
        for pop_name in pop_names
    }

    nonempty_pop_names = [pop_name for pop_name in pop_names if ncells[pop_name] > 0]
    if nonempty_pop_names:
        rate_dyn = calc_net_rate_dynamics(
            {pop_name: net_spikes[pop_name] for pop_name in nonempty_pop_names},
            tuple(t_limits),
            dt_bin,
            tau_smooth,
            {pop_name: ncells[pop_name] for pop_name in nonempty_pop_names},
            nonempty_pop_names,
        )
        tvec = rate_dyn[nonempty_pop_names[0]][0]
    else:
        tvec = _make_rate_tvec(tuple(t_limits), dt_bin)
        rate_dyn = {}

    R = xr.DataArray(
        np.full((len(pop_names), len(tvec)), np.nan),
        dims=['pop', 'time'],
        coords={'pop': pop_names, 'time': tvec},
    )
    for pop_name in nonempty_pop_names:
        R.loc[{'pop': pop_name}] = rate_dyn[pop_name][1]
    return R
