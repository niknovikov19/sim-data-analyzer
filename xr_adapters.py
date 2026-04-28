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
    get_pop_lfps,
    get_pop_names,
)


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
