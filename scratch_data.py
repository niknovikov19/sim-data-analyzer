"""Shared helpers for scratch scripts that reuse cached xarray artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

from sim_data_analyzer.xr_adapters import get_lfp_xr, get_net_rate_dynamics_xr
from sim_data_analyzer.xr_io import load_xr, save_xr


def get_exp_label(fpath_sim_result) -> str:
    """Build the scratch experiment label from a raw result path."""
    fpath_sim_result = Path(fpath_sim_result)
    return f'{fpath_sim_result.parent.name}_0'


def get_proc_dir(fpath_sim_result, dirpath_proc_root) -> Path:
    """Get the scratch processing-cache directory for one experiment."""
    return Path(dirpath_proc_root) / get_exp_label(fpath_sim_result)


def get_lfp_cache_path(fpath_sim_result, dirpath_proc_root) -> Path:
    """Get the cached LFP artifact path for one experiment."""
    exp_label = get_exp_label(fpath_sim_result)
    return get_proc_dir(fpath_sim_result, dirpath_proc_root) / f'{exp_label}_lfp.nc'


def get_rates_cache_path(fpath_sim_result, dirpath_proc_root, rate_dt: float) -> Path:
    """Get the cached population-rate artifact path for one experiment."""
    exp_label = get_exp_label(fpath_sim_result)
    return get_proc_dir(fpath_sim_result, dirpath_proc_root) / f'{exp_label}_rates_dt_{rate_dt:g}.nc'


def load_sim_result(fpath_sim_result) -> dict:
    """Load a pickled NetPyNE simulation result."""
    fpath_sim_result = Path(fpath_sim_result)
    with fpath_sim_result.open('rb') as fobj:
        return pickle.load(fobj)


def load_or_extract_lfp(sim_result: dict | None, fpath_sim_result, dirpath_proc_root):
    """Load cached LFP or extract it from the raw simulation result."""
    fpath_cache = get_lfp_cache_path(fpath_sim_result, dirpath_proc_root)
    if fpath_cache.exists():
        print(f'Loading cached LFP: {fpath_cache}')
        return load_xr(fpath_cache, load=True)

    if sim_result is None:
        raise ValueError('sim_result should be provided when LFP cache is missing')

    print('Extracting LFP from simulation result')
    lfp = get_lfp_xr(sim_result)
    save_xr(lfp, fpath_cache)
    return lfp


def load_or_extract_rates(
        sim_result: dict | None,
        fpath_sim_result,
        dirpath_proc_root,
        rate_dt: float = 5e-3,
        ):
    """Load cached population rate dynamics or extract them."""
    fpath_cache = get_rates_cache_path(fpath_sim_result, dirpath_proc_root, rate_dt)
    if fpath_cache.exists():
        print(f'Loading cached rates: {fpath_cache}')
        return load_xr(fpath_cache, load=True)

    if sim_result is None:
        raise ValueError('sim_result should be provided when rate cache is missing')

    print('Extracting population rate dynamics from simulation result')
    rates = get_net_rate_dynamics_xr(
        sim_result,
        dt_bin=rate_dt,
        avg_cells=True,
    )
    save_xr(rates, fpath_cache)
    return rates
