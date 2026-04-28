"""
Collected low-level processing helpers for NetPyNE-derived spike data.

This module starts from the shared A1/model_tuner rate/CV helper overlap and
also includes the A1 low-level rate-dynamics helpers. Small safety checks from
the A1 lineage are preserved.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def calc_pop_rate(
        pop_spikes: List[np.ndarray],  # per cells or combined into 1 entry
        time_limits: Tuple[float],
        ncells: int = 1
        ) -> float | List[float]:
    """Calculate population firing rate."""
    if (len(pop_spikes) > 1) and (ncells != 1):
        raise ValueError('If pop_spikes contains many elements, ncells should be 1')
    rates = []
    T = time_limits[1] - time_limits[0]
    for spike_times in pop_spikes:
        nspikes = np.sum((spike_times >= time_limits[0]) &
                         (spike_times <= time_limits[1]))
        rates.append(nspikes / T / ncells)
    if len(rates) == 1:
        rates = rates[0]
    return rates


def calc_net_rates(
        net_spikes: Dict[str, List[np.ndarray]],  # {pop: spikes}
        time_limits: Tuple[float],
        ncells: Dict[str, int] | None = None,
        pop_names: List[str] | None = None
        ) -> Dict[str, float | List[float]]:  # {pop: rates}
    net_rates = {}
    pop_names = pop_names or list(net_spikes)
    ncells = ncells or {pop_name: 1 for pop_name in pop_names}
    for pop_name in pop_names:
        net_rates[pop_name] = calc_pop_rate(
            net_spikes[pop_name], time_limits, ncells[pop_name]
        )
    return net_rates


def calc_pop_cv(
        pop_spikes: List[np.ndarray],  # per cells
        time_limits: Tuple[float],
        nspikes_min: int = 3,   # min. number of spikes to compute CV for a cell
        avg_result: bool = True
        ) -> float | List[float]:
    """Calculate population CV."""
    cvs = []
    for spike_times in pop_spikes:
        mask = ((spike_times >= time_limits[0]) & (spike_times <= time_limits[1]))
        s = spike_times[mask]
        if len(s) < nspikes_min:
            continue
        isi = s[1:] - s[:-1]
        cvs.append(np.std(isi) / np.mean(isi))
    cvs = np.array(cvs)
    if avg_result:
        return cvs.mean()
    else:
        return cvs


def calc_net_cvs(
        net_spikes: Dict[str, List[np.ndarray]],  # {pop: spikes}
        time_limits: Tuple[float],
        nspikes_min: int = 3,   # min. number of spikes to compute CV for a cell
        avg_result: bool = True,
        pop_names: List[str] | None = None,
        ) -> Dict[str, float | List[float]]:  # {pop: rates}
    net_cvs = {}
    pop_names = pop_names or list(net_spikes)
    for pop_name in pop_names:
        net_cvs[pop_name] = calc_pop_cv(
            net_spikes[pop_name], time_limits,
            nspikes_min, avg_result
        )
    return net_cvs


def calc_pop_rate_dynamics(
        pop_spikes: list[np.ndarray],  # per cells or combined into 1 entry
        time_range: tuple[float, float],
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        ncells: int = 1,
        epoch_len: float | None = None
        ) -> tuple[np.ndarray, list[np.ndarray] | np.ndarray]:
    """Calculate firing rate dynamics from spike trains."""
    t1 = time_range[0]
    t2 = time_range[1]

    if epoch_len is not None:
        num_epochs = np.floor((time_range[1] - time_range[0]) / epoch_len)
        t2 = t1 + epoch_len * num_epochs
    else:
        num_epochs = 1

    rvecs = []
    for spike_times in pop_spikes:
        mask = (spike_times >= t1) & (spike_times <= t2)
        spike_times = spike_times[mask]

        if epoch_len is not None:
            spike_times = ((spike_times - t1) % epoch_len) + t1
            t2 = t1 + epoch_len

        Nbins = int((t2 - t1) / dt_bin)
        bin_idx = np.floor((spike_times - t1) / dt_bin)
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < Nbins)]
        bin_idx = bin_idx.astype(np.int64)

        rvec = np.bincount(bin_idx, minlength=Nbins)
        rvec = rvec / (dt_bin * ncells * num_epochs)
        if tau_smooth is not None:
            sigma = tau_smooth / dt_bin
            rvec = gaussian_filter1d(rvec, sigma=sigma)
        rvecs.append(rvec)

    if len(rvecs) == 1:
        rvecs = rvecs[0]

    tvec = np.arange(Nbins, dtype=np.float64) * dt_bin + t1
    return tvec, rvecs


def calc_net_rate_dynamics(
        net_spikes: Dict[str, List[np.ndarray]],  # {pop: spikes}
        time_range: Tuple[float, float],
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        ncells: Dict[str, int] | None = None,
        pop_names: List[str] | None = None,
        epoch_len: float | None = None,
        ) -> dict[str, tuple[np.ndarray, list[np.ndarray] | np.ndarray]]:
    net_rate_dynamics = {}
    pop_names = pop_names or list(net_spikes)
    ncells = ncells or {pop_name: 1 for pop_name in pop_names}
    for pop_name in pop_names:
        tvec, rvecs = calc_pop_rate_dynamics(
            net_spikes[pop_name], time_range, dt_bin,
            tau_smooth, ncells[pop_name], epoch_len
        )
        net_rate_dynamics[pop_name] = (tvec, rvecs)
    return net_rate_dynamics
