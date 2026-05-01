"""Demo for lag-windowed xr cross-correlograms on rate and LFP traces."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.scratch_data import (
    get_exp_label,
    get_lfp_cache_path,
    get_proc_dir,
    get_rates_cache_path,
    load_or_extract_lfp,
    load_or_extract_rates,
    load_sim_result,
)
from sim_data_analyzer.xr_signal import calc_xr_crosscorr, interp_time_outliers


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_15s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
EXP_LABEL = get_exp_label(FPATH_SIM_RESULT)
DIRPATH_PROC = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
DIRPATH_OUT = DIRPATH_PROC / 'crosscorr_demo'

T_LIMITS = (10.0, 15.0)
RATE_DT = 5e-3
CHANNEL_Y = 600.0
LAG_WINDOW = (-0.5, 0.5)
POP_NAME = None

CLEAN_LFP_OUTLIERS = True
LFP_OUTLIER_Z_THRESH = 8.0
LFP_OUTLIER_REL_NEIGHBOR_THRESH = 5.0


def _prepare_reference_signal(lfp):
    ref_trace = lfp.sel(time=slice(*T_LIMITS))
    if CLEAN_LFP_OUTLIERS:
        ref_trace = interp_time_outliers(
            ref_trace,
            z_thresh=LFP_OUTLIER_Z_THRESH,
            rel_neighbor_thresh=LFP_OUTLIER_REL_NEIGHBOR_THRESH,
        )

    if 'y' not in ref_trace.dims:
        raise ValueError('Expected a y-dimension in the LFP signal')
    if float(CHANNEL_Y) not in set(map(float, ref_trace.y.values.tolist())):
        raise ValueError(
            f'CHANNEL_Y {CHANNEL_Y:g} is not present in available depths '
            f'{list(map(float, ref_trace.y.values.tolist()))}'
        )

    ref_trace = ref_trace.sel(y=CHANNEL_Y)
    return ref_trace - ref_trace.mean(skipna=True)


def _select_pop_name(rates) -> str:
    if POP_NAME is not None:
        if POP_NAME not in rates.pop.values.tolist():
            raise ValueError(f'Population {POP_NAME!r} is not present in the rates data')
        return POP_NAME

    for pop_name in rates.pop.values.tolist():
        if 'frz' not in pop_name:
            return pop_name
    raise ValueError('Could not find a non-frozen population in the rates data')


def _peak_summary(label: str, corr) -> None:
    if corr.sizes['lag'] == 0 or not np.isfinite(corr.values).any():
        print(f'{label}: no finite lag values in the requested window')
        return

    peak_idx = int(np.nanargmax(np.abs(corr.values)))
    peak_lag = float(corr.coords['lag'].values[peak_idx])
    peak_val = float(corr.values[peak_idx])
    print(f'{label}: peak |corr| at lag={peak_lag:+.4f} s, value={peak_val:+.6g}')


def _make_plot(fpath_out: Path, raw_corr, demeaned_corr, normalized_corr, pop_name: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    traces = [
        ('Raw / N', raw_corr, '#1f6feb'),
        ('Demeaned / N', demeaned_corr, '#d29922'),
        ('Demeaned + normalized', normalized_corr, '#1a7f37'),
    ]

    for ax, (title, corr, color) in zip(axes, traces):
        ax.plot(corr.lag.values, corr.values, color=color, lw=2)
        ax.axvline(0.0, color='0.4', ls='--', lw=1)
        ax.set_ylabel('corr')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('lag (s)')
    fig.suptitle(
        f'Cross-correlogram demo: pop={pop_name}, y={CHANNEL_Y:g}, '
        f'window=[{LAG_WINDOW[0]:g}, {LAG_WINDOW[1]:g}] s'
    )
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    sim_result = None
    lfp_cache = get_lfp_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    rate_cache = get_rates_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    if (not lfp_cache.exists()) or (not rate_cache.exists()):
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = load_sim_result(FPATH_SIM_RESULT)

    lfp = load_or_extract_lfp(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    rates = load_or_extract_rates(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)

    rates = rates.sel(time=slice(*T_LIMITS)).load()
    ref_trace = _prepare_reference_signal(lfp)
    ref_trace = ref_trace.interp(time=rates.time)

    pop_name = _select_pop_name(rates)
    rate_trace = rates.sel(pop=pop_name)

    raw_corr = calc_xr_crosscorr(
        rate_trace,
        ref_trace,
        lag_window=LAG_WINDOW,
        subtract_mean=False,
        normalize=False,
        compute=True,
        store_proc_info=True,
    )
    demeaned_corr = calc_xr_crosscorr(
        rate_trace,
        ref_trace,
        lag_window=LAG_WINDOW,
        subtract_mean=True,
        normalize=False,
        compute=True,
        store_proc_info=True,
    )
    normalized_corr = calc_xr_crosscorr(
        rate_trace,
        ref_trace,
        lag_window=LAG_WINDOW,
        subtract_mean=True,
        normalize=True,
        compute=True,
        store_proc_info=True,
    )

    _peak_summary('Raw / N', raw_corr)
    _peak_summary('Demeaned / N', demeaned_corr)
    _peak_summary('Demeaned + normalized', normalized_corr)

    DIRPATH_OUT.mkdir(parents=True, exist_ok=True)
    fpath_out = DIRPATH_OUT / f'crosscorr_{pop_name}_y_{CHANNEL_Y:g}.png'
    _make_plot(fpath_out, raw_corr, demeaned_corr, normalized_corr, pop_name)
    print(f'Saved plot: {fpath_out}')


if __name__ == '__main__':
    main()
