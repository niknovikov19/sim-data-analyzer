"""Summarize pop-vs-LFP/CSD CPSD into a per-population pandas table."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DIR_PACKAGE = Path(__file__).resolve().parent.parent
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.xr_adapters import get_lfp_xr, get_net_rate_dynamics_xr
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_signal import interp_time_outliers
from sim_data_analyzer.xr_spect import calc_xr_cpsd


# Data source and cache locations
FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_15s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
EXP_LABEL = f'{FPATH_SIM_RESULT.parent.name}_0'
DIRPATH_PROC = DIRPATH_PROC_ROOT / EXP_LABEL
DIRPATH_OUT = DIRPATH_PROC / 'rates_lfp_cpsd'

# Time limits used for all downstream processing
T_LIMITS = (10.0, 15.0)

# Rate extraction parameters
RATE_DT = 5e-3

# Reference signal parameters
SIGNAL_KIND = 'lfp'  # 'lfp' or 'csd'
CHANNEL_Y = 600.0
CLEAN_LFP_OUTLIERS = True
LFP_OUTLIER_Z_THRESH = 8.0
LFP_OUTLIER_REL_NEIGHBOR_THRESH = 5.0

# CPSD parameters
WIN_LEN = 1.0
WIN_OVERLAP = 0.5
FMAX = 100.0
FBAND = (8.0, 14.0)
AMPLITUDE_MULT = 100.0
ROUND_NDIGITS = 2


def _make_output_path() -> Path:
    """Build the CSV output path from the current script parameters."""
    round_part = '' if ROUND_NDIGITS is None else f'_round_{ROUND_NDIGITS}'
    fname_table = (
        f'rates_vs_{SIGNAL_KIND.lower()}_y_{CHANNEL_Y:g}'
        f'_dt_{RATE_DT:g}_fband_{FBAND[0]:g}_{FBAND[1]:g}'
        f'_ampx{AMPLITUDE_MULT:g}{round_part}.csv'
    )
    return DIRPATH_OUT / fname_table


def _load_sim_result(fpath: Path) -> dict:
    """Load a pickled NetPyNE simulation result."""
    with fpath.open('rb') as fobj:
        return pickle.load(fobj)


def _load_or_extract_lfp(sim_result: dict | None):
    """Load cached LFP or extract it from the raw simulation result."""
    fpath_cache = DIRPATH_PROC / f'{EXP_LABEL}_lfp.nc'
    if fpath_cache.exists():
        print(f'Loading cached LFP: {fpath_cache}')
        return load_xr(fpath_cache, load=True)

    if sim_result is None:
        raise ValueError('sim_result should be provided when LFP cache is missing')

    print('Extracting LFP from simulation result')
    lfp = get_lfp_xr(sim_result)
    save_xr(lfp, fpath_cache)
    return lfp


def _load_or_extract_rates(sim_result: dict | None):
    """Load cached population rate dynamics or extract them."""
    fpath_cache = DIRPATH_PROC / f'{EXP_LABEL}_rates_dt_{RATE_DT:g}.nc'
    if fpath_cache.exists():
        print(f'Loading cached rates: {fpath_cache}')
        return load_xr(fpath_cache, load=True)

    if sim_result is None:
        raise ValueError('sim_result should be provided when rate cache is missing')

    print('Extracting population rate dynamics from simulation result')
    rates = get_net_rate_dynamics_xr(
        sim_result,
        dt_bin=RATE_DT,
        avg_cells=True,
    )
    save_xr(rates, fpath_cache)
    return rates


def _prepare_reference_signal(lfp) -> tuple[object, float]:
    """Prepare the selected LFP/CSD reference signal."""
    signal_kind = SIGNAL_KIND.lower()
    signal_xr = lfp.sel(time=slice(*T_LIMITS))

    if CLEAN_LFP_OUTLIERS:
        signal_xr = interp_time_outliers(
            signal_xr,
            z_thresh=LFP_OUTLIER_Z_THRESH,
            rel_neighbor_thresh=LFP_OUTLIER_REL_NEIGHBOR_THRESH,
        )

    if signal_kind == 'csd':
        signal_xr = calc_xr_csd(signal_xr, store_proc_info=True)
    elif signal_kind != 'lfp':
        raise ValueError(f'Unsupported SIGNAL_KIND {SIGNAL_KIND!r}')

    if 'y' not in signal_xr.dims:
        raise ValueError('Expected a y-dimension in the selected signal')
    if float(CHANNEL_Y) not in set(map(float, signal_xr.y.values.tolist())):
        raise ValueError(
            f'CHANNEL_Y {CHANNEL_Y:g} is not present in available depths '
            f'{list(map(float, signal_xr.y.values.tolist()))}'
        )

    ref_trace = signal_xr.sel(y=CHANNEL_Y)
    channel_y = float(ref_trace.coords['y'].item())
    ref_trace = ref_trace - ref_trace.mean(skipna=True)
    return ref_trace, channel_y


def _select_analysis_pops(rates):
    """Drop frozen populations from the rate-dynamics analysis."""
    pop_names = [pop_name for pop_name in rates.pop.values.tolist() if 'frz' not in pop_name]
    return rates.sel(pop=pop_names)


def _align_reference_to_rates(ref_trace, rates):
    """Interpolate the reference trace onto the rate-dynamics time grid."""
    ref_aligned = ref_trace.interp(time=rates.time)
    ref_aligned = ref_aligned - ref_aligned.mean(skipna=True)
    return ref_aligned


def _make_nan_spectrum(template, pop_name: str):
    """Create a NaN CPSD spectrum for a population."""
    values = np.full(template.shape, np.nan + 0j, dtype=np.complex128)
    W = xr.DataArray(values, dims=template.dims, coords=template.coords)
    W.attrs = template.attrs.copy()
    W = W.expand_dims(pop=[pop_name])
    return W


def _summarize_cpsd(W, pop_name: str) -> dict:
    """Summarize a complex CPSD spectrum over the selected frequency band."""
    fmask = (W.freq >= FBAND[0]) & (W.freq <= FBAND[1])
    row = {'pop': pop_name, 'cpsd_amp': np.nan, 'cpsd_phase': np.nan}

    if int(np.count_nonzero(fmask.values)) == 0:
        return row

    W_band = W.where(fmask, drop=True)
    band_mean = complex(W_band.mean().item())
    amp_mean = float(np.abs(W_band).mean().item()) * AMPLITUDE_MULT

    row['cpsd_amp'] = amp_mean
    if np.isfinite(np.real(band_mean)) and np.isfinite(np.imag(band_mean)):
        row['cpsd_phase'] = float(np.angle(band_mean))
    if ROUND_NDIGITS is not None:
        for key in ('cpsd_amp', 'cpsd_phase'):
            if pd.notna(row[key]):
                row[key] = round(row[key], ROUND_NDIGITS)
    return row


def main() -> None:
    """Run the per-population CPSD summary workflow."""
    sim_result = None

    need_raw = not (DIRPATH_PROC / f'{EXP_LABEL}_lfp.nc').exists()
    need_raw = need_raw or not (DIRPATH_PROC / f'{EXP_LABEL}_rates_dt_{RATE_DT:g}.nc').exists()
    if need_raw:
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = _load_sim_result(FPATH_SIM_RESULT)

    lfp = _load_or_extract_lfp(sim_result)
    rates = _load_or_extract_rates(sim_result)

    rates = rates.sel(time=slice(*T_LIMITS)).load()
    rates = _select_analysis_pops(rates)
    ref_trace, _channel_y = _prepare_reference_signal(lfp)
    ref_trace = _align_reference_to_rates(ref_trace, rates)

    template = calc_xr_cpsd(
        ref_trace,
        ref_trace,
        win_len=WIN_LEN,
        win_overlap=WIN_OVERLAP,
        fmax=FMAX,
        compute=True,
        store_proc_info=True,
    )

    rows = []
    for pop_name in rates.pop.values.tolist():
        rate_trace = rates.sel(pop=pop_name)

        if not np.isfinite(rate_trace.values).any():
            W = _make_nan_spectrum(template, pop_name)
            rows.append(_summarize_cpsd(W.isel(pop=0), pop_name))
            continue

        try:
            W = calc_xr_cpsd(
                rate_trace,
                ref_trace,
                win_len=WIN_LEN,
                win_overlap=WIN_OVERLAP,
                fmax=FMAX,
                compute=True,
                store_proc_info=True,
            )
        except Exception as exc:
            print(f'Warning: failed to compute CPSD for {pop_name}: {exc}')
            W = template.copy(data=np.full(template.shape, np.nan + 0j, dtype=np.complex128))

        rows.append(_summarize_cpsd(W, pop_name))

    df = pd.DataFrame(rows)
    df = df.sort_values('pop').reset_index(drop=True)
    print(df)

    fpath_table = _make_output_path()
    fpath_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fpath_table, index=False)
    print(f'Saved summary table: {fpath_table}')


if __name__ == '__main__':
    main()
