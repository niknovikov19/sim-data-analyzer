"""Per-population firing-rate PSD analysis with cached Welch spectra."""

from __future__ import annotations

import json
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
    get_proc_dir,
    get_rates_cache_path,
    load_or_extract_rates,
    load_sim_result,
)
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_spect import calc_xr_welch


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'

RESULT_GROUP = 'rate_psd'
PSD_CACHE_GROUP = 'rate_psd_cache'

POP_NAMES = 'all'
T_LIMITS_S = None
RATE_DT = 1e-3
PSD_WIN_LEN = 2.0
PSD_WIN_OVERLAP = 0.75
PSD_FMIN = 2.0
PSD_FMAX = 30.0
PSD_AVERAGE = 'mean'
NORMALIZE = 0
NORMALIZE_F_BAND = (5.0, 30.0)
SMOOTH_FREQ_BINS = 1
PLOT_F_LIMITS = (2.0, 50.0)

LOGX = 0
LOGY = 0
GROUP_PLOTS = 1

POP_GROUPS = {
    'IT': ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'IT5B', 'IT6'],
    'PYR': ['CT5A', 'CT5B', 'PT5B', 'CT6'],
    'PV': ['PV2', 'PV3', 'PV4', 'PV5A', 'PV5B', 'PV6'],
    'SOM': ['SOM2', 'SOM3', 'SOM4', 'SOM5A', 'SOM5B', 'SOM6'],
    'VIP': ['VIP2', 'VIP3', 'VIP4', 'VIP5A', 'VIP5B', 'VIP6'],
    'NGF': ['NGF1', 'NGF2', 'NGF3', 'NGF4', 'NGF5A', 'NGF5B', 'NGF6'],
    'THAL': ['TC', 'HTC', 'TI', 'IRE', 'TCM', 'TIM', 'IREM'],
}


def _encode_dataset_attr(value):
    """Convert one xarray attr value into a NetCDF-friendly scalar."""
    if value is None:
        return 'null'
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    return json.dumps(value, sort_keys=True)


def _make_netcdf_safe_dataarray(X):
    """Return a copy with attrs converted to NetCDF-friendly scalar values."""
    X_out = X.copy(deep=False)
    X_out.attrs = {
        str(key): _encode_dataset_attr(value)
        for key, value in dict(X.attrs).items()
    }
    return X_out


def _format_tag_value(value: float) -> str:
    """Format one numeric value into a filesystem-safe compact tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _format_interval_tag(values) -> str:
    """Format an optional numeric interval into a compact tag."""
    if values is None:
        return 'full'
    return f'{_format_tag_value(values[0])}_{_format_tag_value(values[1])}'


def _bool_tag(value: bool) -> str:
    """Return a compact integer tag for one boolean flag."""
    return str(int(bool(value)))


def _resolve_pop_names(all_pop_names, requested_pop_names=None) -> list[str]:
    """Resolve stable analysis populations after excluding frozen populations."""
    eligible = [str(pop_name) for pop_name in list(all_pop_names) if 'frz' not in str(pop_name)]
    if requested_pop_names is None:
        if not eligible:
            raise ValueError('No non-frozen populations are available for analysis')
        return eligible
    if isinstance(requested_pop_names, str):
        if requested_pop_names.strip().lower() == 'all':
            if not eligible:
                raise ValueError('No non-frozen populations are available for analysis')
            return eligible
        requested = [requested_pop_names]
    else:
        requested = [str(pop_name) for pop_name in list(requested_pop_names)]
        if len(requested) == 1 and requested[0].strip().lower() == 'all':
            if not eligible:
                raise ValueError('No non-frozen populations are available for analysis')
            return eligible
    missing = [pop_name for pop_name in requested if pop_name not in eligible]
    if missing:
        raise ValueError(
            'Requested populations are unavailable after excluding frozen populations: '
            + ', '.join(missing)
        )
    if not requested:
        raise ValueError('Population selection is empty after validation')
    return requested


def _resolve_pop_groups(pop_names, pop_groups=None) -> dict[str, list[str]]:
    """Resolve hardcoded group definitions to only the populations present in the PSD."""
    pop_set = set(str(pop_name) for pop_name in pop_names)
    pop_groups = POP_GROUPS if pop_groups is None else pop_groups
    resolved = {}
    for group_name, group_pops in dict(pop_groups).items():
        members = [str(pop_name) for pop_name in group_pops if str(pop_name) in pop_set]
        if members:
            resolved[str(group_name)] = members
    return resolved


def _get_psd_param_tag(
        rate_dt,
        win_len,
        win_overlap,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        t_limits_s=None,
        plot_f_limits=None,
        ) -> str:
    """Build the parameter tag shared by result paths."""
    parts = [
        f't_{_format_interval_tag(t_limits_s)}',
        f'dt_{_format_tag_value(rate_dt)}',
        f'win_{_format_tag_value(win_len)}',
        f'over_{_format_tag_value(win_overlap)}',
        f'avg_{str(average)}',
        f'norm_{_bool_tag(normalize)}',
        f'smooth_{int(smooth_freq_bins)}',
    ]
    if normalize:
        parts.append(
            f'nband_{_format_tag_value(normalize_f_band[0])}_{_format_tag_value(normalize_f_band[1])}'
        )
    if plot_f_limits is not None:
        parts.append(
            f'plotf_{_format_tag_value(plot_f_limits[0])}_{_format_tag_value(plot_f_limits[1])}'
        )
    return '__'.join(parts)


def _get_psd_cache_tag(
        rate_dt,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        t_limits_s=None,
        plot_f_limits=None,
        ) -> str:
    """Build the full cache tag including Welch frequency limits."""
    return (
        f'{_get_psd_param_tag(rate_dt, win_len, win_overlap, average, normalize, normalize_f_band, smooth_freq_bins, t_limits_s=t_limits_s, plot_f_limits=plot_f_limits)}'
        f'__f_{_format_tag_value(fmin)}_{_format_tag_value(fmax)}'
    )


def _get_results_dirname(
        rate_dt,
        win_len,
        win_overlap,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        logx,
        logy,
        t_limits_s=None,
        plot_f_limits=None,
        ) -> str:
    """Build the output folder name for one rate-PSD plotting configuration."""
    return (
        f'{_get_psd_param_tag(rate_dt, win_len, win_overlap, average, normalize, normalize_f_band, smooth_freq_bins, t_limits_s=t_limits_s, plot_f_limits=plot_f_limits)}'
        f'__logx_{_bool_tag(logx)}'
        f'__logy_{_bool_tag(logy)}'
    )


def _get_results_dir(
        results_root: Path,
        exp_label: str,
        rate_dt,
        win_len,
        win_overlap,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        logx,
        logy,
        t_limits_s=None,
        plot_f_limits=None,
        ) -> Path:
    """Return the final results directory for one rate PSD configuration."""
    return (
        Path(results_root)
        / exp_label
        / RESULT_GROUP
        / _get_results_dirname(
            rate_dt,
            win_len,
            win_overlap,
            average,
            normalize,
            normalize_f_band,
            smooth_freq_bins,
            logx,
            logy,
            t_limits_s=t_limits_s,
            plot_f_limits=plot_f_limits,
        )
    )


def _get_psd_cache_path(
        dirpath_proc: Path,
        rate_dt,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        t_limits_s=None,
        plot_f_limits=None,
        ) -> Path:
    """Return the cached Welch PSD path for one parameter set."""
    return (
        Path(dirpath_proc)
        / PSD_CACHE_GROUP
        / f'{_get_psd_cache_tag(rate_dt, win_len, win_overlap, fmin, fmax, average, normalize, normalize_f_band, smooth_freq_bins, t_limits_s=t_limits_s, plot_f_limits=plot_f_limits)}.nc'
    )


def _get_rates_for_psd(fpath_sim_result, dirpath_proc_root, rate_dt: float, t_limits_s=None):
    """Load cached population rates and apply optional time limits."""
    fpath_rates_cache = get_rates_cache_path(fpath_sim_result, dirpath_proc_root, rate_dt)
    sim_result = None
    if not fpath_rates_cache.exists():
        sim_result = load_sim_result(fpath_sim_result)
    rates = load_or_extract_rates(sim_result, fpath_sim_result, dirpath_proc_root, rate_dt=rate_dt)
    if t_limits_s is not None:
        rates = rates.sel(time=slice(*t_limits_s))
    return rates, fpath_rates_cache


def _postprocess_rate_psd(psd, normalize: bool, normalize_f_band, smooth_freq_bins: int, plot_f_limits=None):
    """Apply notebook-style normalization, smoothing, and plot-frequency selection."""
    psd_out = psd.copy(deep=True)
    if normalize:
        baseline = psd_out.sel(freq=slice(*normalize_f_band)).mean(dim='freq')
        psd_out = psd_out / baseline
    if int(smooth_freq_bins) > 1:
        psd_out = psd_out.rolling(freq=int(smooth_freq_bins), center=True, min_periods=1).mean()
    if plot_f_limits is not None:
        psd_out = psd_out.sel(freq=slice(*plot_f_limits))
    return psd_out


def _load_or_compute_psd_cache(
        rates,
        cache_path: Path,
        *,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        normalize,
        normalize_f_band,
        smooth_freq_bins,
        plot_f_limits=None,
        ) -> tuple[object, bool]:
    """Load cached rate PSD or compute, postprocess, and save it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_xr(cache_path, load=True), True

    psd = calc_xr_welch(
        rates,
        win_len=win_len,
        win_overlap=win_overlap,
        fmin=fmin,
        fmax=fmax,
        average=average,
        compute=True,
        store_proc_info=True,
    )
    psd = _postprocess_rate_psd(
        psd,
        normalize=normalize,
        normalize_f_band=normalize_f_band,
        smooth_freq_bins=smooth_freq_bins,
        plot_f_limits=plot_f_limits,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_xr(_make_netcdf_safe_dataarray(psd), cache_path)
    return psd, False


def _plot_pop_psd(psd, pop_name: str, fpath_out: Path, logx: bool, logy: bool) -> None:
    """Plot one population PSD line and save it as PNG."""
    row = psd.sel(pop=pop_name)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        np.asarray(row.coords['freq'].values, dtype=float),
        np.asarray(row.values, dtype=float),
        color='k',
        linewidth=1.5,
    )
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'Rate PSD: {pop_name}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _plot_group_psd(psd, group_name: str, pop_names, fpath_out: Path, logx: bool, logy: bool) -> None:
    """Plot all populations from one hardcoded group into one figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for pop_name in pop_names:
        row = psd.sel(pop=pop_name)
        ax.plot(
            np.asarray(row.coords['freq'].values, dtype=float),
            np.asarray(row.values, dtype=float),
            linewidth=1.5,
            label=str(pop_name),
        )
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(str(group_name))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run cached per-population firing-rate PSD analysis."""
    exp_label = get_exp_label(FPATH_SIM_RESULT)
    dirpath_proc = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    dirpath_out = _get_results_dir(
        DIRPATH_RESULTS_ROOT,
        exp_label,
        RATE_DT,
        PSD_WIN_LEN,
        PSD_WIN_OVERLAP,
        PSD_AVERAGE,
        NORMALIZE,
        NORMALIZE_F_BAND,
        SMOOTH_FREQ_BINS,
        LOGX,
        LOGY,
        t_limits_s=T_LIMITS_S,
        plot_f_limits=PLOT_F_LIMITS,
    )
    cache_path = _get_psd_cache_path(
        dirpath_proc,
        RATE_DT,
        PSD_WIN_LEN,
        PSD_WIN_OVERLAP,
        PSD_FMIN,
        PSD_FMAX,
        PSD_AVERAGE,
        NORMALIZE,
        NORMALIZE_F_BAND,
        SMOOTH_FREQ_BINS,
        t_limits_s=T_LIMITS_S,
        plot_f_limits=PLOT_F_LIMITS,
    )

    rates, fpath_rates_cache = _get_rates_for_psd(
        FPATH_SIM_RESULT,
        DIRPATH_PROC_ROOT,
        RATE_DT,
        t_limits_s=T_LIMITS_S,
    )
    pop_names = _resolve_pop_names(rates.coords['pop'].values.tolist(), requested_pop_names=POP_NAMES)
    rates = rates.sel(pop=pop_names)
    psd, cache_hit = _load_or_compute_psd_cache(
        rates,
        cache_path,
        win_len=PSD_WIN_LEN,
        win_overlap=PSD_WIN_OVERLAP,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        average=PSD_AVERAGE,
        normalize=NORMALIZE,
        normalize_f_band=NORMALIZE_F_BAND,
        smooth_freq_bins=SMOOTH_FREQ_BINS,
        plot_f_limits=PLOT_F_LIMITS,
    )

    dirpath_out.mkdir(parents=True, exist_ok=True)
    if GROUP_PLOTS:
        for group_name, group_pops in _resolve_pop_groups(pop_names).items():
            _plot_group_psd(
                psd,
                group_name,
                group_pops,
                dirpath_out / f'psd_group_{group_name}.png',
                logx=LOGX,
                logy=LOGY,
            )
    else:
        for pop_name in pop_names:
            _plot_pop_psd(
                psd,
                str(pop_name),
                dirpath_out / f'psd_{pop_name}.png',
                logx=LOGX,
                logy=LOGY,
            )

    print(f'Output dir: {dirpath_out}')
    print(f'Rates cache: {fpath_rates_cache}')
    print(f'PSD cache: {"hit" if cache_hit else "miss"} at {cache_path}')


if __name__ == '__main__':
    main()
