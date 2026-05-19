"""Per-channel LFP PSD analysis with cached Welch spectra."""

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
    get_lfp_cache_path,
    get_proc_dir,
    load_or_extract_lfp,
    load_sim_result,
)
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_signal import interp_time_outliers
from sim_data_analyzer.xr_spect import calc_xr_welch


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'

RESULT_GROUP = 'lfp_psd'
PSD_CACHE_GROUP = 'psd_cache'

SIGNAL_TYPE = 'lfp'
T_LIMITS_S = None
PSD_WIN_LEN = 2.0
PSD_WIN_OVERLAP = 0.5
PSD_FMIN = 2.0
PSD_FMAX = 100.0
PSD_AVERAGE = 'mean'

LOGX = 0
LOGY = 1


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


def _normalize_signal_type(signal_type: str) -> str:
    """Validate the configured signal type."""
    signal_type = str(signal_type).strip().lower()
    if signal_type not in {'lfp', 'csd'}:
        raise ValueError("SIGNAL_TYPE should be either 'lfp' or 'csd'")
    return signal_type


def _get_psd_param_tag(
        win_len,
        win_overlap,
        average,
        t_limits_s=None,
        ) -> str:
    """Build the parameter tag shared by result paths."""
    return (
        f't_{_format_interval_tag(t_limits_s)}'
        f'__win_{_format_tag_value(win_len)}'
        f'__over_{_format_tag_value(win_overlap)}'
        f'__avg_{str(average)}'
    )


def _get_psd_cache_tag(
        signal_type,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        t_limits_s=None,
        ) -> str:
    """Build the full cache tag including signal type and frequency range."""
    return (
        f'{_normalize_signal_type(signal_type)}'
        f'__{_get_psd_param_tag(win_len, win_overlap, average, t_limits_s=t_limits_s)}'
        f'__f_{_format_tag_value(fmin)}_{_format_tag_value(fmax)}'
    )


def _get_results_dirname(
        signal_type,
        win_len,
        win_overlap,
        average,
        logx,
        logy,
        t_limits_s=None,
        ) -> str:
    """Build the output folder name for one PSD plotting configuration."""
    return (
        f'{_normalize_signal_type(signal_type)}'
        f'__{_get_psd_param_tag(win_len, win_overlap, average, t_limits_s=t_limits_s)}'
        f'__logx_{_bool_tag(logx)}'
        f'__logy_{_bool_tag(logy)}'
    )


def _get_results_dir(
        results_root: Path,
        exp_label: str,
        signal_type,
        win_len,
        win_overlap,
        average,
        logx,
        logy,
        t_limits_s=None,
        ) -> Path:
    """Return the final results directory for one PSD configuration."""
    return (
        Path(results_root)
        / exp_label
        / RESULT_GROUP
        / _get_results_dirname(
            signal_type,
            win_len,
            win_overlap,
            average,
            logx,
            logy,
            t_limits_s=t_limits_s,
        )
    )


def _get_psd_cache_path(
        dirpath_proc: Path,
        signal_type,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        t_limits_s=None,
        ) -> Path:
    """Return the cached Welch PSD path for one parameter set."""
    return (
        Path(dirpath_proc)
        / PSD_CACHE_GROUP
        / f'{_get_psd_cache_tag(signal_type, win_len, win_overlap, fmin, fmax, average, t_limits_s=t_limits_s)}.nc'
    )


def _get_signal_for_psd(fpath_sim_result, dirpath_proc_root, signal_type: str, t_limits_s=None):
    """Load cached LFP, optionally derive CSD, and apply optional time limits."""
    signal_type = _normalize_signal_type(signal_type)
    fpath_lfp_cache = get_lfp_cache_path(fpath_sim_result, dirpath_proc_root)
    sim_result = None
    if not fpath_lfp_cache.exists():
        sim_result = load_sim_result(fpath_sim_result)
    lfp = load_or_extract_lfp(sim_result, fpath_sim_result, dirpath_proc_root)
    lfp = interp_time_outliers(lfp)
    if t_limits_s is not None:
        lfp = lfp.sel(time=slice(*t_limits_s))
    signal = lfp if signal_type == 'lfp' else calc_xr_csd(lfp)
    signal.name = signal_type
    return signal, fpath_lfp_cache


def _load_or_compute_psd_cache(
        signal,
        cache_path: Path,
        *,
        win_len,
        win_overlap,
        fmin,
        fmax,
        average,
        ) -> tuple[object, bool]:
    """Load cached Welch PSD or compute and save it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_xr(cache_path, load=True), True

    psd = calc_xr_welch(
        signal,
        win_len=win_len,
        win_overlap=win_overlap,
        fmin=fmin,
        fmax=fmax,
        average=average,
        compute=True,
        store_proc_info=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_xr(_make_netcdf_safe_dataarray(psd), cache_path)
    return psd, False


def _plot_channel_psd(psd, resolved_y: float, fpath_out: Path, signal_type: str, logx: bool, logy: bool) -> None:
    """Plot one channel PSD line and save it as PNG."""
    signal_type = _normalize_signal_type(signal_type)
    row = psd.sel(y=resolved_y)
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
    ax.set_title(f'{signal_type.upper()} PSD at y={resolved_y:g}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run cached per-channel LFP PSD analysis."""
    exp_label = get_exp_label(FPATH_SIM_RESULT)
    dirpath_proc = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    signal_type = _normalize_signal_type(SIGNAL_TYPE)
    dirpath_out = _get_results_dir(
        DIRPATH_RESULTS_ROOT,
        exp_label,
        signal_type,
        PSD_WIN_LEN,
        PSD_WIN_OVERLAP,
        PSD_AVERAGE,
        LOGX,
        LOGY,
        t_limits_s=T_LIMITS_S,
    )
    cache_path = _get_psd_cache_path(
        dirpath_proc,
        signal_type,
        PSD_WIN_LEN,
        PSD_WIN_OVERLAP,
        PSD_FMIN,
        PSD_FMAX,
        PSD_AVERAGE,
        t_limits_s=T_LIMITS_S,
    )

    signal, fpath_lfp_cache = _get_signal_for_psd(
        FPATH_SIM_RESULT,
        DIRPATH_PROC_ROOT,
        signal_type,
        t_limits_s=T_LIMITS_S,
    )
    psd, cache_hit = _load_or_compute_psd_cache(
        signal,
        cache_path,
        win_len=PSD_WIN_LEN,
        win_overlap=PSD_WIN_OVERLAP,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        average=PSD_AVERAGE,
    )

    dirpath_out.mkdir(parents=True, exist_ok=True)
    for resolved_y in np.asarray(psd.coords['y'].values, dtype=float).tolist():
        _plot_channel_psd(
            psd,
            float(resolved_y),
            dirpath_out / f'psd_y_{resolved_y:g}.png',
            signal_type=signal_type,
            logx=LOGX,
            logy=LOGY,
        )

    print(f'Output dir: {dirpath_out}')
    print(f'LFP cache: {fpath_lfp_cache}')
    print(f'PSD cache: {"hit" if cache_hit else "miss"} at {cache_path}')


if __name__ == '__main__':
    main()
