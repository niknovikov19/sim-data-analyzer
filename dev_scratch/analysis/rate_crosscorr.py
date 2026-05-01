"""Pairwise population rate cross-correlation analysis."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr

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
from sim_data_analyzer.xr_signal import calc_xr_crosscorr, filter_xr_signal
from sim_data_analyzer.xr_io import load_xr, save_xr


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
EXP_LABEL = get_exp_label(FPATH_SIM_RESULT)
DIRPATH_PROC = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
RESULT_GROUP = 'rate_xcorr'

ANALYSIS_LABEL = 'rate_xcorr_allpops'
POP_NAMES = None

#ANALYSIS_LABEL = 'rate_xcorr_L3_5A'
#POP_NAMES = ['IT3', 'PV3', 'SOM3', 'VIP3', 'NGF3',
#             'IT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A']

T_LIMITS = (10.0, 30.0)
RATE_DT = 1e-3

LAG_WINDOW = (-0.5, 0.5)

FILTER_FBAND = (8, 14)
FILTER_ORDER = 3

DO_PLOT = 0
PLOT_AMP_THRESHOLD = 0.08

CSV_ROUND_DIGITS = 3
DO_PLOT_MATRICES = 1
MATRIX_THRESHOLD = 0.5

CORRCACHE_VERSION = 'v1'


def _resolve_analysis_pop_names(all_pop_names, requested_pop_names=None):
    """Resolve the stable analysis population list after excluding frozen pops."""
    eligible_pop_names = [pop_name for pop_name in all_pop_names if 'frz' not in pop_name]
    if requested_pop_names is None:
        if not eligible_pop_names:
            raise ValueError('No non-frozen populations are available for analysis')
        return eligible_pop_names

    requested_pop_names = list(requested_pop_names)
    missing = [pop_name for pop_name in requested_pop_names if pop_name not in eligible_pop_names]
    if missing:
        raise ValueError(
            'Requested populations are unavailable after excluding frozen populations: '
            + ', '.join(missing)
        )
    if not requested_pop_names:
        raise ValueError('Population allowlist is empty after validation')
    return requested_pop_names


def _select_analysis_pops(rates, requested_pop_names=None):
    """Select the populations included in the pairwise analysis."""
    pop_names = _resolve_analysis_pop_names(
        rates.pop.values.tolist(),
        requested_pop_names=requested_pop_names,
    )
    return rates.sel(pop=pop_names)


def _iter_pop_pairs(pop_names):
    """Yield self-pairs and unordered population pairs in stable order."""
    for idx_i, pop_i in enumerate(pop_names):
        for pop_j in pop_names[idx_i:]:
            yield pop_i, pop_j


def _format_tag_value(value: float) -> str:
    """Format a numeric value into a compact filesystem-safe tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _get_filter_tag(filter_fband) -> str:
    """Build the filter-state tag used in the output folder name."""
    if filter_fband is None:
        return 'nofilt'

    filter_fband = np.asarray(filter_fband, dtype=float)
    if filter_fband.shape != (2,) or not np.all(np.isfinite(filter_fband)):
        raise ValueError('FILTER_FBAND should be a length-2 finite sequence')
    if filter_fband[0] >= filter_fband[1]:
        raise ValueError('FILTER_FBAND lower edge should be smaller than upper edge')
    return f'bp_{_format_tag_value(filter_fband[0])}_{_format_tag_value(filter_fband[1])}'


def _get_output_dir(results_root: Path, exp_label: str, analysis_label: str, filter_fband) -> Path:
    """Construct the final results directory for this analysis configuration."""
    return results_root / exp_label / RESULT_GROUP / f'{analysis_label}__{_get_filter_tag(filter_fband)}'


def _maybe_filter_rates(rates, filter_fband, filter_order: int):
    """Optionally bandpass-filter the population-rate traces before correlation."""
    if filter_fband is None:
        return rates
    return filter_xr_signal(
        rates,
        fband=tuple(float(x) for x in filter_fband),
        order=filter_order,
        btype='bandpass',
        compute=True,
        store_proc_info=True,
    )


def _extract_peak_metrics(corr):
    """Return peak amplitude and lag from the largest-|corr| sample."""
    if corr.sizes['lag'] == 0 or not np.isfinite(corr.values).any():
        return np.nan, np.nan

    peak_idx = int(np.nanargmax(np.abs(corr.values)))
    peak_lag = float(corr.coords['lag'].values[peak_idx])
    peak_val = float(corr.values[peak_idx])
    return peak_val, peak_lag


def _peak_summary(label: str, corr) -> str:
    """Return a short textual peak-lag summary for one correlation view."""
    peak_val, peak_lag = _extract_peak_metrics(corr)
    if not np.isfinite(peak_val):
        return f'{label}: no finite lag values in the requested window'
    return f'{label}: peak |corr| at lag={peak_lag:+.4f} s, value={peak_val:+.6g}'


def _init_metric_table(pop_names):
    """Create an empty square pop x pop table for one summary metric."""
    pop_names = list(pop_names)
    return np.full((len(pop_names), len(pop_names)), np.nan, dtype=float)


def _fill_symmetric_pair(table, pop_index, pop_i: str, pop_j: str, value: float) -> None:
    """Fill one unordered/self pair into both symmetric table locations."""
    idx_i = pop_index[pop_i]
    idx_j = pop_index[pop_j]
    table[idx_i, idx_j] = value
    table[idx_j, idx_i] = value


def _normalize_round_digits(round_digits):
    """Validate optional CSV rounding precision."""
    if round_digits is None:
        return None
    if isinstance(round_digits, bool) or not isinstance(round_digits, int):
        raise ValueError('CSV_ROUND_DIGITS should be an integer or None')
    if round_digits < 0:
        raise ValueError('CSV_ROUND_DIGITS should be non-negative')
    return round_digits


def _normalize_matrix_threshold(matrix_threshold):
    """Validate optional amplitude threshold for matrix visualization."""
    if matrix_threshold is None:
        return None
    matrix_threshold = float(matrix_threshold)
    if not np.isfinite(matrix_threshold):
        raise ValueError('MATRIX_THRESHOLD should be finite or None')
    if matrix_threshold < 0:
        raise ValueError('MATRIX_THRESHOLD should be non-negative')
    if matrix_threshold > 1:
        raise ValueError('MATRIX_THRESHOLD should not exceed 1 for normalized amplitudes')
    return matrix_threshold


def _normalize_plot_amp_threshold(plot_amp_threshold):
    """Validate optional amplitude threshold for pair-plot gating."""
    if plot_amp_threshold is None:
        return None
    plot_amp_threshold = float(plot_amp_threshold)
    if not np.isfinite(plot_amp_threshold):
        raise ValueError('PLOT_AMP_THRESHOLD should be finite or None')
    if plot_amp_threshold < 0:
        raise ValueError('PLOT_AMP_THRESHOLD should be non-negative')
    if plot_amp_threshold > 1:
        raise ValueError('PLOT_AMP_THRESHOLD should not exceed 1 for normalized amplitudes')
    return plot_amp_threshold


def _round_metric_value(value: float, round_digits):
    """Optionally round one metric value before CSV export."""
    if not np.isfinite(value):
        return value

    round_digits = _normalize_round_digits(round_digits)
    if round_digits is None:
        return value
    return float(np.round(value, round_digits))


def _csv_cell(value: float, round_digits=None) -> str:
    """Format one numeric CSV cell."""
    if not np.isfinite(value):
        return 'nan'
    round_digits = _normalize_round_digits(round_digits)
    if round_digits is not None:
        return f'{_round_metric_value(value, round_digits):.{round_digits}f}'
    return f'{value:.10g}'


def _write_metric_csv(fpath_csv: Path, pop_names, table, round_digits=None) -> None:
    """Write one square summary table to CSV with pop labels on rows/cols."""
    with fpath_csv.open('w', newline='', encoding='utf-8') as fobj:
        writer = csv.writer(fobj)
        writer.writerow(['pop', *pop_names])
        for pop_name, row in zip(pop_names, table):
            writer.writerow([pop_name, *[_csv_cell(value, round_digits=round_digits) for value in row]])


def _get_pair_png_dirname(plot_amp_threshold) -> str:
    """Construct the pair-PNG subfolder name."""
    plot_amp_threshold = _normalize_plot_amp_threshold(plot_amp_threshold)
    if plot_amp_threshold is None:
        return 'pair_pngs'
    return f'pair_pngs__thr_{_format_tag_value(plot_amp_threshold)}'


def _get_pair_png_dir(dirpath_out: Path, plot_amp_threshold) -> Path:
    """Construct the pair-PNG output subfolder."""
    return Path(dirpath_out) / _get_pair_png_dirname(plot_amp_threshold)


def _get_crosscorr_cache_dir(dirpath_proc: Path) -> Path:
    """Construct the grouped cross-correlation cache directory."""
    return Path(dirpath_proc) / 'crosscorr_cache'


def _format_window_tag(values) -> str:
    """Format a 2-point time window into a compact tag."""
    return f'{_format_tag_value(values[0])}_{_format_tag_value(values[1])}'


def _get_pop_selection_tag(pop_names) -> str:
    """Construct a stable tag for the selected population set."""
    pop_names = list(pop_names)
    if not pop_names:
        raise ValueError('At least one population is required for the cross-correlation cache')
    return '__'.join(pop_names)


def _get_crosscorr_cache_name(
        analysis_label: str,
        pop_names,
        rate_dt: float,
        t_limits,
        lag_window,
        filter_fband,
        ) -> str:
    """Construct the full cross-correlation cache filename."""
    return (
        f'{analysis_label}'
        f'__npops_{len(pop_names)}'
        f'__dt_{_format_tag_value(rate_dt)}'
        f'__t_{_format_window_tag(t_limits)}'
        f'__lag_{_format_window_tag(lag_window)}'
        f'__{_get_filter_tag(filter_fband)}'
        f'__{CORRCACHE_VERSION}.nc'
    )


def _get_crosscorr_cache_path(
        dirpath_proc: Path,
        analysis_label: str,
        pop_names,
        rate_dt: float,
        t_limits,
        lag_window,
        filter_fband,
        ) -> Path:
    """Construct the full cross-correlation cache file path."""
    return _get_crosscorr_cache_dir(dirpath_proc) / _get_crosscorr_cache_name(
        analysis_label,
        pop_names,
        rate_dt,
        t_limits,
        lag_window,
        filter_fband,
    )


def _make_pair_label(pop_i: str, pop_j: str) -> str:
    """Construct one stable pair label."""
    return f'{pop_i}__{pop_j}'


def _pair_passes_plot_threshold(peak_amp: float, plot_amp_threshold) -> bool:
    """Decide whether one pair should get a correlation PNG."""
    plot_amp_threshold = _normalize_plot_amp_threshold(plot_amp_threshold)
    if not np.isfinite(peak_amp):
        return False
    if plot_amp_threshold is None:
        return True
    return abs(float(peak_amp)) >= plot_amp_threshold


def _pair_is_self(pair_label: str, corr_ds: xr.Dataset) -> bool:
    """Check whether one cached pair corresponds to a self-correlation."""
    pair_corr = corr_ds['normalized_corr'].sel(pair=pair_label)
    pop_i = str(pair_corr.coords['pop_i'].item())
    pop_j = str(pair_corr.coords['pop_j'].item())
    return pop_i == pop_j


def _compute_crosscorr_cache_dataset(rates, pair_list):
    """Compute lag-resolved cross-correlation traces for all pairs."""
    pair_labels = []
    pop_i_vals = []
    pop_j_vals = []
    raw_vals = []
    demeaned_vals = []
    normalized_vals = []
    lag_vals = None

    # Compute and collect all three correlation views once per unordered pair.
    pair_count = len(pair_list)
    for pair_idx, (pop_i, pop_j) in enumerate(pair_list, start=1):
        _print_progress(pair_idx, pair_count, pop_i, pop_j)
        rate_i = rates.sel(pop=pop_i)
        rate_j = rates.sel(pop=pop_j)

        raw_corr = calc_xr_crosscorr(
            rate_i,
            rate_j,
            lag_window=LAG_WINDOW,
            subtract_mean=False,
            normalize=False,
            compute=True,
            store_proc_info=True,
        )
        demeaned_corr = calc_xr_crosscorr(
            rate_i,
            rate_j,
            lag_window=LAG_WINDOW,
            subtract_mean=True,
            normalize=False,
            compute=True,
            store_proc_info=True,
        )
        normalized_corr = calc_xr_crosscorr(
            rate_i,
            rate_j,
            lag_window=LAG_WINDOW,
            subtract_mean=True,
            normalize=True,
            compute=True,
            store_proc_info=True,
        )

        print(f'  {_peak_summary("Raw / N", raw_corr)}')
        print(f'  {_peak_summary("Demeaned / N", demeaned_corr)}')
        print(f'  {_peak_summary("Demeaned + normalized", normalized_corr)}')

        if lag_vals is None:
            lag_vals = raw_corr.lag.values

        pair_labels.append(_make_pair_label(pop_i, pop_j))
        pop_i_vals.append(pop_i)
        pop_j_vals.append(pop_j)
        raw_vals.append(np.asarray(raw_corr.values, dtype=float))
        demeaned_vals.append(np.asarray(demeaned_corr.values, dtype=float))
        normalized_vals.append(np.asarray(normalized_corr.values, dtype=float))

    # Persist the full lag-resolved outputs in one dataset for cheap reuse.
    return xr.Dataset(
        data_vars={
            'raw_corr': (['pair', 'lag'], np.asarray(raw_vals, dtype=float)),
            'demeaned_corr': (['pair', 'lag'], np.asarray(demeaned_vals, dtype=float)),
            'normalized_corr': (['pair', 'lag'], np.asarray(normalized_vals, dtype=float)),
        },
        coords={
            'pair': pair_labels,
            'lag': lag_vals,
            'pop_i': ('pair', pop_i_vals),
            'pop_j': ('pair', pop_j_vals),
        },
        attrs={
            'cache_version': CORRCACHE_VERSION,
            'analysis_label': ANALYSIS_LABEL,
            'rate_dt': RATE_DT,
            't_limits': list(T_LIMITS),
            'lag_window': list(LAG_WINDOW),
            'filter_fband': None if FILTER_FBAND is None else list(map(float, FILTER_FBAND)),
            'filter_order': FILTER_ORDER,
        },
    )


def _load_or_compute_crosscorr_cache(rates, pair_list, fpath_cache: Path):
    """Load an existing correlogram cache or compute and save it."""
    if fpath_cache.exists():
        print(f'Loading cached cross-correlograms: {fpath_cache}')
        return load_xr(fpath_cache, data_type='dataset', load=True)

    print(f'Computing cross-correlograms and caching to: {fpath_cache}')
    corr_ds = _compute_crosscorr_cache_dataset(rates, pair_list)
    save_xr(corr_ds, fpath_cache)
    return corr_ds


def _compute_peak_tables_from_cache(corr_ds, pop_names):
    """Derive peak amplitude and lag tables from the cached normalized traces."""
    pop_index = {pop_name: idx for idx, pop_name in enumerate(pop_names)}
    amp_table = _init_metric_table(pop_names)
    lag_table = _init_metric_table(pop_names)
    peak_amp_by_pair = {}

    # Collapse each cached normalized trace to its peak amplitude and lag.
    for pair_label in corr_ds.pair.values.tolist():
        normalized_corr = corr_ds['normalized_corr'].sel(pair=pair_label)
        peak_amp, peak_lag = _extract_peak_metrics(normalized_corr)
        pop_i = str(normalized_corr.coords['pop_i'].item())
        pop_j = str(normalized_corr.coords['pop_j'].item())
        _fill_symmetric_pair(amp_table, pop_index, pop_i, pop_j, peak_amp)
        _fill_symmetric_pair(lag_table, pop_index, pop_i, pop_j, peak_lag)
        peak_amp_by_pair[pair_label] = peak_amp

    return amp_table, lag_table, peak_amp_by_pair


def _get_matrix_png_name(analysis_label: str, matrix_threshold=None, masked: bool = False) -> str:
    """Construct one matrix-summary PNG filename."""
    matrix_threshold = _normalize_matrix_threshold(matrix_threshold)
    if masked and (matrix_threshold is None):
        raise ValueError('Masked matrix PNG naming requires a non-None MATRIX_THRESHOLD')
    if not masked:
        return 'matrices.png'
    return f'matrices__thr_{_format_tag_value(matrix_threshold)}.png'


def _get_matrix_png_names(analysis_label: str, matrix_threshold):
    """Construct the plain and optional masked matrix-summary PNG filenames."""
    matrix_threshold = _normalize_matrix_threshold(matrix_threshold)
    names = [ _get_matrix_png_name(analysis_label, masked=False) ]
    if matrix_threshold is not None:
        names.append(_get_matrix_png_name(analysis_label, matrix_threshold, masked=True))
    return names


def _prepare_matrix_tables_for_plot(amp_table, lag_table, matrix_threshold):
    """Prepare plotted tables and a weak-cell mask for visualization."""
    matrix_threshold = _normalize_matrix_threshold(matrix_threshold)
    amp_plot = np.array(amp_table, dtype=float, copy=True)
    lag_plot = np.array(lag_table, dtype=float, copy=True)
    diag_mask = np.eye(amp_plot.shape[0], dtype=bool)
    if matrix_threshold is None:
        return amp_plot, lag_plot, None, diag_mask

    weak_mask = np.isfinite(amp_plot) & (np.abs(amp_plot) < matrix_threshold)
    return amp_plot, lag_plot, weak_mask, diag_mask


def _get_symmetric_plot_limit(values, fallback: float) -> float:
    """Auto-scale to the largest finite absolute value while keeping symmetry."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback)

    vmax = float(np.max(np.abs(finite)))
    if vmax <= 0:
        return float(fallback)
    return vmax


def _get_lag_plot_limit(lag_plot, plot_mask, fallback: float) -> float:
    """Get the symmetric lag limit from the lag cells visible in the current view."""
    visible_lags = np.array(lag_plot, dtype=float, copy=True)
    if plot_mask is not None:
        visible_lags[np.asarray(plot_mask, dtype=bool)] = np.nan
    return _get_symmetric_plot_limit(visible_lags, fallback=fallback)


def _get_amp_plot_limit(amp_plot, diag_mask, fallback: float) -> float:
    """Get the symmetric amplitude limit from visible off-diagonal values."""
    visible_amp = np.array(amp_plot, dtype=float, copy=True)
    visible_amp[np.asarray(diag_mask, dtype=bool)] = np.nan
    return _get_symmetric_plot_limit(visible_amp, fallback=fallback)


def _overlay_amp_mask(ax, weak_mask, edgecolor=(0.45, 0.45, 0.45, 0.95), hatch='///') -> None:
    """Overlay weak amplitude cells with a transparent hatch while preserving base colors."""
    if weak_mask is None:
        return

    weak_mask = np.asarray(weak_mask, dtype=bool)
    for row_idx, col_idx in zip(*np.where(weak_mask)):
        ax.add_patch(
            Rectangle(
                (col_idx - 0.5, row_idx - 0.5),
                1.0,
                1.0,
                facecolor=(1.0, 1.0, 1.0, 0.0),
                edgecolor=edgecolor,
                linewidth=0.0,
                hatch=hatch,
                fill=True,
            )
        )


def _make_matrix_plot(
        fpath_out: Path,
        pop_names,
        amp_table,
        lag_table,
        analysis_label: str,
        filter_fband,
        matrix_threshold,
        use_mask: bool,
        ) -> None:
    """Render amplitude and lag summary matrices into one PNG."""
    # Build the matrix view first so scaling and masking use the same arrays.
    amp_plot, lag_plot, weak_mask, diag_mask = _prepare_matrix_tables_for_plot(
        amp_table, lag_table, matrix_threshold
    )
    amp_abs = _get_amp_plot_limit(amp_plot, diag_mask, fallback=1.0)
    lag_mask = diag_mask if weak_mask is None else (diag_mask | weak_mask)
    lag_abs = _get_lag_plot_limit(
        lag_plot,
        lag_mask if use_mask else diag_mask,
        fallback=max(abs(float(LAG_WINDOW[0])), abs(float(LAG_WINDOW[1])), 1e-12),
    )

    # The masked variant keeps amplitude colors but hides weak lag cells.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if use_mask and (weak_mask is not None):
        lag_cmap = plt.get_cmap('bwr').copy()
        lag_cmap.set_bad(color='white')
        lag_display = np.ma.masked_where(lag_mask, lag_plot)
        amp_display = np.ma.masked_where(diag_mask, amp_plot)
    else:
        lag_cmap = plt.get_cmap('bwr').copy()
        lag_cmap.set_bad(color='white')
        lag_display = np.ma.masked_where(diag_mask, lag_plot)
        amp_display = np.ma.masked_where(diag_mask, amp_plot)

    plot_specs = [
        ('Peak amplitude', amp_display, {'cmap': 'bwr', 'vmin': -amp_abs, 'vmax': amp_abs}),
        ('Peak lag (s)', lag_display, {'cmap': lag_cmap, 'vmin': -lag_abs, 'vmax': lag_abs}),
    ]

    for ax_idx, (ax, (title, table, image_kwargs)) in enumerate(zip(axes, plot_specs)):
        image = ax.imshow(table, aspect='auto', **image_kwargs)
        if use_mask and (ax_idx == 0):
            _overlay_amp_mask(ax, weak_mask)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(pop_names)))
        ax.set_yticks(np.arange(len(pop_names)))
        ax.set_xticklabels(pop_names, rotation=45, ha='right')
        ax.set_yticklabels(pop_names)
        fig.colorbar(image, ax=ax, shrink=0.9)

    fig.suptitle(
        f'{analysis_label}: normalized mean-subtracted cross-correlation peaks '
        f'({_get_filter_tag(filter_fband)})'
    )
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _print_progress(pair_idx: int, pair_count: int, pop_i: str, pop_j: str) -> None:
    """Print one compact progress line for the current pair."""
    print(f'[{pair_idx:>3d}/{pair_count}] {pop_i} vs {pop_j}')


def _print_plot_progress(plot_idx: int, plot_count: int, pop_i: str, pop_j: str) -> None:
    """Print one compact progress line for pair-PNG generation."""
    print(f'[plot {plot_idx:>3d}/{plot_count}] {pop_i} vs {pop_j}')


def _make_plot(fpath_out: Path, raw_corr, demeaned_corr, normalized_corr, pop_i: str, pop_j: str) -> None:
    """Render the three standard cross-correlation views for one pair."""
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
        f'Population rate cross-correlation: {pop_i} vs {pop_j}, '
        f'window=[{LAG_WINDOW[0]:g}, {LAG_WINDOW[1]:g}] s'
    )
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _write_metadata(
        fpath_md: Path,
        dirpath_out: Path,
        pop_names,
        pair_count: int,
        rate_cache_path: Path,
        crosscorr_cache_dir: Path,
        crosscorr_cache_path: Path,
        amp_csv_name: str,
        lag_csv_name: str,
        matrix_png_names,
        pair_png_dir_name: str | None,
        plotted_pair_count: int,
        do_plot: bool,
        do_plot_matrices: bool,
        filter_fband,
        filter_order: int,
        requested_pop_names,
        csv_round_digits,
        matrix_threshold,
        plot_amp_threshold,
        ) -> None:
    """Write one Markdown metadata file for the analysis output folder."""
    # Capture the current run settings in a machine-readable block first.
    params = {
        'ANALYSIS_LABEL': ANALYSIS_LABEL,
        'T_LIMITS': list(T_LIMITS),
        'RATE_DT': RATE_DT,
        'LAG_WINDOW': list(LAG_WINDOW),
        'POP_NAMES': None if requested_pop_names is None else list(requested_pop_names),
        'DO_PLOT': bool(do_plot),
        'DO_PLOT_MATRICES': bool(do_plot_matrices),
        'FILTER_FBAND': None if filter_fband is None else list(map(float, filter_fband)),
        'FILTER_ORDER': int(filter_order),
        'CSV_ROUND_DIGITS': _normalize_round_digits(csv_round_digits),
        'MATRIX_THRESHOLD': _normalize_matrix_threshold(matrix_threshold),
        'PLOT_AMP_THRESHOLD': _normalize_plot_amp_threshold(plot_amp_threshold),
        'pair_enumeration': 'self pairs plus unordered cross-pop pairs in filtered pop order',
        'correlation_views': [
            'raw_over_N',
            'demeaned_over_N',
            'demeaned_normalized',
        ],
        'summary_metric': (
            'largest-absolute normalized mean-subtracted cross-correlation peak '
            '(amplitude and lag)'
        ),
    }

    # Then summarize the concrete artifacts and thresholds for quick inspection.
    lines = [
        '# Rate Cross-Correlation Analysis',
        '',
        'Pairwise population-rate cross-correlations for populations whose names do not contain `frz`,',
        'optionally restricted by `POP_NAMES` and optionally bandpass-filtered before correlation.',
        '',
        '## Paths',
        '',
        f'- Script: `{Path(__file__).resolve()}`',
        f'- Raw source: `{FPATH_SIM_RESULT.resolve()}`',
        f'- Rate cache used: `{rate_cache_path.resolve()}`',
        f'- Intermediate/cache root: `{DIRPATH_PROC.resolve()}`',
        f'- Cross-correlation cache dir: `{crosscorr_cache_dir.resolve()}`',
        f'- Cross-correlation cache file: `{crosscorr_cache_path.resolve()}`',
        f'- Results folder: `{dirpath_out.resolve()}`',
        '',
        '## Parameters',
        '',
        '```json',
        json.dumps(params, indent=2),
        '```',
        '',
        '## Output Naming',
        '',
        '- PNG naming convention: `<pop_i>__<pop_j>.png`',
        f'- Peak-amplitude CSV: `{amp_csv_name}`',
        f'- Peak-lag CSV: `{lag_csv_name}`',
        (
            f'- Pair-PNG subfolder: `{pair_png_dir_name}`'
            if pair_png_dir_name is not None else '- Pair-PNG subfolder: not generated'
        ),
        (
            '- Matrix-summary PNGs: '
            + ', '.join(f'`{name}`' for name in matrix_png_names)
            if matrix_png_names else '- Matrix-summary PNGs: not generated'
        ),
        '- CSV metrics come from the normalized, mean-subtracted cross-correlation peak',
        '',
        '## Populations',
        '',
        f'- Included populations: {", ".join(pop_names)}',
        f'- Number of analyzed pairs: {pair_count}',
        f'- Plotting enabled: {bool(do_plot)}',
        f'- Number of pair PNGs written: {plotted_pair_count}',
        f'- Matrix plotting enabled: {bool(do_plot_matrices)}',
        (
            f'- Matrix threshold: {_normalize_matrix_threshold(matrix_threshold)} '
            '(masked view uses amplitude hatching and white lag cells for smaller |amplitude| values)'
        ),
        (
            f'- Pair plot threshold: {_normalize_plot_amp_threshold(plot_amp_threshold)} '
            '(pair PNGs use normalized, mean-subtracted peak amplitude gating)'
        ),
    ]
    fpath_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    """Run the pairwise population rate cross-correlation workflow."""
    dirpath_out = _get_output_dir(DIRPATH_RESULTS_ROOT, EXP_LABEL, ANALYSIS_LABEL, FILTER_FBAND)
    amp_csv_name = f'{ANALYSIS_LABEL}__amp.csv'
    lag_csv_name = f'{ANALYSIS_LABEL}__lag.csv'
    matrix_png_names = _get_matrix_png_names(ANALYSIS_LABEL, MATRIX_THRESHOLD)

    sim_result = None
    rate_cache = get_rates_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    if not rate_cache.exists():
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = load_sim_result(FPATH_SIM_RESULT)

    rates = load_or_extract_rates(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    rates = rates.sel(time=slice(*T_LIMITS)).load()
    rates = _select_analysis_pops(rates, requested_pop_names=POP_NAMES)
    rates = _maybe_filter_rates(rates, FILTER_FBAND, FILTER_ORDER)

    # Resolve the cache and results layout from the finalized analysis config.
    pop_names = rates.pop.values.tolist()
    pair_list = list(_iter_pop_pairs(pop_names))
    pair_count = len(pair_list)
    crosscorr_cache_dir = _get_crosscorr_cache_dir(DIRPATH_PROC)
    crosscorr_cache_path = _get_crosscorr_cache_path(
        DIRPATH_PROC,
        ANALYSIS_LABEL,
        pop_names,
        RATE_DT,
        T_LIMITS,
        LAG_WINDOW,
        FILTER_FBAND,
    )

    dirpath_out.mkdir(parents=True, exist_ok=True)
    print(f'Running {ANALYSIS_LABEL} for {pair_count} pairs in {dirpath_out}')
    # Build or reload the expensive lag-resolved correlograms before plotting.
    corr_ds = _load_or_compute_crosscorr_cache(rates, pair_list, crosscorr_cache_path)
    amp_table, lag_table, peak_amp_by_pair = _compute_peak_tables_from_cache(corr_ds, pop_names)
    plotted_pair_count = 0
    pair_png_dir_name = None

    if DO_PLOT:
        # Plot only the non-self pairs that pass the normalized peak threshold.
        pair_png_dir = _get_pair_png_dir(dirpath_out, PLOT_AMP_THRESHOLD)
        pair_png_dir_name = pair_png_dir.name
        pair_png_dir.mkdir(parents=True, exist_ok=True)
        pair_labels_to_plot = [
            pair_label for pair_label in corr_ds.pair.values.tolist()
            if (not _pair_is_self(pair_label, corr_ds))
            and _pair_passes_plot_threshold(peak_amp_by_pair[pair_label], PLOT_AMP_THRESHOLD)
        ]
        print(f'Writing {len(pair_labels_to_plot)} pair PNGs to {pair_png_dir}')
        for plot_idx, pair_label in enumerate(pair_labels_to_plot, start=1):
            peak_amp = peak_amp_by_pair[pair_label]
            raw_corr = corr_ds['raw_corr'].sel(pair=pair_label)
            demeaned_corr = corr_ds['demeaned_corr'].sel(pair=pair_label)
            normalized_corr = corr_ds['normalized_corr'].sel(pair=pair_label)
            pop_i = str(raw_corr.coords['pop_i'].item())
            pop_j = str(raw_corr.coords['pop_j'].item())
            _print_plot_progress(plot_idx, len(pair_labels_to_plot), pop_i, pop_j)
            fpath_png = pair_png_dir / f'{pop_i}__{pop_j}.png'
            _make_plot(fpath_png, raw_corr, demeaned_corr, normalized_corr, pop_i, pop_j)
            plotted_pair_count += 1

    # Write the tabular summaries before the optional matrix visualizations.
    _write_metric_csv(
        dirpath_out / amp_csv_name,
        pop_names,
        amp_table,
        round_digits=CSV_ROUND_DIGITS,
    )
    _write_metric_csv(
        dirpath_out / lag_csv_name,
        pop_names,
        lag_table,
        round_digits=CSV_ROUND_DIGITS,
    )
    if DO_PLOT_MATRICES:
        # Export the plain matrix view and, when requested, the thresholded one.
        _make_matrix_plot(
            dirpath_out / matrix_png_names[0],
            pop_names,
            amp_table,
            lag_table,
            ANALYSIS_LABEL,
            FILTER_FBAND,
            MATRIX_THRESHOLD,
            use_mask=False,
        )
        if MATRIX_THRESHOLD is not None:
            _make_matrix_plot(
                dirpath_out / matrix_png_names[1],
                pop_names,
                amp_table,
                lag_table,
                ANALYSIS_LABEL,
                FILTER_FBAND,
                MATRIX_THRESHOLD,
                use_mask=True,
            )
    # Finish with a README that points at the exact cache and result artifacts.
    _write_metadata(
        dirpath_out / 'README.md',
        dirpath_out,
        pop_names,
        pair_count,
        rate_cache,
        crosscorr_cache_dir,
        crosscorr_cache_path,
        amp_csv_name,
        lag_csv_name,
        matrix_png_names if DO_PLOT_MATRICES else [],
        pair_png_dir_name,
        plotted_pair_count,
        do_plot=DO_PLOT,
        do_plot_matrices=DO_PLOT_MATRICES,
        filter_fband=FILTER_FBAND,
        filter_order=FILTER_ORDER,
        requested_pop_names=POP_NAMES,
        csv_round_digits=CSV_ROUND_DIGITS,
        matrix_threshold=MATRIX_THRESHOLD,
        plot_amp_threshold=PLOT_AMP_THRESHOLD,
    )
    print(f'Saved results: {dirpath_out}')


if __name__ == '__main__':
    main()
