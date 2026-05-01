"""Pairwise population rate coherence analysis."""

from __future__ import annotations

import csv
import hashlib
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
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_spect import calc_xr_cpsd


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
EXP_LABEL = get_exp_label(FPATH_SIM_RESULT)
DIRPATH_PROC = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
RESULT_GROUP = 'rate_coher'

ANALYSIS_LABEL = 'rate_coher_allpops'
POP_NAMES = None

T_LIMITS = (10.0, 30.0)
RATE_DT = 1e-3

WIN_LEN = 2
WIN_OVERLAP = 0.5
FMAX = 100
FBAND = (8, 14)

DO_PLOT_MATRICES = 1
COHERENCE_THRESHOLD = 0.5
CSV_ROUND_DIGITS = 3

COHERCACHE_VERSION = 'v1'


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


def _format_window_tag(values) -> str:
    """Format a 2-point numeric interval into a compact tag."""
    return f'{_format_tag_value(values[0])}_{_format_tag_value(values[1])}'


def _get_fband_tag(fband) -> str:
    """Build the coherence-band tag used in result filenames."""
    fband = np.asarray(fband, dtype=float)
    if fband.shape != (2,) or not np.all(np.isfinite(fband)):
        raise ValueError('FBAND should be a length-2 finite sequence')
    if fband[0] >= fband[1]:
        raise ValueError('FBAND lower edge should be smaller than upper edge')
    return f'fband_{_format_tag_value(fband[0])}_{_format_tag_value(fband[1])}'


def _get_output_dirname(analysis_label: str, win_len: float, win_overlap: float, fband) -> str:
    """Construct the concise analysis-specific results folder name."""
    return (
        f'{analysis_label}'
        f'__win_{_format_tag_value(win_len)}'
        f'_over_{_format_tag_value(win_overlap)}'
        f'_{_get_fband_tag(fband)}'
    )


def _get_output_dir(
        results_root: Path,
        exp_label: str,
        result_group: str,
        analysis_label: str,
        win_len: float,
        win_overlap: float,
        fband,
        ) -> Path:
    """Construct the final grouped results directory for this analysis configuration."""
    return results_root / exp_label / result_group / _get_output_dirname(
        analysis_label,
        win_len,
        win_overlap,
        fband,
    )


def _normalize_round_digits(round_digits):
    """Validate optional CSV rounding precision."""
    if round_digits is None:
        return None
    if isinstance(round_digits, bool) or not isinstance(round_digits, int):
        raise ValueError('CSV_ROUND_DIGITS should be an integer or None')
    if round_digits < 0:
        raise ValueError('CSV_ROUND_DIGITS should be non-negative')
    return round_digits


def _normalize_coherence_threshold(coherence_threshold):
    """Validate optional coherence threshold for matrix masking."""
    if coherence_threshold is None:
        return None
    coherence_threshold = float(coherence_threshold)
    if not np.isfinite(coherence_threshold):
        raise ValueError('COHERENCE_THRESHOLD should be finite or None')
    if coherence_threshold < 0:
        raise ValueError('COHERENCE_THRESHOLD should be non-negative')
    if coherence_threshold > 1:
        raise ValueError('COHERENCE_THRESHOLD should not exceed 1')
    return coherence_threshold


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


def _get_matrix_png_name(coherence_threshold=None, masked: bool = False) -> str:
    """Construct one matrix-summary PNG filename."""
    coherence_threshold = _normalize_coherence_threshold(coherence_threshold)
    if masked and (coherence_threshold is None):
        raise ValueError('Masked matrix PNG naming requires a non-None COHERENCE_THRESHOLD')
    if not masked:
        return 'matrices.png'
    return f'matrices__thr_{_format_tag_value(coherence_threshold)}.png'


def _get_matrix_png_names(coherence_threshold):
    """Construct the plain and optional masked matrix-summary PNG filenames."""
    coherence_threshold = _normalize_coherence_threshold(coherence_threshold)
    names = [_get_matrix_png_name(masked=False)]
    if coherence_threshold is not None:
        names.append(_get_matrix_png_name(coherence_threshold, masked=True))
    return names


def _make_pair_label(pop_i: str, pop_j: str) -> str:
    """Construct one stable pair label."""
    return f'{pop_i}__{pop_j}'


def _get_coherence_cache_dir(dirpath_proc: Path) -> Path:
    """Construct the grouped coherence-cache directory."""
    return Path(dirpath_proc) / 'coherence_cache'


def _get_pop_selection_tag(pop_names) -> str:
    """Construct a compact stable tag for the selected population set."""
    pop_names = list(pop_names)
    if not pop_names:
        raise ValueError('At least one population is required for the coherence cache')
    digest = hashlib.sha1('|'.join(pop_names).encode('utf-8')).hexdigest()[:10]
    return f'n{len(pop_names)}_{digest}'


def _get_coherence_cache_name(
        analysis_label: str,
        pop_names,
        rate_dt: float,
        t_limits,
        win_len: float,
        win_overlap: float,
        fmax: float,
        ) -> str:
    """Construct the full coherence-cache filename."""
    return (
        f'{analysis_label}'
        f'__pops_{_get_pop_selection_tag(pop_names)}'
        f'__dt_{_format_tag_value(rate_dt)}'
        f'__t_{_format_window_tag(t_limits)}'
        f'__win_{_format_tag_value(win_len)}'
        f'__ov_{_format_tag_value(win_overlap)}'
        f'__fmax_{_format_tag_value(fmax)}'
        f'__{COHERCACHE_VERSION}.nc'
    )


def _get_coherence_cache_path(
        dirpath_proc: Path,
        analysis_label: str,
        pop_names,
        rate_dt: float,
        t_limits,
        win_len: float,
        win_overlap: float,
        fmax: float,
        ) -> Path:
    """Construct the full coherence-cache file path."""
    return _get_coherence_cache_dir(dirpath_proc) / _get_coherence_cache_name(
        analysis_label,
        pop_names,
        rate_dt,
        t_limits,
        win_len,
        win_overlap,
        fmax,
    )


def _print_auto_progress(pop_idx: int, pop_count: int, pop_name: str) -> None:
    """Print progress for auto-spectrum computation."""
    print(f'[auto {pop_idx:>3d}/{pop_count}] {pop_name}')


def _print_pair_progress(pair_idx: int, pair_count: int, pop_i: str, pop_j: str) -> None:
    """Print progress for pairwise CPSD/coherence computation."""
    print(f'[pair {pair_idx:>3d}/{pair_count}] {pop_i} vs {pop_j}')


def _compute_coherence_cache_dataset(rates, pop_names, pair_list):
    """Compute frequency-resolved CPSD and coherence for all pairs."""
    auto_psd = {}
    pop_count = len(pop_names)
    for pop_idx, pop_name in enumerate(pop_names, start=1):
        _print_auto_progress(pop_idx, pop_count, pop_name)
        rate_trace = rates.sel(pop=pop_name)
        auto_psd[pop_name] = calc_xr_cpsd(
            rate_trace,
            rate_trace,
            win_len=WIN_LEN,
            win_overlap=WIN_OVERLAP,
            fmax=FMAX,
            compute=True,
            store_proc_info=True,
        )

    pair_labels = []
    pop_i_vals = []
    pop_j_vals = []
    cpsd_vals = []
    coherence_vals = []
    freq_vals = None

    pair_count = len(pair_list)
    for pair_idx, (pop_i, pop_j) in enumerate(pair_list, start=1):
        _print_pair_progress(pair_idx, pair_count, pop_i, pop_j)

        if pop_i == pop_j:
            cpsd = auto_psd[pop_i]
        else:
            cpsd = calc_xr_cpsd(
                rates.sel(pop=pop_i),
                rates.sel(pop=pop_j),
                win_len=WIN_LEN,
                win_overlap=WIN_OVERLAP,
                fmax=FMAX,
                compute=True,
                store_proc_info=True,
            )

        if freq_vals is None:
            freq_vals = cpsd.freq.values

        denom = np.sqrt(np.abs(auto_psd[pop_i].values) * np.abs(auto_psd[pop_j].values))
        cpsd_arr = np.asarray(cpsd.values, dtype=np.complex128)
        coherence_arr = np.full(cpsd_arr.shape, np.nan + 0j, dtype=np.complex128)
        valid = np.isfinite(denom) & (denom > np.finfo(float).eps)
        coherence_arr[valid] = cpsd_arr[valid] / denom[valid]

        pair_labels.append(_make_pair_label(pop_i, pop_j))
        pop_i_vals.append(pop_i)
        pop_j_vals.append(pop_j)
        cpsd_vals.append(cpsd_arr)
        coherence_vals.append(coherence_arr)

    return xr.Dataset(
        data_vars={
            'cpsd': (['pair', 'freq'], np.asarray(cpsd_vals, dtype=np.complex128)),
            'coherence': (['pair', 'freq'], np.asarray(coherence_vals, dtype=np.complex128)),
        },
        coords={
            'pair': pair_labels,
            'freq': freq_vals,
            'pop_i': ('pair', pop_i_vals),
            'pop_j': ('pair', pop_j_vals),
        },
        attrs={
            'cache_version': COHERCACHE_VERSION,
            'analysis_label': ANALYSIS_LABEL,
            'rate_dt': RATE_DT,
            't_limits': list(T_LIMITS),
            'win_len': WIN_LEN,
            'win_overlap': WIN_OVERLAP,
            'fmax': FMAX,
        },
    )


def _load_or_compute_coherence_cache(rates, pop_names, pair_list, fpath_cache: Path):
    """Load an existing coherence cache or compute and save it."""
    if fpath_cache.exists():
        print(f'Loading cached coherence spectra: {fpath_cache}')
        return load_xr(fpath_cache, data_type='dataset', load=True)

    print(f'Computing coherence spectra and caching to: {fpath_cache}')
    coh_ds = _compute_coherence_cache_dataset(rates, pop_names, pair_list)
    save_xr(coh_ds, fpath_cache)
    return coh_ds


def _compute_band_mean_tables_from_cache(coh_ds, pop_names):
    """Derive band-averaged coherence magnitude and phase matrices from the cache."""
    fmask = (coh_ds.freq >= FBAND[0]) & (coh_ds.freq <= FBAND[1])
    if int(np.count_nonzero(fmask.values)) == 0:
        raise ValueError(f'No cached frequencies fall inside FBAND={FBAND}')

    band_mean = coh_ds['coherence'].where(fmask, drop=True).mean('freq')
    pop_index = {pop_name: idx for idx, pop_name in enumerate(pop_names)}
    coherence_table = np.full((len(pop_names), len(pop_names)), np.nan, dtype=float)
    phase_table = np.full((len(pop_names), len(pop_names)), np.nan, dtype=float)
    complex_by_pair = {}

    for pair_label in band_mean.pair.values.tolist():
        band_val = complex(band_mean.sel(pair=pair_label).item())
        pop_i = str(band_mean.sel(pair=pair_label).coords['pop_i'].item())
        pop_j = str(band_mean.sel(pair=pair_label).coords['pop_j'].item())
        idx_i = pop_index[pop_i]
        idx_j = pop_index[pop_j]

        if np.isfinite(np.real(band_val)) and np.isfinite(np.imag(band_val)):
            coh_mag = float(np.abs(band_val))
            phase_val = float(np.angle(band_val))
        else:
            coh_mag = np.nan
            phase_val = np.nan

        coherence_table[idx_i, idx_j] = coh_mag
        coherence_table[idx_j, idx_i] = coh_mag

        if pop_i == pop_j:
            phase_table[idx_i, idx_j] = 0.0 if np.isfinite(phase_val) else np.nan
        else:
            phase_table[idx_i, idx_j] = phase_val
            phase_table[idx_j, idx_i] = -phase_val if np.isfinite(phase_val) else np.nan

        complex_by_pair[pair_label] = band_val

    return coherence_table, phase_table, complex_by_pair


def _prepare_matrix_tables_for_plot(coherence_table, phase_table, coherence_threshold):
    """Prepare plotted coherence/phase tables and masks for visualization."""
    coherence_threshold = _normalize_coherence_threshold(coherence_threshold)
    coherence_plot = np.array(coherence_table, dtype=float, copy=True)
    phase_plot = np.array(phase_table, dtype=float, copy=True)
    diag_mask = np.eye(coherence_plot.shape[0], dtype=bool)
    if coherence_threshold is None:
        return coherence_plot, phase_plot, None, diag_mask

    weak_mask = np.isfinite(coherence_plot) & (coherence_plot < coherence_threshold)
    return coherence_plot, phase_plot, weak_mask, diag_mask


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


def _get_phase_plot_limit(phase_plot, plot_mask, fallback: float) -> float:
    """Get the symmetric phase limit from the phase cells visible in the current view."""
    visible_phase = np.array(phase_plot, dtype=float, copy=True)
    if plot_mask is not None:
        visible_phase[np.asarray(plot_mask, dtype=bool)] = np.nan
    return _get_symmetric_plot_limit(visible_phase, fallback=fallback)


def _overlay_coherence_mask(ax, weak_mask, edgecolor=(0.45, 0.45, 0.45, 0.95), hatch='///') -> None:
    """Overlay weak coherence cells with a transparent hatch."""
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
        coherence_table,
        phase_table,
        analysis_label: str,
        fband,
        coherence_threshold,
        use_mask: bool,
        ) -> None:
    """Render coherence magnitude and phase-difference matrices into one PNG."""
    coherence_plot, phase_plot, weak_mask, diag_mask = _prepare_matrix_tables_for_plot(
        coherence_table, phase_table, coherence_threshold
    )
    phase_mask = diag_mask if weak_mask is None else (diag_mask | weak_mask)
    phase_abs = _get_phase_plot_limit(
        phase_plot,
        phase_mask if use_mask else diag_mask,
        fallback=np.pi,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if use_mask and (weak_mask is not None):
        phase_cmap = plt.get_cmap('bwr').copy()
        phase_cmap.set_bad(color='white')
        phase_display = np.ma.masked_where(phase_mask, phase_plot)
        coherence_display = np.ma.masked_where(diag_mask, coherence_plot)
    else:
        phase_cmap = plt.get_cmap('bwr').copy()
        phase_cmap.set_bad(color='white')
        phase_display = np.ma.masked_where(diag_mask, phase_plot)
        coherence_display = np.ma.masked_where(diag_mask, coherence_plot)

    plot_specs = [
        ('Coherence', coherence_display, {'cmap': 'viridis', 'vmin': 0.0, 'vmax': 1.0}),
        ('Phase difference (rad)', phase_display, {'cmap': phase_cmap, 'vmin': -phase_abs, 'vmax': phase_abs}),
    ]

    for ax_idx, (ax, (title, table, image_kwargs)) in enumerate(zip(axes, plot_specs)):
        image = ax.imshow(table, aspect='auto', **image_kwargs)
        if use_mask and (ax_idx == 0):
            _overlay_coherence_mask(ax, weak_mask)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(pop_names)))
        ax.set_yticks(np.arange(len(pop_names)))
        ax.set_xticklabels(pop_names, rotation=45, ha='right')
        ax.set_yticklabels(pop_names)
        fig.colorbar(image, ax=ax, shrink=0.9)

    fig.suptitle(
        f'{analysis_label}: band-averaged firing-rate coherence ({_get_fband_tag(fband)})'
    )
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _write_metadata(
        fpath_md: Path,
        dirpath_out: Path,
        pop_names,
        pair_count: int,
        rate_cache_path: Path,
        coherence_cache_dir: Path,
        coherence_cache_path: Path,
        coherence_csv_name: str,
        phase_csv_name: str,
        matrix_png_names,
        requested_pop_names,
        csv_round_digits,
        coherence_threshold,
        ) -> None:
    """Write one Markdown metadata file for the analysis output folder."""
    params = {
        'ANALYSIS_LABEL': ANALYSIS_LABEL,
        'T_LIMITS': list(T_LIMITS),
        'RATE_DT': RATE_DT,
        'WIN_LEN': WIN_LEN,
        'WIN_OVERLAP': WIN_OVERLAP,
        'FMAX': FMAX,
        'FBAND': list(FBAND),
        'POP_NAMES': None if requested_pop_names is None else list(requested_pop_names),
        'CSV_ROUND_DIGITS': _normalize_round_digits(csv_round_digits),
        'COHERENCE_THRESHOLD': _normalize_coherence_threshold(coherence_threshold),
        'pair_enumeration': 'self pairs plus unordered cross-pop pairs in filtered pop order',
        'band_summary': 'complex mean of coherence over FBAND; magnitude and phase taken from that mean',
    }

    lines = [
        '# Rate Coherence Analysis',
        '',
        'Pairwise population-rate coherence and phase-difference matrices derived from cached CPSD spectra.',
        '',
        '## Paths',
        '',
        f'- Script: `{Path(__file__).resolve()}`',
        f'- Raw source: `{FPATH_SIM_RESULT.resolve()}`',
        f'- Rate cache used: `{rate_cache_path.resolve()}`',
        f'- Intermediate/cache root: `{DIRPATH_PROC.resolve()}`',
        f'- Coherence cache dir: `{coherence_cache_dir.resolve()}`',
        f'- Coherence cache file: `{coherence_cache_path.resolve()}`',
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
        f'- Coherence CSV: `{coherence_csv_name}`',
        f'- Phase CSV: `{phase_csv_name}`',
        (
            '- Matrix-summary PNGs: '
            + ', '.join(f'`{name}`' for name in matrix_png_names)
            if matrix_png_names else '- Matrix-summary PNGs: not generated'
        ),
        '',
        '## Populations',
        '',
        f'- Included populations: {", ".join(pop_names)}',
        f'- Number of analyzed pairs: {pair_count}',
        f'- Matrix plotting enabled: {bool(DO_PLOT_MATRICES)}',
        (
            f'- Coherence threshold: {_normalize_coherence_threshold(coherence_threshold)} '
            '(masked view hatches weak coherence cells and whitens their phase cells)'
        ),
    ]
    fpath_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    """Run the pairwise population rate coherence workflow."""
    dirpath_out = _get_output_dir(
        DIRPATH_RESULTS_ROOT,
        EXP_LABEL,
        RESULT_GROUP,
        ANALYSIS_LABEL,
        WIN_LEN,
        WIN_OVERLAP,
        FBAND,
    )
    coherence_csv_name = 'coherence.csv'
    phase_csv_name = 'phase.csv'
    matrix_png_names = _get_matrix_png_names(COHERENCE_THRESHOLD)

    sim_result = None
    rate_cache = get_rates_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    if not rate_cache.exists():
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = load_sim_result(FPATH_SIM_RESULT)

    rates = load_or_extract_rates(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    rates = rates.sel(time=slice(*T_LIMITS)).load()
    rates = _select_analysis_pops(rates, requested_pop_names=POP_NAMES)

    pop_names = rates.pop.values.tolist()
    pair_list = list(_iter_pop_pairs(pop_names))
    pair_count = len(pair_list)
    coherence_cache_dir = _get_coherence_cache_dir(DIRPATH_PROC)
    coherence_cache_path = _get_coherence_cache_path(
        DIRPATH_PROC,
        ANALYSIS_LABEL,
        pop_names,
        RATE_DT,
        T_LIMITS,
        WIN_LEN,
        WIN_OVERLAP,
        FMAX,
    )

    dirpath_out.mkdir(parents=True, exist_ok=True)
    print(f'Running {ANALYSIS_LABEL} for {pair_count} pairs in {dirpath_out}')
    coh_ds = _load_or_compute_coherence_cache(rates, pop_names, pair_list, coherence_cache_path)
    coherence_table, phase_table, _complex_by_pair = _compute_band_mean_tables_from_cache(
        coh_ds, pop_names
    )

    _write_metric_csv(
        dirpath_out / coherence_csv_name,
        pop_names,
        coherence_table,
        round_digits=CSV_ROUND_DIGITS,
    )
    _write_metric_csv(
        dirpath_out / phase_csv_name,
        pop_names,
        phase_table,
        round_digits=CSV_ROUND_DIGITS,
    )

    if DO_PLOT_MATRICES:
        print(f'Writing matrix PNGs to {dirpath_out}')
        _make_matrix_plot(
            dirpath_out / matrix_png_names[0],
            pop_names,
            coherence_table,
            phase_table,
            ANALYSIS_LABEL,
            FBAND,
            COHERENCE_THRESHOLD,
            use_mask=False,
        )
        if COHERENCE_THRESHOLD is not None:
            _make_matrix_plot(
                dirpath_out / matrix_png_names[1],
                pop_names,
                coherence_table,
                phase_table,
                ANALYSIS_LABEL,
                FBAND,
                COHERENCE_THRESHOLD,
                use_mask=True,
            )

    _write_metadata(
        dirpath_out / 'README.md',
        dirpath_out,
        pop_names,
        pair_count,
        rate_cache,
        coherence_cache_dir,
        coherence_cache_path,
        coherence_csv_name,
        phase_csv_name,
        matrix_png_names if DO_PLOT_MATRICES else [],
        POP_NAMES,
        CSV_ROUND_DIGITS,
        COHERENCE_THRESHOLD,
    )
    print(f'Saved results: {dirpath_out}')


if __name__ == '__main__':
    main()
