"""Demo for detecting alpha-band bursts in the cached A1 LFP with OEvent."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent

FPATH_LFP = (
    DIR_PACKAGE / 'dev_scratch' / 'data_proc' / 'a1_lfp_30s_0' / 'a1_lfp_30s_0_lfp.nc'
)
DIRPATH_OUT = DIR_PACKAGE / 'dev_scratch' / 'results' / 'a1_lfp_30s_0' / 'demo' / 'oevent_demo'

# Set the default analysis region.
T_START = 5.0
T_STOP = 30.0
Y = 1600.0
SIGNAL_KIND = 'csd'

# Set the default OEvent spectrogram parameters.
WINSZ = T_STOP - T_START
FREQMIN = 0.25
FREQMAX = 40.0
FREQSTEP = 0.25
GETPHASE = True
USELOGLFREQ = False
MSPECWIDTH = 7.0
NORMOP_NAME = 'mednorm'
NOISEAMP = 20.0

# Set the replayable OEvent detection parameters.
MEDTHRESH = 4.0
OVERLAPTH = 0.5
USE_DYN_THRESH = False
THRESHFCTR = 2.0
ENDFCTR = 0.5
ALPHA_BAND_HZ = (7.0, 15.0)

# Set the notebook-style preprocessing parameters.
OUTLIER_Z_THRESH = 8.0
OUTLIER_REL_NEIGHBOR_THRESH = 5.0

# Set the demo-specific post-filters.
MIN_NCYCLE = 3.0
MAX_FOCT = 1.5
MIN_FILTSIGCOR = 0.0
CSV_ROUND_DIGITS = 3

# Set the visualization-only parameters.
PLOT_XLIM = None
PLOT_FILTER_FBAND = None
PLOT_FILTER_ORDER = 3
BURST_EDGE_COLOR = 'red'


def _bootstrap_paths() -> None:
    """Expose the repo root on sys.path for local package imports."""
    # Add the local package root before importing shared helpers.
    path_str = str(DIR_REPO)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _prepare_matplotlib_cache() -> None:
    """Send Matplotlib cache writes to a repo-local directory."""
    # Keep Matplotlib cache writes inside the repo workspace.
    cache_dir = DIR_PACKAGE / '.mplcache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache_dir))


_bootstrap_paths()
_prepare_matplotlib_cache()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sim_data_analyzer.oevent_utils import (
    OEventAnalyzer,
    OEventDetectionParams,
    OEventSpectrogramBundle,
    OEventSpectrogramParams,
)
from sim_data_analyzer.signal_filters import filter_signal
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_signal import interp_time_outliers


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lfp-path', type=Path, default=FPATH_LFP)
    parser.add_argument('--out-dir', type=Path, default=DIRPATH_OUT)
    parser.add_argument('--signal-kind', type=str, choices=['lfp', 'csd'], default=SIGNAL_KIND)
    parser.add_argument('--y', type=float, default=Y)
    parser.add_argument('--t-start', type=float, default=T_START)
    parser.add_argument('--t-stop', type=float, default=T_STOP)
    parser.add_argument('--winsz', type=float, default=WINSZ)
    parser.add_argument('--freqmin', type=float, default=FREQMIN)
    parser.add_argument('--freqmax', type=float, default=FREQMAX)
    parser.add_argument('--freqstep', type=float, default=FREQSTEP)
    parser.add_argument('--mspecwidth', type=float, default=MSPECWIDTH)
    parser.add_argument('--noiseamp', type=float, default=NOISEAMP)
    parser.add_argument('--medthresh', type=float, default=MEDTHRESH)
    parser.add_argument('--overlapth', type=float, default=OVERLAPTH)
    parser.add_argument('--use-dyn-thresh', action='store_true', default=USE_DYN_THRESH)
    parser.add_argument('--threshfctr', type=float, default=THRESHFCTR)
    parser.add_argument('--endfctr', type=float, default=ENDFCTR)
    parser.add_argument('--min-ncycle', type=float, default=MIN_NCYCLE)
    parser.add_argument('--max-foct', type=float, default=MAX_FOCT)
    parser.add_argument('--min-filtsigcor', type=float, default=MIN_FILTSIGCOR)
    parser.add_argument('--csv-round-digits', type=int, default=CSV_ROUND_DIGITS)
    parser.add_argument('--verbose-oevent', action='store_true')
    return parser


def _format_tag_value(value: float) -> str:
    """Format one numeric value for a filename or cache tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _format_optional_window_tag(values) -> str:
    """Format one 2-point window for a filename tag."""
    return f'{_format_tag_value(values[0])}_{_format_tag_value(values[1])}'


def _build_run_tag(args: argparse.Namespace, resolved_y: float) -> str:
    """Build the output stem for one configured run."""
    run_tag = (
        f'oevent_alpha_{args.signal_kind}_y_{_format_tag_value(resolved_y)}'
        f'__t_{_format_tag_value(args.t_start)}_{_format_tag_value(args.t_stop)}'
    )
    if PLOT_XLIM is not None:
        run_tag += f'__xlim_{_format_optional_window_tag(PLOT_XLIM)}'
    if PLOT_FILTER_FBAND is not None:
        run_tag += f'__fplot_{_format_optional_window_tag(PLOT_FILTER_FBAND)}'
    return run_tag


def _build_signal_source(lfp: xr.DataArray, signal_kind: str) -> xr.DataArray:
    """Return the source array used for analysis."""
    if signal_kind == 'lfp':
        return lfp
    if signal_kind == 'csd':
        return calc_xr_csd(lfp, compute=True, store_proc_info=False)
    raise ValueError(f'Unsupported signal_kind: {signal_kind!r}')


def _build_plot_signal(signal: np.ndarray, sampr: float) -> np.ndarray:
    """Return the trace used for visualization only."""
    if PLOT_FILTER_FBAND is None:
        return np.asarray(signal, dtype=float)
    return filter_signal(
        np.asarray(signal, dtype=float),
        fs=float(sampr),
        fband=tuple(float(x) for x in PLOT_FILTER_FBAND),
        order=int(PLOT_FILTER_ORDER),
        btype='bandpass',
    )


def _make_spectrogram_params(
        args: argparse.Namespace,
        sampr: float,
        ) -> OEventSpectrogramParams:
    """Build the dataclass that controls spectrogram construction."""
    # Collect the Morlet settings in one reusable config block.
    return OEventSpectrogramParams(
        winsz=float(args.winsz),
        sampr=float(sampr),
        freqmin=float(args.freqmin),
        freqmax=float(args.freqmax),
        freqstep=float(args.freqstep),
        getphase=bool(GETPHASE),
        useloglfreq=bool(USELOGLFREQ),
        mspecwidth=float(args.mspecwidth),
        noiseamp=float(args.noiseamp),
        normop_name=NORMOP_NAME,
    )


def _make_detection_params(args: argparse.Namespace) -> OEventDetectionParams:
    """Build the dataclass that controls replayable event detection."""
    # Collect the threshold and band-label settings in one reusable config block.
    return OEventDetectionParams(
        medthresh=float(args.medthresh),
        overlapth=float(args.overlapth),
        use_dyn_thresh=bool(args.use_dyn_thresh),
        threshfctr=float(args.threshfctr),
        endfctr=float(args.endfctr),
        band_overrides={'alpha': tuple(float(x) for x in ALPHA_BAND_HZ)},
    )


def _finalize_alpha_event_table(
        alpha: pd.DataFrame,
        sampr: float,
        time_offset_s: float,
        resolved_y: float,
        ) -> pd.DataFrame:
    """Annotate and reorder one alpha-event table."""
    alpha = alpha.copy()
    alpha['y'] = resolved_y
    alpha['t_offset_s'] = float(time_offset_s)
    alpha['peak_time_s'] = time_offset_s + alpha['absPeakT'] / 1e3
    alpha['start_time_s'] = time_offset_s + alpha['absminT'] / 1e3
    alpha['stop_time_s'] = time_offset_s + alpha['absmaxT'] / 1e3
    alpha['duration_s'] = alpha['dur'] / 1e3
    alpha['sampr_hz'] = float(sampr)

    summary_cols = [
        'y',
        'selection_status',
        'peak_time_s',
        'start_time_s',
        'stop_time_s',
        'duration_s',
        'peakF',
        'ncycle',
        'Foct',
        'filtsigcor',
        'avgpow',
        'avgpowevent',
        'OSCscore',
    ]
    for col in summary_cols:
        if col not in alpha.columns:
            alpha[col] = np.nan
    alpha = alpha.sort_values('peak_time_s').reset_index(drop=True)
    return alpha[summary_cols + [col for col in alpha.columns if col not in summary_cols]]


def _is_csv_friendly_value(value) -> bool:
    """Return whether one value can be shown as a compact one-line CSV cell."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        return ('\n' not in text) and ('\r' not in text)
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _prepare_csv_table(alpha: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    """Drop cluttered columns and round numeric values for CSV export."""
    alpha_csv = alpha.copy()
    # Drop known bulky waveform columns from the exported summary table.
    alpha_csv = alpha_csv.drop(columns=['CSDwvf', 'filtsig'], errors='ignore')
    # Keep only scalar one-line columns in the CSV view.
    keep_cols = [
        col for col in alpha_csv.columns
        if alpha_csv[col].map(_is_csv_friendly_value).all()
    ]
    alpha_csv = alpha_csv.loc[:, keep_cols]
    # Round numeric columns for a more compact CSV view.
    numeric_cols = alpha_csv.select_dtypes(include=[np.number]).columns
    alpha_csv.loc[:, numeric_cols] = alpha_csv.loc[:, numeric_cols].round(int(round_digits))
    return alpha_csv


def _load_trace(
        fpath_lfp: Path,
        signal_kind: str,
        y_requested: float,
        t_start: float,
        t_stop: float,
        outlier_z_thresh: float,
        outlier_rel_neighbor_thresh: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float]:
    """Load one depth trace and return raw and preprocessed signals plus metadata."""
    if not fpath_lfp.exists():
        raise FileNotFoundError(f'LFP NetCDF file not found: {fpath_lfp}')
    if t_stop <= t_start:
        raise ValueError('Expected t_stop > t_start')

    lfp = xr.open_dataarray(fpath_lfp)
    if 'time' not in lfp.dims or 'y' not in lfp.dims:
        raise ValueError(f'Expected LFP array with dims including time and y, got {lfp.dims}')

    lfp = lfp.sel(time=slice(t_start, t_stop))
    if lfp.sizes['time'] < 2:
        raise ValueError('Selected time window is too short for burst detection')

    # Match the notebook preprocessing: time slice -> outlier interpolation -> mean subtraction.
    lfp_interp = interp_time_outliers(
        lfp,
        z_thresh=float(outlier_z_thresh),
        rel_neighbor_thresh=float(outlier_rel_neighbor_thresh),
    )

    # Build the requested analysis source from the raw and interpolated arrays.
    signal_src_raw = _build_signal_source(lfp, signal_kind=signal_kind)
    signal_src_interp = _build_signal_source(lfp_interp, signal_kind=signal_kind)

    # Extract one depth from the raw and interpolated source arrays.
    trace_raw = signal_src_raw.sel(y=y_requested, method='nearest').load()
    trace_interp = signal_src_interp.sel(y=y_requested, method='nearest').load()
    trace = trace_interp - trace_interp.mean(skipna=True)

    resolved_y = float(trace.coords['y'].item())
    time_s = trace.coords['time'].values.astype(float)
    signal_raw = np.asarray(trace_raw.values, dtype=float)
    signal_interp = np.asarray(trace_interp.values, dtype=float)
    signal = np.asarray(trace.values, dtype=float)

    interp_mask = np.isfinite(signal_raw) & np.isfinite(signal_interp)
    n_interpolated = int(np.count_nonzero(interp_mask & (~np.isclose(signal_raw, signal_interp))))
    removed_mean = float(np.mean(signal_interp))

    dt = float(time_s[1] - time_s[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f'Could not infer a positive dt from time coordinate: {dt}')
    sampr = float(round(1.0 / dt))
    return time_s, signal_raw, signal, resolved_y, sampr, n_interpolated, removed_mean


def _get_exp_label(fpath_lfp: Path) -> str:
    """Infer the experiment label from the cached LFP path."""
    return Path(fpath_lfp).parent.name


def _get_oevent_cache_dir(fpath_lfp: Path) -> Path:
    """Return the compute-or-load cache directory for OEvent spectrograms."""
    exp_label = _get_exp_label(fpath_lfp)
    return DIR_PACKAGE / 'dev_scratch' / 'data_proc' / exp_label / 'oevent_cache'


def _get_oevent_cache_name(
        fpath_lfp: Path,
        signal_kind: str,
        resolved_y: float,
        t_start: float,
        t_stop: float,
        spectrogram_params: OEventSpectrogramParams,
        ) -> str:
    """Build the deterministic spectrogram cache filename."""
    # Encode the preprocessing identity plus spectrogram settings in the cache key.
    preprocessing_tag = (
        f'kind_{signal_kind}'
        f'__y_{_format_tag_value(resolved_y)}'
        f'__t_{_format_tag_value(t_start)}_{_format_tag_value(t_stop)}'
        f'__outz_{_format_tag_value(OUTLIER_Z_THRESH)}'
        f'__outrel_{_format_tag_value(OUTLIER_REL_NEIGHBOR_THRESH)}'
        '__meansub_1'
    )
    spect_tag_parts = []
    for key, value in spectrogram_params.to_cache_dict().items():
        if isinstance(value, bool):
            spect_tag_parts.append(f'{key}_{int(value)}')
        elif isinstance(value, str):
            spect_tag_parts.append(f'{key}_{value}')
        else:
            spect_tag_parts.append(f'{key}_{_format_tag_value(float(value))}')
    exp_label = _get_exp_label(fpath_lfp)
    return f'{exp_label}__oevent_spec__{preprocessing_tag}__{"__".join(spect_tag_parts)}.pkl'


def _get_oevent_cache_path(
        fpath_lfp: Path,
        signal_kind: str,
        resolved_y: float,
        t_start: float,
        t_stop: float,
        spectrogram_params: OEventSpectrogramParams,
        ) -> Path:
    """Return the full cache path for one OEvent spectrogram bundle."""
    cache_dir = _get_oevent_cache_dir(fpath_lfp)
    cache_name = _get_oevent_cache_name(
        fpath_lfp,
        signal_kind=signal_kind,
        resolved_y=resolved_y,
        t_start=t_start,
        t_stop=t_stop,
        spectrogram_params=spectrogram_params,
    )
    return cache_dir / cache_name


def _oevent_stdout_context(verbose_oevent: bool):
    """Return a context that optionally suppresses OEvent stdout."""
    if verbose_oevent:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _load_or_compute_oevent_bundle(
        signal: np.ndarray,
        analyzer: OEventAnalyzer,
        fpath_lfp: Path,
        signal_kind: str,
        resolved_y: float,
        t_start: float,
        t_stop: float,
        verbose_oevent: bool,
        ) -> tuple[OEventSpectrogramBundle, Path, bool]:
    """Load a cached OEvent spectrogram bundle or compute and save it."""
    cache_path = _get_oevent_cache_path(
        fpath_lfp,
        signal_kind=signal_kind,
        resolved_y=resolved_y,
        t_start=t_start,
        t_stop=t_stop,
        spectrogram_params=analyzer.spectrogram_params,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f'Loading spectrogram from cache: {cache_path}')
        with cache_path.open('rb') as fobj:
            bundle = pickle.load(fobj)
        return bundle, cache_path, True

    # Compute the heavy Morlet bundle once before replaying thresholds.
    print(f'Started computing spectrogram: {cache_path}')
    with _oevent_stdout_context(verbose_oevent):
        bundle = analyzer.build_bundle(signal)
    print('Computed spectrogram bundle')
    with cache_path.open('wb') as fobj:
        pickle.dump(bundle, fobj, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved spectrogram to cache: {cache_path}')
    return bundle, cache_path, False


def _detect_alpha_events(
        signal: np.ndarray,
        analyzer: OEventAnalyzer,
        bundle: OEventSpectrogramBundle,
        detection_params: OEventDetectionParams,
        args: argparse.Namespace,
        time_offset_s: float,
        resolved_y: float,
        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run OEvent on a cached bundle and return kept/raw alpha events."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
        warnings.filterwarnings('ignore', message='Mean of empty slice.')
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')
        warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
        warnings.filterwarnings(
            'ignore',
            message='Setting an item of incompatible dtype is deprecated',
            category=FutureWarning,
        )
        with _oevent_stdout_context(args.verbose_oevent):
            dout = analyzer.detect_from_bundle(
                bundle,
                signal,
                detection_params=detection_params,
                MUA=None,
            )
            events = analyzer.to_dataframe(dout, signal, MUA=None, haveMUA=False).copy()

    # Split the raw alpha-labeled events from the later quality-filtered subset.
    alpha_all = events.loc[events['band'] == 'alpha'].copy()

    # Apply post-detection event quality filters.
    keep_mask = np.ones(len(alpha_all), dtype=bool)
    if args.min_ncycle is not None:
        keep_mask &= alpha_all['ncycle'].to_numpy(dtype=float) >= float(args.min_ncycle)
    if args.max_foct is not None:
        keep_mask &= alpha_all['Foct'].to_numpy(dtype=float) <= float(args.max_foct)
    if args.min_filtsigcor is not None:
        keep_mask &= alpha_all['filtsigcor'].to_numpy(dtype=float) >= float(args.min_filtsigcor)
    alpha_all['selection_status'] = np.where(keep_mask, 'passed', 'discarded')
    alpha_kept = alpha_all.loc[keep_mask].copy()

    alpha_all = _finalize_alpha_event_table(
        alpha_all,
        sampr=analyzer.spectrogram_params.sampr,
        time_offset_s=time_offset_s,
        resolved_y=resolved_y,
    )
    alpha_kept = _finalize_alpha_event_table(
        alpha_kept,
        sampr=analyzer.spectrogram_params.sampr,
        time_offset_s=time_offset_s,
        resolved_y=resolved_y,
    )
    return alpha_kept, alpha_all, dout


def _draw_alpha_event_frames(
        ax,
        events: pd.DataFrame,
        alpha_band_hz,
        edgecolor: str,
        linewidth: float,
        linestyle: str,
        marker: str,
        marker_size: float,
        label: str | None = None,
        ) -> None:
    """Draw alpha-event frames clipped to the alpha band plus peak markers."""
    alpha_lo, alpha_hi = [float(x) for x in alpha_band_hz]
    added_label = False
    for event in events.itertuples(index=False):
        disp_min_f = max(float(event.minF), alpha_lo)
        disp_max_f = min(float(event.maxF), alpha_hi)
        if disp_max_f <= disp_min_f:
            disp_min_f = max(min(float(event.peakF), alpha_hi), alpha_lo)
            disp_max_f = min(disp_min_f + 1e-6, alpha_hi)
        width = max(float(event.stop_time_s - event.start_time_s), np.finfo(float).eps)
        height = max(float(disp_max_f - disp_min_f), np.finfo(float).eps)
        rect = Rectangle(
            (float(event.start_time_s), disp_min_f),
            width,
            height,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if (label is not None and not added_label) else None,
        )
        ax.add_patch(rect)
        ax.plot(
            float(event.peak_time_s),
            float(event.peakF),
            marker=marker,
            color=edgecolor,
            markersize=marker_size,
            markeredgewidth=1.0,
        )
        added_label = True


def _make_overview_plot(
        fpath_out: Path,
        time_s: np.ndarray,
        signal_raw: np.ndarray,
        signal_plot: np.ndarray,
        alpha_events: pd.DataFrame,
        resolved_y: float,
        signal_kind: str,
        xlim=None,
        ) -> None:
    """Render raw and visualization traces and overlay detected alpha events."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [2.4, 1.0]},
    )

    ax_sig, ax_feat = axes

    # Show the raw trace in gray and the visualization trace in blue.
    ax_sig.plot(time_s, signal_raw, color='0.7', lw=0.8, alpha=0.8, label='raw')
    ax_sig.plot(time_s, signal_plot, color='#1f6feb', lw=1.0, label='plot')
    ax_sig.set_ylabel(signal_kind.upper())
    ax_sig.set_title(f'OEvent alpha-burst demo on {signal_kind.upper()}, y={resolved_y:g} um')
    ax_sig.legend(loc='upper right')
    ax_sig.grid(True, alpha=0.25)
    if xlim is not None:
        ax_sig.set_xlim(xlim)

    # Overlay the detected alpha event windows and their peak times.
    for event in alpha_events.itertuples(index=False):
        ax_sig.axvspan(event.start_time_s, event.stop_time_s, color='#f2cc60', alpha=0.25)
        ax_sig.axvline(event.peak_time_s, color='#d29922', lw=1.0, alpha=0.8)

    if alpha_events.empty:
        ax_feat.text(
            0.5,
            0.5,
            'No alpha events passed the current filters.',
            transform=ax_feat.transAxes,
            ha='center',
            va='center',
        )
        ax_feat.set_yticks([])
    else:
        # Plot event peak frequencies colored by oscillation score.
        ax_feat.scatter(
            alpha_events['peak_time_s'],
            alpha_events['peakF'],
            c=alpha_events['OSCscore'].fillna(0.0),
            cmap='viridis',
            s=50,
            edgecolor='k',
            linewidth=0.3,
        )
        ax_feat.set_ylabel('peakF (Hz)')
        ax_feat.grid(True, alpha=0.25)

    ax_feat.set_xlabel('time (s)')
    if xlim is not None:
        ax_feat.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _make_spectrogram_plot(
        fpath_out: Path,
        bundle: OEventSpectrogramBundle,
        alpha_raw_events: pd.DataFrame,
        alpha_kept_events: pd.DataFrame,
        time_offset_s: float,
        resolved_y: float,
        signal_kind: str,
        xlim=None,
        ) -> None:
    """Render the cached OEvent spectrogram and mark detected alpha bursts."""
    time_s = bundle.time_axis_s(time_offset_s=time_offset_s)
    freq_hz = bundle.freq_axis_hz()
    spec = bundle.stacked_tfr(normalized=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw the normalized spectrogram windows that OEvent generated.
    img = ax.imshow(
        spec,
        extent=(time_s[0], time_s[-1], freq_hz[0], freq_hz[-1]),
        origin='lower',
        aspect='auto',
        cmap=plt.get_cmap('jet'),
    )

    # Mark the configured alpha band for visual reference.
    alpha_band_hz = tuple(float(x) for x in ALPHA_BAND_HZ)
    ax.axhline(alpha_band_hz[0], color='white', lw=0.8, ls=':')
    ax.axhline(alpha_band_hz[1], color='white', lw=0.8, ls=':')

    # Draw raw alpha-labeled detections before the demo-specific quality filters.
    _draw_alpha_event_frames(
        ax,
        alpha_raw_events,
        alpha_band_hz=alpha_band_hz,
        edgecolor=BURST_EDGE_COLOR,
        linewidth=0.9,
        linestyle='--',
        marker='x',
        marker_size=5.0,
        label='raw alpha',
    )

    # Draw the kept alpha events in a stronger overlay.
    _draw_alpha_event_frames(
        ax,
        alpha_kept_events,
        alpha_band_hz=alpha_band_hz,
        edgecolor=BURST_EDGE_COLOR,
        linewidth=1.6,
        linestyle='-',
        marker='o',
        marker_size=4.5,
        label='kept alpha',
    )

    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(f'OEvent spectrogram on {signal_kind.upper()}, y={resolved_y:g} um')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc='upper right', framealpha=0.85)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('normalized power')

    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the demo and save the summary artifacts."""
    args = _build_parser().parse_args()

    # Load the selected trace with the same preprocessing used in the notebook.
    time_s, signal_raw, signal, resolved_y, sampr, n_interpolated, removed_mean = _load_trace(
        args.lfp_path,
        signal_kind=args.signal_kind,
        y_requested=args.y,
        t_start=args.t_start,
        t_stop=args.t_stop,
        outlier_z_thresh=OUTLIER_Z_THRESH,
        outlier_rel_neighbor_thresh=OUTLIER_REL_NEIGHBOR_THRESH,
    )

    # Build the spectrogram and detection configs explicitly.
    spectrogram_params = _make_spectrogram_params(args, sampr=sampr)
    detection_params = _make_detection_params(args)
    analyzer = OEventAnalyzer(spectrogram_params)

    # Load or compute the cached Morlet bundle in data_proc.
    bundle, cache_path, cache_hit = _load_or_compute_oevent_bundle(
        signal,
        analyzer=analyzer,
        fpath_lfp=args.lfp_path,
        signal_kind=args.signal_kind,
        resolved_y=resolved_y,
        t_start=args.t_start,
        t_stop=args.t_stop,
        verbose_oevent=args.verbose_oevent,
    )

    # Replay alpha detection from the cached spectrogram bundle.
    alpha_events, alpha_events_raw, _ = _detect_alpha_events(
        signal=signal,
        analyzer=analyzer,
        bundle=bundle,
        detection_params=detection_params,
        args=args,
        time_offset_s=float(time_s[0]),
        resolved_y=resolved_y,
    )
    signal_plot = _build_plot_signal(signal, sampr)

    # Build output paths and write the summary artifacts.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dirpath_csv = args.out_dir / 'csv'
    dirpath_csv.mkdir(parents=True, exist_ok=True)
    dirpath_2d = args.out_dir / '2d'
    dirpath_2d.mkdir(parents=True, exist_ok=True)
    run_tag = _build_run_tag(args, resolved_y)
    fpath_csv = dirpath_csv / f'{run_tag}.csv'
    fpath_png = args.out_dir / f'{run_tag}.png'
    fpath_spec_png = dirpath_2d / f'{run_tag}__2d.png'

    alpha_events_csv = _prepare_csv_table(alpha_events_raw, round_digits=args.csv_round_digits)
    alpha_events_csv.to_csv(fpath_csv, index=False)
    _make_overview_plot(
        fpath_png,
        time_s,
        signal_raw,
        signal_plot,
        alpha_events,
        resolved_y,
        signal_kind=args.signal_kind,
        xlim=PLOT_XLIM,
    )
    _make_spectrogram_plot(
        fpath_spec_png,
        bundle=bundle,
        alpha_raw_events=alpha_events_raw,
        alpha_kept_events=alpha_events,
        time_offset_s=float(time_s[0]),
        resolved_y=resolved_y,
        signal_kind=args.signal_kind,
        xlim=PLOT_XLIM,
    )

    # Print a compact run summary for the terminal.
    print(f'Input LFP: {args.lfp_path}')
    print(f'Signal kind: {args.signal_kind}')
    print(
        f'Selected depth y={resolved_y:g} um, time window=[{time_s[0]:.3f}, {time_s[-1]:.3f}] s, '
        f'sampling rate={sampr:g} Hz'
    )
    print(
        'Preprocessing: '
        f'interp_time_outliers(z_thresh={OUTLIER_Z_THRESH:g}, '
        f'rel_neighbor_thresh={OUTLIER_REL_NEIGHBOR_THRESH:g}), '
        f'mean subtraction (removed mean={removed_mean:+.6g}), '
        f'interpolated samples={n_interpolated}'
    )
    print(
        'OEvent spectrogram params: '
        f'{spectrogram_params.to_cache_dict()}'
    )
    print(
        'OEvent detection params: '
        f'medthresh={detection_params.medthresh:g}, '
        f'overlapth={detection_params.overlapth:g}, '
        f'use_dyn_thresh={detection_params.use_dyn_thresh}, '
        f'threshfctr={detection_params.threshfctr:g}, '
        f'endfctr={detection_params.endfctr:g}, '
        f'alpha_band={detection_params.band_overrides["alpha"]}'
    )
    print(
        'Visualization: '
        f'plot_filter_fband={PLOT_FILTER_FBAND}, '
        f'plot_filter_order={PLOT_FILTER_ORDER}, '
        f'plot_xlim={PLOT_XLIM}'
    )
    print(
        'CSV formatting: '
        f'round_digits={args.csv_round_digits}, '
        'drop_columns=["CSDwvf", "filtsig"], '
        'rows=raw alpha events with selection_status'
    )
    print(f'Spectrogram cache: {"hit" if cache_hit else "miss"} at {cache_path}')
    print(
        f'Alpha events kept: {len(alpha_events)} '
        f'(min_ncycle={args.min_ncycle}, max_foct={args.max_foct}, '
        f'min_filtsigcor={args.min_filtsigcor})'
    )
    print(f'Raw alpha-labeled events before filtering: {len(alpha_events_raw)}')
    if not alpha_events.empty:
        print(
            alpha_events[
                ['peak_time_s', 'duration_s', 'peakF', 'ncycle', 'Foct', 'filtsigcor', 'OSCscore']
            ]
            .head(10)
            .to_string(index=False)
        )
    print(f'Saved CSV: {fpath_csv}')
    print(f'Saved plot: {fpath_png}')
    print(f'Saved spectrogram plot: {fpath_spec_png}')


if __name__ == '__main__':
    main()
