"""Band-event analysis for cached LFP/CSD signals using OEvent."""

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import io
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.oevent_utils import (
    OEventAnalyzer,
    OEventDetectionParams,
    OEventSpectrogramParams,
    event_table_from_dataset,
    is_scalar_event_value,
    normalize_band_event_table,
    prepare_csv_event_table,
    resolve_xr_channel_selection,
)
from sim_data_analyzer.scratch_data import (
    get_exp_label,
    get_lfp_cache_path,
    get_proc_dir,
    load_or_extract_lfp,
    load_sim_result,
)
from sim_data_analyzer.signal_filters import filter_signal
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_signal import interp_time_outliers


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
DIRPATH_CFG_ROOT = DIR_PACKAGE / 'dev_scratch' / 'analysis' / 'configs' / 'oevent'

OEVENT_CFG_NAME = 'default'
EXP_LABEL = 'exp1'
RESULT_GROUP = 'oevent'
OEVENT_CFG_KEYS = {
    'band_overrides',
    'spectrogram_params',
    'detection_params',
    'preprocessing',
    'event_filters',
}

SIGNAL_KIND = 'csd'
T_LIMITS = (5, 30)

CHANNEL_MODE = 'multi'
Y_RANGE = (0, 3000)
Y_VALUES = None
Y, Y_STEP = None, None

BANDS_OF_INTEREST = ['alpha']
CSV_ROUND_DIGITS = 3

MAKE_PER_CHANNEL_OVERVIEW_PLOTS = 0
MAKE_SPECTROGRAM_PLOTS = 0
MAKE_STACKED_PLOT = 0

PLOT_XLIM = None
PLOT_FILTER_FBAND = (4, 25)
PLOT_FILTER_ORDER = 3
STACK_PLOT_T_RANGE = (10, 20)
STACK_PLOT_Y_RANGE = (2200, 2800)
STACK_TRACE_AMP_SCALE = 0.3
SPECT_EVENT_COLOR = 'red'

SPECTROGRAM_CACHE_VERSION = 'v1'
RESULT_CACHE_VERSION = 'v1'


def _read_json(fpath: Path) -> dict:
    """Read one JSON file into a dictionary."""
    return json.loads(Path(fpath).read_text(encoding='utf-8'))

def _get_cfg_path(cfg_name: str) -> Path:
    """Build the path to one named OEvent config file."""
    return DIRPATH_CFG_ROOT / f'{cfg_name}.json'

def _load_cfg_raw(cfg_name: str) -> dict:
    """Load one named OEvent config from disk."""
    fpath_cfg = _get_cfg_path(cfg_name)
    if not fpath_cfg.exists():
        raise FileNotFoundError(f'Missing OEvent config: {fpath_cfg}')
    cfg_raw = _read_json(fpath_cfg)
    _validate_cfg_raw(cfg_raw)
    return cfg_raw

def _validate_cfg_raw(cfg_raw: dict) -> None:
    """Validate that the JSON file contains only OEvent-related settings."""
    unknown = sorted(set(cfg_raw) - OEVENT_CFG_KEYS)
    if unknown:
        raise ValueError(f'OEvent config should only contain OEvent-related keys, got extras: {unknown}')
    missing = sorted(OEVENT_CFG_KEYS - set(cfg_raw))
    if missing:
        raise ValueError(f'OEvent config is missing required keys: {missing}')

def _extract_oevent_cfg(cfg_raw: dict) -> dict:
    """Project one config dictionary onto the OEvent-only keys."""
    return {
        key: cfg_raw[key]
        for key in sorted(OEVENT_CFG_KEYS)
        if key in cfg_raw
    }


CFG_RAW = _load_cfg_raw(OEVENT_CFG_NAME)
SRC_EXP_GROUP = get_exp_label(FPATH_SIM_RESULT)
DIRPATH_PROC = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)

BAND_OVERRIDES = {
    str(key): tuple(float(value) for value in values)
    for key, values in CFG_RAW.get('band_overrides', {}).items()
}

SPECTROGRAM_CFG = CFG_RAW['spectrogram_params']
DETECTION_CFG = CFG_RAW['detection_params']
PREPROC_CFG = CFG_RAW['preprocessing']
EVENT_FILTER_CFG = CFG_RAW['event_filters']

OUTLIER_Z_THRESH = float(PREPROC_CFG['outlier_z_thresh'])
OUTLIER_REL_NEIGHBOR_THRESH = float(PREPROC_CFG['outlier_rel_neighbor_thresh'])
MIN_NCYCLE = None if EVENT_FILTER_CFG.get('min_ncycle') is None else float(EVENT_FILTER_CFG['min_ncycle'])
MAX_FOCT = None if EVENT_FILTER_CFG.get('max_foct') is None else float(EVENT_FILTER_CFG['max_foct'])
MIN_FILTSIGCOR = (
    None if EVENT_FILTER_CFG.get('min_filtsigcor') is None else float(EVENT_FILTER_CFG['min_filtsigcor'])
)


def _format_tag_value(value: float) -> str:
    """Format one numeric value into a compact filesystem-safe tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _format_window_tag(values) -> str:
    """Format one numeric interval into a compact tag."""
    return f'{_format_tag_value(values[0])}_{_format_tag_value(values[1])}'


def _json_digest(payload: dict) -> str:
    """Build a short stable digest from one JSON-serializable payload."""
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def _normalize_bands_of_interest(bands_of_interest, detection_params: OEventDetectionParams) -> list[str]:
    """Validate the selected OEvent bands."""
    if not bands_of_interest:
        raise ValueError('BANDS_OF_INTEREST should contain at least one band')
    resolved_bands = detection_params.resolved_bands()
    normalized = [str(band) for band in bands_of_interest]
    missing = [band for band in normalized if band not in resolved_bands]
    if missing:
        raise ValueError(f'Unknown bands in BANDS_OF_INTEREST: {missing}')
    return normalized


def _normalize_round_digits(round_digits):
    """Validate optional CSV rounding precision."""
    if round_digits is None:
        return None
    if isinstance(round_digits, bool) or not isinstance(round_digits, int):
        raise ValueError('CSV_ROUND_DIGITS should be an integer or None')
    if round_digits < 0:
        raise ValueError('CSV_ROUND_DIGITS should be non-negative')
    return round_digits


def _get_band_tag(bands_of_interest) -> str:
    """Construct a compact selected-band tag."""
    normalized = [str(band) for band in bands_of_interest]
    if not normalized:
        raise ValueError('At least one band is required')
    return '-'.join(normalized)


def _get_requested_y_tag() -> str:
    """Construct a readable tag for the requested channel selection."""
    if CHANNEL_MODE == 'single':
        if Y is None:
            raise ValueError('Y should be set in single-channel mode')
        return f'y_{_format_tag_value(Y)}'
    if Y_RANGE is not None:
        return f'y_{_format_window_tag(Y_RANGE)}'
    if Y_VALUES is not None:
        values = [float(value) for value in Y_VALUES]
        if not values:
            raise ValueError('Y_VALUES should not be empty when provided')
        if len(values) <= 4:
            return 'y_' + '_'.join(_format_tag_value(value) for value in values)
        digest = hashlib.sha1('|'.join(f'{value:.9g}' for value in values).encode('utf-8')).hexdigest()[:8]
        return f'yvals_n{len(values)}_{digest}'
    return 'y_all'


def _get_channel_selection_tag(channel_y) -> str:
    """Construct a compact stable tag for the resolved channel set."""
    channel_y = np.asarray(channel_y, dtype=float)
    if channel_y.size == 0:
        raise ValueError('At least one channel should be selected')
    digest = hashlib.sha1('|'.join(f'{value:.9g}' for value in channel_y).encode('utf-8')).hexdigest()[:10]
    return f'n{channel_y.size}_{digest}'


def _get_run_tag(signal_kind: str, bands_of_interest) -> str:
    """Construct the compact tag shared by outputs and caches."""
    parts = [
        EXP_LABEL,
        signal_kind,
        _get_band_tag(bands_of_interest),
        f't_{_format_window_tag(T_LIMITS)}',
        _get_requested_y_tag(),
        f'oevcfg_{OEVENT_CFG_NAME}',
    ]
    return '__'.join(parts)


def _get_plot_suffix() -> str:
    """Construct the PNG suffix for plot-only visualization settings."""
    parts = []
    if PLOT_FILTER_FBAND is not None:
        parts.append(f'filt_{_format_window_tag(PLOT_FILTER_FBAND)}')
    return '' if not parts else '__' + '__'.join(parts)


def _get_stacked_plot_suffix() -> str:
    """Construct the stacked-plot suffix for view-only settings."""
    parts = []
    if STACK_PLOT_T_RANGE is not None:
        parts.append(f'tvis_{_format_window_tag(STACK_PLOT_T_RANGE)}')
    if STACK_PLOT_Y_RANGE is not None:
        parts.append(f'yvis_{_format_window_tag(STACK_PLOT_Y_RANGE)}')
    return '' if not parts else '__' + '__'.join(parts)


def _get_output_dir(results_root: Path, src_exp_group: str, signal_kind: str, bands_of_interest) -> Path:
    """Construct the grouped results directory for this analysis."""
    return results_root / src_exp_group / RESULT_GROUP / _get_run_tag(signal_kind, bands_of_interest)


def _get_oevent_cache_root(dirpath_proc: Path) -> Path:
    """Construct the shared OEvent cache root."""
    return Path(dirpath_proc) / 'oevent_cache'


def _get_spectrogram_cache_dir(dirpath_proc: Path) -> Path:
    """Construct the grouped spectrogram-cache directory."""
    return _get_oevent_cache_root(dirpath_proc) / 'spectrogram'


def _get_result_cache_dir(dirpath_proc: Path) -> Path:
    """Construct the grouped lightweight-result cache directory."""
    return _get_oevent_cache_root(dirpath_proc) / 'result'


def _get_mask_dir(dirpath_proc: Path) -> Path:
    """Construct the exported burst-mask directory."""
    return Path(dirpath_proc) / 'oevent_mask'


def _get_preprocessing_tag(signal_kind: str) -> str:
    """Construct the stable preprocessing tag used by both cache layers."""
    return (
        f'kind_{signal_kind}'
        f'__t_{_format_window_tag(T_LIMITS)}'
        f'__outz_{_format_tag_value(OUTLIER_Z_THRESH)}'
        f'__outrel_{_format_tag_value(OUTLIER_REL_NEIGHBOR_THRESH)}'
        '__meansub_1'
    )


def _get_result_filter_tag() -> str:
    """Construct the stable post-OEvent filter tag."""
    return (
        f'__ncyc_{_format_tag_value(MIN_NCYCLE if MIN_NCYCLE is not None else -1)}'
        f'__foct_{_format_tag_value(MAX_FOCT if MAX_FOCT is not None else -1)}'
        f'__fsc_{_format_tag_value(MIN_FILTSIGCOR if MIN_FILTSIGCOR is not None else -1)}'
    )


def _get_spectrogram_cache_path(
        dirpath_proc: Path,
        signal_kind: str,
        resolved_y: float,
        ) -> Path:
    """Construct the per-channel spectrogram cache path."""
    payload = {
        'preproc_tag': _get_preprocessing_tag(signal_kind),
        'cfg_name': OEVENT_CFG_NAME,
        'cfg_raw': CFG_RAW,
        'exp_label': EXP_LABEL,
        'resolved_y': float(resolved_y),
        'version': SPECTROGRAM_CACHE_VERSION,
    }
    fname = (
        f'spec__{_get_run_tag(signal_kind, BANDS_OF_INTEREST)}'
        f'__{_get_preprocessing_tag(signal_kind)}'
        f'__y_{_format_tag_value(resolved_y)}'
        f'__d_{_json_digest(payload)}'
        f'__{SPECTROGRAM_CACHE_VERSION}.pkl'
    )
    return _get_spectrogram_cache_dir(dirpath_proc) / fname


def _get_result_cache_path(
        dirpath_proc: Path,
        signal_kind: str,
        channel_y,
        ) -> Path:
    """Construct the lightweight manifest path for one run."""
    payload = {
        'preproc_tag': _get_preprocessing_tag(signal_kind),
        'result_filter_tag': _get_result_filter_tag(),
        'cfg_name': OEVENT_CFG_NAME,
        'cfg_raw': CFG_RAW,
        'exp_label': EXP_LABEL,
        'channels': [float(value) for value in np.asarray(channel_y, dtype=float).tolist()],
        'version': RESULT_CACHE_VERSION,
    }
    fname = (
        f'result_manifest__{_get_run_tag(signal_kind, BANDS_OF_INTEREST)}'
        f'__{_get_preprocessing_tag(signal_kind)}'
        f'{_get_result_filter_tag()}'
        f'__chs_{_get_channel_selection_tag(channel_y)}'
        f'__d_{_json_digest(payload)}'
        f'__{RESULT_CACHE_VERSION}.json'
    )
    return _get_result_cache_dir(dirpath_proc) / fname


def _get_mask_path(dirpath_proc: Path, signal_kind: str, bands_of_interest) -> Path:
    """Construct the exported in-burst mask path."""
    return _get_mask_dir(dirpath_proc) / f'{_get_run_tag(signal_kind, bands_of_interest)}.nc'


def _get_channel_result_cache_path(
        dirpath_proc: Path,
        signal_kind: str,
        resolved_y: float,
        ) -> Path:
    """Construct the lightweight per-channel result cache path."""
    payload = {
        'preproc_tag': _get_preprocessing_tag(signal_kind),
        'result_filter_tag': _get_result_filter_tag(),
        'cfg_name': OEVENT_CFG_NAME,
        'cfg_raw': CFG_RAW,
        'exp_label': EXP_LABEL,
        'resolved_y': float(resolved_y),
        'version': RESULT_CACHE_VERSION,
    }
    fname = (
        f'result_channel__{_get_run_tag(signal_kind, BANDS_OF_INTEREST)}'
        f'__{_get_preprocessing_tag(signal_kind)}'
        f'{_get_result_filter_tag()}'
        f'__y_{_format_tag_value(resolved_y)}'
        f'__d_{_json_digest(payload)}'
        f'__{RESULT_CACHE_VERSION}.nc'
    )
    return _get_result_cache_dir(dirpath_proc) / fname


def _build_signal_source(lfp: xr.DataArray, signal_kind: str) -> xr.DataArray:
    """Return the source array used for analysis."""
    if signal_kind == 'lfp':
        return lfp
    if signal_kind == 'csd':
        return calc_xr_csd(lfp, compute=True, store_proc_info=False)
    raise ValueError(f'Unsupported signal_kind: {signal_kind!r}')


def _load_cached_or_extracted_lfp(sim_result) -> xr.DataArray:
    """Load the cached LFP, falling back to xarray auto-detection when needed."""
    try:
        return load_or_extract_lfp(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    except OSError as exc:
        fpath_lfp = get_lfp_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
        if not fpath_lfp.exists():
            raise
        print(f'Falling back to direct xarray open for cached LFP: {fpath_lfp}')
        print(f'Original cache-open error: {exc}')
        return xr.open_dataarray(fpath_lfp).load()


def _build_plot_signal(signal: np.ndarray, sampr: float) -> np.ndarray:
    """Return the trace used for visualization only."""
    if PLOT_FILTER_FBAND is None:
        return np.asarray(signal, dtype=float)
    return filter_signal(
        np.asarray(signal, dtype=float),
        fs=float(sampr),
        fband=tuple(float(value) for value in PLOT_FILTER_FBAND),
        order=int(PLOT_FILTER_ORDER),
        btype='bandpass',
    )


def _encode_dataset_attr(value):
    """Convert one dataset attr value into a NetCDF-friendly scalar."""
    if value is None:
        return 'null'
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return value
    return json.dumps(value, sort_keys=True)


def _build_channel_result_dataset(
        signal_proc: np.ndarray,
        spectrogram_norm: np.ndarray,
        time_s: np.ndarray,
        spec_time_s: np.ndarray,
        freq_hz: np.ndarray,
        event_table: pd.DataFrame,
        resolved_y: float,
        channel_index: int,
        attrs: dict,
        ) -> xr.Dataset:
    """Pack one channel trace, spectrogram, and event table into a lightweight dataset."""
    signal_proc = np.asarray(signal_proc, dtype=float)
    spectrogram_norm = np.asarray(spectrogram_norm, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    spec_time_s = np.asarray(spec_time_s, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    event_table = event_table.reset_index(drop=True).copy()

    event_vars = {}
    keep_cols = [
        col for col in event_table.columns
        if event_table[col].map(is_scalar_event_value).all()
    ]
    for col in keep_cols:
        values = event_table[col].to_numpy()
        if pd.api.types.is_bool_dtype(event_table[col]):
            arr = np.asarray(values, dtype=bool)
        elif pd.api.types.is_numeric_dtype(event_table[col]):
            arr = np.asarray(values)
        else:
            arr = np.asarray(['' if pd.isna(value) else str(value) for value in values], dtype=str)
        event_vars[col] = ('event', arr)

    dataset = xr.Dataset(
        data_vars={
            'signal_proc': ('time', signal_proc),
            'spectrogram_norm': (['freq', 'spec_time'], spectrogram_norm),
            **event_vars,
        },
        coords={
            'time': time_s,
            'freq': freq_hz,
            'spec_time': spec_time_s,
            'event': np.arange(len(event_table), dtype=int),
        },
        attrs={
            **{key: _encode_dataset_attr(value) for key, value in attrs.items()},
            'channel_y': float(resolved_y),
            'channel_index': int(channel_index),
        },
    )
    return dataset


def _make_spectrogram_params(sampr: float) -> OEventSpectrogramParams:
    """Build the spectrogram-configuration dataclass."""
    winsz = SPECTROGRAM_CFG.get('winsz_s')
    if winsz is None:
        winsz = float(T_LIMITS[1] - T_LIMITS[0])
    return OEventSpectrogramParams(
        winsz=float(winsz),
        sampr=float(sampr),
        freqmin=float(SPECTROGRAM_CFG['freqmin']),
        freqmax=float(SPECTROGRAM_CFG['freqmax']),
        freqstep=float(SPECTROGRAM_CFG['freqstep']),
        getphase=bool(SPECTROGRAM_CFG['getphase']),
        useloglfreq=bool(SPECTROGRAM_CFG['useloglfreq']),
        mspecwidth=float(SPECTROGRAM_CFG['mspecwidth']),
        noiseamp=float(SPECTROGRAM_CFG['noiseamp']),
        normop_name=str(SPECTROGRAM_CFG['normop_name']),
    )


def _make_detection_params() -> OEventDetectionParams:
    """Build the detection-configuration dataclass."""
    return OEventDetectionParams(
        medthresh=float(DETECTION_CFG['medthresh']),
        overlapth=float(DETECTION_CFG['overlapth']),
        use_dyn_thresh=bool(DETECTION_CFG['use_dyn_thresh']),
        threshfctr=float(DETECTION_CFG['threshfctr']),
        endfctr=float(DETECTION_CFG['endfctr']),
        band_overrides=BAND_OVERRIDES,
    )


def _oevent_stdout_context() -> contextlib.AbstractContextManager:
    """Return a context that suppresses verbose OEvent stdout."""
    return contextlib.redirect_stdout(io.StringIO())


def _write_or_validate_result_cfg_copy(dirpath_out: Path) -> Path:
    """Write the used config into the results folder or raise on mismatch."""
    fpath_copy = dirpath_out / f'oevcfg_{OEVENT_CFG_NAME}.json'
    expected_text = json.dumps(CFG_RAW, indent=2, sort_keys=True) + '\n'
    if fpath_copy.exists():
        existing_raw = json.loads(fpath_copy.read_text(encoding='utf-8'))
        if _extract_oevent_cfg(existing_raw) != CFG_RAW:
            raise RuntimeError(
                f'Result folder already contains a different OEvent config: {fpath_copy}'
            )
        if fpath_copy.read_text(encoding='utf-8') != expected_text:
            fpath_copy.write_text(expected_text, encoding='utf-8')
        return fpath_copy
    fpath_copy.write_text(expected_text, encoding='utf-8')
    return fpath_copy


def _load_selected_signals(
        sim_result,
        signal_kind: str,
        channel_mode: str,
        y: float | None,
        y_values,
        y_range,
        y_step,
        ) -> tuple[xr.DataArray, xr.DataArray, np.ndarray, float, int]:
    """Load and preprocess the selected raw and processed channel set."""
    # Load the cached LFP once and trim it to the requested time window.
    lfp = _load_cached_or_extracted_lfp(sim_result)
    lfp = lfp.sel(time=slice(*T_LIMITS)).load()
    if lfp.sizes['time'] < 2:
        raise ValueError('Selected time window is too short for event detection')

    # Match the notebook preprocessing before any channel selection.
    lfp_interp = interp_time_outliers(
        lfp,
        z_thresh=float(OUTLIER_Z_THRESH),
        rel_neighbor_thresh=float(OUTLIER_REL_NEIGHBOR_THRESH),
    )

    # Build the requested signal kind from the raw and interpolated LFP.
    signal_raw = _build_signal_source(lfp, signal_kind=signal_kind)
    signal_interp = _build_signal_source(lfp_interp, signal_kind=signal_kind)
    signal_proc = signal_interp - signal_interp.mean(dim='time', skipna=True)

    # Resolve the final single- or multi-channel selection on both arrays.
    raw_selected, proc_selected = resolve_xr_channel_selection(
        signal_raw,
        signal_proc,
        channel_mode=channel_mode,
        y=y,
        y_values=y_values,
        y_range=y_range,
        y_step=y_step,
    )

    # Count interpolated samples on the resolved channel set for reporting.
    signal_raw_arr = np.asarray(raw_selected.values, dtype=float)
    signal_interp_arr = np.asarray(
        resolve_xr_channel_selection(
            signal_interp,
            signal_interp,
            channel_mode=channel_mode,
            y=y,
            y_values=y_values,
            y_range=y_range,
            y_step=y_step,
        )[0].values,
        dtype=float,
    )
    interp_mask = np.isfinite(signal_raw_arr) & np.isfinite(signal_interp_arr)
    n_interpolated = int(np.count_nonzero(interp_mask & (~np.isclose(signal_raw_arr, signal_interp_arr))))

    time_s = np.asarray(raw_selected.coords['time'].values, dtype=float)
    dt = float(time_s[1] - time_s[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f'Could not infer a positive dt from the time coordinate: {dt}')
    sampr = float(round(1.0 / dt))
    return raw_selected, proc_selected, time_s, sampr, n_interpolated


def _load_or_compute_spectrogram_bundle(
        signal: np.ndarray,
        analyzer: OEventAnalyzer,
        signal_kind: str,
        resolved_y: float,
        ) -> tuple[object, Path, bool]:
    """Load or compute one per-channel spectrogram bundle."""
    # Reuse one spectrogram bundle per resolved depth whenever possible.
    cache_path = _get_spectrogram_cache_path(
        DIRPATH_PROC,
        signal_kind=signal_kind,
        resolved_y=resolved_y,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f'Loading spectrogram from cache: {cache_path}')
        with cache_path.open('rb') as fobj:
            return pickle.load(fobj), cache_path, True

    print(f'Started computing spectrogram: {cache_path}')
    with _oevent_stdout_context():
        bundle = analyzer.build_bundle(signal)
    print('Computed spectrogram bundle')
    with cache_path.open('wb') as fobj:
        pickle.dump(bundle, fobj, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved spectrogram to cache: {cache_path}')
    return bundle, cache_path, False


def _load_or_compute_channel_result_cache(
        signal: np.ndarray,
        time_s: np.ndarray,
        sampr: float,
        signal_kind: str,
        resolved_y: float,
        channel_index: int,
        bands_of_interest,
        analyzer: OEventAnalyzer,
        detection_params: OEventDetectionParams,
        ) -> tuple[Path, bool, bool | None, Path | None]:
    """Load or compute one lightweight per-channel result cache."""
    # Save each channel result independently so bulky data can be released immediately.
    cache_path = _get_channel_result_cache_path(
        DIRPATH_PROC,
        signal_kind=signal_kind,
        resolved_y=resolved_y,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f'Loading band-event channel cache: {cache_path}')
        return cache_path, True, None, None

    bundle, bundle_path, bundle_hit = _load_or_compute_spectrogram_bundle(
        signal,
        analyzer=analyzer,
        signal_kind=signal_kind,
        resolved_y=resolved_y,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with _oevent_stdout_context():
            dout = analyzer.detect_from_bundle(bundle, signal, detection_params=detection_params, MUA=None)
            events = analyzer.to_dataframe(dout, signal, MUA=None, haveMUA=False).copy()
    selected_events, _passed_events = normalize_band_event_table(
        events,
        bands_of_interest=bands_of_interest,
        sampr=sampr,
        time_offset_s=float(time_s[0]),
        resolved_y=resolved_y,
        channel_index=channel_index,
        min_ncycle=MIN_NCYCLE,
        max_foct=MAX_FOCT,
        min_filtsigcor=MIN_FILTSIGCOR,
    )

    attrs = {
        'cfg_name': OEVENT_CFG_NAME,
        'exp_label': EXP_LABEL,
        'signal_kind': signal_kind,
        'bands_of_interest': list(bands_of_interest),
        'band_overrides': {key: list(value) for key, value in BAND_OVERRIDES.items()},
        'time_limits': list(T_LIMITS),
        'sampr_hz': float(sampr),
        'outlier_z_thresh': float(OUTLIER_Z_THRESH),
        'outlier_rel_neighbor_thresh': float(OUTLIER_REL_NEIGHBOR_THRESH),
        'min_ncycle': None if MIN_NCYCLE is None else float(MIN_NCYCLE),
        'max_foct': None if MAX_FOCT is None else float(MAX_FOCT),
        'min_filtsigcor': None if MIN_FILTSIGCOR is None else float(MIN_FILTSIGCOR),
        'spectrogram_cache_version': SPECTROGRAM_CACHE_VERSION,
        'result_cache_version': RESULT_CACHE_VERSION,
        'source_sim_result': str(FPATH_SIM_RESULT.resolve()),
        'source_lfp_cache': str(get_lfp_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT).resolve()),
    }
    dataset = _build_channel_result_dataset(
        signal_proc=signal,
        spectrogram_norm=bundle.stacked_tfr(normalized=True),
        time_s=time_s,
        spec_time_s=bundle.time_axis_s(time_offset_s=float(time_s[0])),
        freq_hz=bundle.freq_axis_hz(),
        event_table=selected_events,
        resolved_y=resolved_y,
        channel_index=channel_index,
        attrs=attrs,
    )
    save_xr(dataset, cache_path)
    del dataset, bundle, events, selected_events, dout
    gc.collect()
    print(f'Saved band-event channel cache: {cache_path}')
    return cache_path, False, bundle_hit, bundle_path


def _write_result_manifest(
        cache_path: Path,
        signal_kind: str,
        channel_y,
        channel_cache_paths,
        ) -> None:
    """Write one small manifest for the current channel-cache set."""
    manifest = {
        'cfg_name': OEVENT_CFG_NAME,
        'exp_label': EXP_LABEL,
        'signal_kind': signal_kind,
        'bands_of_interest': list(BANDS_OF_INTEREST),
        'channel_y': [float(value) for value in np.asarray(channel_y, dtype=float).tolist()],
        'channel_cache_paths': [str(Path(path).resolve()) for path in channel_cache_paths],
        'preproc_tag': _get_preprocessing_tag(signal_kind),
        'result_filter_tag': _get_result_filter_tag(),
        'result_cache_version': RESULT_CACHE_VERSION,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _ensure_channel_result_caches(
        raw_selected: xr.DataArray,
        proc_selected: xr.DataArray,
        time_s: np.ndarray,
        sampr: float,
        signal_kind: str,
        bands_of_interest,
        spectrogram_params: OEventSpectrogramParams,
        detection_params: OEventDetectionParams,
        ) -> tuple[Path, bool, list[Path], list[bool], list[bool], list[Path]]:
    """Ensure that lightweight per-channel result caches exist for the run."""
    # Process one channel at a time so spectrograms never accumulate in RAM.
    channel_y = np.asarray(raw_selected.coords['channel_y'].values, dtype=float)
    manifest_path = _get_result_cache_path(
        DIRPATH_PROC,
        signal_kind=signal_kind,
        channel_y=channel_y,
    )
    analyzer = OEventAnalyzer(spectrogram_params)
    channel_cache_paths = []
    channel_cache_hits = []
    spectrogram_cache_hits = []
    spectrogram_cache_paths = []
    for channel_index in range(raw_selected.sizes['channel']):
        resolved_y = float(channel_y[channel_index])
        signal = np.asarray(proc_selected.isel(channel=channel_index).values, dtype=float)
        channel_cache_path, channel_cache_hit, spectrogram_cache_hit, spectrogram_cache_path = (
            _load_or_compute_channel_result_cache(
                signal,
                time_s=time_s,
                sampr=sampr,
                signal_kind=signal_kind,
                resolved_y=resolved_y,
                channel_index=channel_index,
                bands_of_interest=bands_of_interest,
                analyzer=analyzer,
                detection_params=detection_params,
            )
        )
        channel_cache_paths.append(channel_cache_path)
        channel_cache_hits.append(channel_cache_hit)
        if spectrogram_cache_hit is not None and spectrogram_cache_path is not None:
            spectrogram_cache_hits.append(bool(spectrogram_cache_hit))
            spectrogram_cache_paths.append(spectrogram_cache_path)
    manifest_hit = manifest_path.exists() and all(Path(path).exists() for path in channel_cache_paths)
    _write_result_manifest(
        manifest_path,
        signal_kind=signal_kind,
        channel_y=channel_y,
        channel_cache_paths=channel_cache_paths,
    )
    return (
        manifest_path,
        bool(manifest_hit) and all(channel_cache_hits),
        channel_cache_paths,
        channel_cache_hits,
        spectrogram_cache_hits,
        spectrogram_cache_paths,
    )


def _get_band_colors(bands_of_interest) -> dict[str, tuple[float, float, float, float]]:
    """Assign stable plot colors to the selected bands."""
    cmap = plt.get_cmap('tab10')
    return {
        str(band): cmap(idx % cmap.N)
        for idx, band in enumerate(bands_of_interest)
    }


def _make_overview_plot(
        fpath_out: Path,
        time_s: np.ndarray,
        signal_plot: np.ndarray,
        passed_events: pd.DataFrame,
        resolved_y: float,
        signal_kind: str,
        band_colors: dict[str, tuple[float, float, float, float]],
        bands_of_interest,
        xlim=None,
        ) -> None:
    """Render one per-channel overview plot for the passed events."""
    # Show the raw trace, the plot trace, and the passed event windows together.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [2.4, 1.0]},
    )
    ax_sig, ax_feat = axes

    ax_sig.plot(time_s, signal_plot, color='#1f6feb', lw=1.0, label='plot')
    ax_sig.set_ylabel(signal_kind.upper())
    ax_sig.set_title(
        f'OEvent band events on {signal_kind.upper()}, y={resolved_y:g} um, '
        f'bands={",".join(bands_of_interest)}'
    )
    ax_sig.legend(loc='upper right')
    ax_sig.grid(True, alpha=0.25)
    if xlim is not None:
        ax_sig.set_xlim(xlim)

    for event in passed_events.itertuples(index=False):
        color = band_colors.get(str(event.event_band), '#f2cc60')
        ax_sig.axvspan(event.start_time_s, event.stop_time_s, color=color, alpha=0.20)
        ax_sig.axvline(event.peak_time_s, color=color, lw=1.0, alpha=0.85)

    if passed_events.empty:
        ax_feat.text(
            0.5,
            0.5,
            'No selected-band events passed the current filters.',
            transform=ax_feat.transAxes,
            ha='center',
            va='center',
        )
        ax_feat.set_yticks([])
    else:
        for band, band_events in passed_events.groupby('event_band'):
            ax_feat.scatter(
                band_events['peak_time_s'],
                band_events['peakF'],
                color=band_colors.get(str(band), '#d29922'),
                s=46,
                edgecolor='k',
                linewidth=0.3,
                label=str(band),
            )
        ax_feat.set_ylabel('peakF (Hz)')
        ax_feat.grid(True, alpha=0.25)
        ax_feat.legend(loc='upper right')

    ax_feat.set_xlabel('time (s)')
    if xlim is not None:
        ax_feat.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _scale_stack_traces(signal_plot, channel_y, trace_amp_scale: float) -> np.ndarray:
    """Scale one channel x time matrix into y-positioned stacked traces."""
    signal_plot = np.asarray(signal_plot, dtype=float)
    channel_y = np.asarray(channel_y, dtype=float)
    if signal_plot.ndim != 2:
        raise ValueError('signal_plot should have shape (channel, time)')
    if signal_plot.shape[0] != channel_y.size:
        raise ValueError('signal_plot and channel_y should share the same channel axis')
    if channel_y.size == 1:
        channel_spacing = 1.0
    else:
        channel_spacing = float(np.median(np.diff(np.sort(channel_y))))
    amp_ref = float(np.nanpercentile(np.abs(signal_plot), 95))
    if not np.isfinite(amp_ref) or amp_ref <= 0:
        amp_ref = 1.0
    trace_scale = float(trace_amp_scale) * channel_spacing / amp_ref
    return signal_plot * trace_scale + channel_y[:, None]


def _make_all_channels_plot(
        fpath_out: Path,
        time_s: np.ndarray,
        signal_plot: np.ndarray,
        channel_y,
        passed_events: pd.DataFrame,
        signal_kind: str,
        bands_of_interest,
        trace_amp_scale: float,
        t_range=None,
        y_range=None,
        xlim=None,
        ) -> None:
    """Render one stacked multi-channel trace plot with highlighted bursts."""
    # Stack the filtered traces on the physical y axis and highlight passed bursts.
    signal_plot = np.asarray(signal_plot, dtype=float)
    channel_y = np.asarray(channel_y, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    channel_mask = np.ones(channel_y.size, dtype=bool)
    time_mask = np.ones(time_s.size, dtype=bool)
    if y_range is not None:
        y0, y1 = [float(value) for value in y_range]
        if y1 < y0:
            y0, y1 = y1, y0
        channel_mask = (channel_y >= y0) & (channel_y <= y1)
    if t_range is not None:
        t0, t1 = [float(value) for value in t_range]
        if t1 < t0:
            t0, t1 = t1, t0
        time_mask = (time_s >= t0) & (time_s <= t1)
    if not np.any(channel_mask):
        raise ValueError('STACK_PLOT_Y_RANGE selected no channels')
    if not np.any(time_mask):
        raise ValueError('STACK_PLOT_T_RANGE selected no time points')
    channel_y_sel = channel_y[channel_mask]
    time_s_sel = time_s[time_mask]
    signal_plot_sel = signal_plot[channel_mask][:, time_mask]
    stacked = _scale_stack_traces(signal_plot_sel, channel_y_sel, trace_amp_scale=trace_amp_scale)
    fig, ax = plt.subplots(figsize=(12, 8))
    selected_indices = np.flatnonzero(channel_mask)
    for local_index, channel_index in enumerate(selected_indices):
        y_value = float(channel_y[channel_index])
        ax.plot(time_s_sel, stacked[local_index], color='#1f6feb', lw=0.9, alpha=0.95)
        channel_events = passed_events.loc[passed_events['event_channel'] == int(channel_index)]
        for event in channel_events.itertuples(index=False):
            mask = (time_s_sel >= float(event.start_time_s)) & (time_s_sel <= float(event.stop_time_s))
            if not np.any(mask):
                continue
            ax.plot(time_s_sel[mask], stacked[local_index, mask], color=SPECT_EVENT_COLOR, lw=2.0, alpha=0.95)

    ax.set_xlabel('time (s)')
    ax.set_ylabel('y (um)')
    ax.set_title(
        f'Stacked {signal_kind.upper()} traces with passed band events, '
        f'bands={",".join(bands_of_interest)}'
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.18)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _draw_event_frames(
        ax,
        events: pd.DataFrame,
        resolved_bands: dict[str, tuple[float, float]],
        linewidth: float,
        linestyle: str,
        marker: str,
        marker_size: float,
        label_suffix: str,
        event_color: str,
        ) -> None:
    """Draw event frames clipped to their configured band ranges."""
    first_label = True
    for event in events.itertuples(index=False):
        band_name = str(event.event_band)
        band_lo, band_hi = resolved_bands[band_name]
        disp_min_f = max(float(event.minF), float(band_lo))
        disp_max_f = min(float(event.maxF), float(band_hi))
        if disp_max_f <= disp_min_f:
            disp_min_f = max(min(float(event.peakF), float(band_hi)), float(band_lo))
            disp_max_f = min(disp_min_f + 1e-6, float(band_hi))
        rect = Rectangle(
            (float(event.start_time_s), disp_min_f),
            max(float(event.stop_time_s - event.start_time_s), np.finfo(float).eps),
            max(float(disp_max_f - disp_min_f), np.finfo(float).eps),
            fill=False,
            edgecolor=event_color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label_suffix if first_label else None,
        )
        ax.add_patch(rect)
        ax.plot(
            float(event.peak_time_s),
            float(event.peakF),
            marker=marker,
            color=event_color,
            markersize=marker_size,
            markeredgewidth=1.0,
            linestyle='None',
            label=None,
        )
        first_label = False


def _make_spectrogram_plot(
        fpath_out: Path,
        spec_time_s: np.ndarray,
        freq_hz: np.ndarray,
        spectrogram_norm: np.ndarray,
        raw_events: pd.DataFrame,
        passed_events: pd.DataFrame,
        resolved_bands: dict[str, tuple[float, float]],
        band_colors: dict[str, tuple[float, float, float, float]],
        resolved_y: float,
        signal_kind: str,
        bands_of_interest,
        xlim=None,
        ) -> None:
    """Render one per-channel spectrogram plot for the selected bands."""
    # Show the cached spectrogram with raw and passed event frames overlaid.
    fig, ax = plt.subplots(figsize=(12, 5))
    img = ax.imshow(
        spectrogram_norm,
        extent=(spec_time_s[0], spec_time_s[-1], freq_hz[0], freq_hz[-1]),
        origin='lower',
        aspect='auto',
        cmap=plt.get_cmap('jet'),
    )
    for band in bands_of_interest:
        band_lo, band_hi = resolved_bands[str(band)]
        ax.axhline(float(band_lo), color=SPECT_EVENT_COLOR, lw=0.8, ls=':', alpha=0.8)
        ax.axhline(float(band_hi), color=SPECT_EVENT_COLOR, lw=0.8, ls=':', alpha=0.8)

    _draw_event_frames(
        ax,
        raw_events,
        resolved_bands=resolved_bands,
        linewidth=1.0,
        linestyle='--',
        marker='x',
        marker_size=6.0,
        label_suffix='dropped/raw',
        event_color=SPECT_EVENT_COLOR,
    )
    _draw_event_frames(
        ax,
        passed_events,
        resolved_bands=resolved_bands,
        linewidth=1.7,
        linestyle='-',
        marker='o',
        marker_size=5.0,
        label_suffix='passed',
        event_color=SPECT_EVENT_COLOR,
    )

    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title(
        f'OEvent spectrogram on {signal_kind.upper()}, y={resolved_y:g} um, '
        f'bands={",".join(bands_of_interest)}'
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc='upper right', framealpha=0.85)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('normalized power')
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _write_metadata(
        fpath_md: Path,
        dirpath_out: Path,
        fpath_cfg_copy: Path,
        result_cache_path: Path,
        fpath_mask: Path,
        spectrogram_cache_paths,
        channel_y,
        bands_of_interest,
        signal_kind: str,
        result_cache_hit: bool,
        spectrogram_cache_hits,
        channel_cache_hits,
        ) -> None:
    """Write one short Markdown metadata file for the run."""
    params = {
        'OEVENT_CFG_NAME': OEVENT_CFG_NAME,
        'EXP_LABEL': EXP_LABEL,
        'SIGNAL_KIND': signal_kind,
        'T_LIMITS': list(T_LIMITS),
        'CHANNEL_MODE': CHANNEL_MODE,
        'Y': Y,
        'Y_VALUES': Y_VALUES,
        'Y_RANGE': Y_RANGE,
        'Y_STEP': Y_STEP,
        'BANDS_OF_INTEREST': list(bands_of_interest),
        'BAND_OVERRIDES': {key: list(value) for key, value in BAND_OVERRIDES.items()},
        'CSV_ROUND_DIGITS': _normalize_round_digits(CSV_ROUND_DIGITS),
        'PLOT_FILTER_FBAND': None if PLOT_FILTER_FBAND is None else list(PLOT_FILTER_FBAND),
        'PLOT_XLIM': None if PLOT_XLIM is None else list(PLOT_XLIM),
        'STACK_PLOT_T_RANGE': None if STACK_PLOT_T_RANGE is None else list(STACK_PLOT_T_RANGE),
        'STACK_PLOT_Y_RANGE': None if STACK_PLOT_Y_RANGE is None else list(STACK_PLOT_Y_RANGE),
        'MAKE_PER_CHANNEL_OVERVIEW_PLOTS': bool(MAKE_PER_CHANNEL_OVERVIEW_PLOTS),
        'MAKE_SPECTROGRAM_PLOTS': bool(MAKE_SPECTROGRAM_PLOTS),
        'MAKE_STACKED_PLOT': bool(MAKE_STACKED_PLOT),
        'STACK_TRACE_AMP_SCALE': float(STACK_TRACE_AMP_SCALE),
        'SPECT_EVENT_COLOR': SPECT_EVENT_COLOR,
    }
    lines = [
        '# OEvent Band-Event Analysis',
        '',
        'Independent per-channel OEvent analysis over cached LFP/CSD traces with a lightweight result cache.',
        '',
        '## Paths',
        '',
        f'- Script: `{Path(__file__).resolve()}`',
        f'- Raw source: `{FPATH_SIM_RESULT.resolve()}`',
        f'- LFP cache: `{get_lfp_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT).resolve()}`',
        f'- OEvent config: `{_get_cfg_path(OEVENT_CFG_NAME).resolve()}`',
        f'- Copied config: `{fpath_cfg_copy.resolve()}`',
        f'- Intermediate/cache root: `{DIRPATH_PROC.resolve()}`',
        f'- Result-cache manifest: `{result_cache_path.resolve()}`',
        f'- In-burst mask: `{fpath_mask.resolve()}`',
        f'- Results folder: `{dirpath_out.resolve()}`',
        '',
        '## Parameters',
        '',
        '```json',
        json.dumps(params, indent=2),
        '```',
        '',
        '## Channels',
        '',
        f'- Resolved channel depths: {", ".join(f"{float(value):g}" for value in np.asarray(channel_y, dtype=float))}',
        f'- Number of channels: {len(channel_y)}',
        '',
        '## Cache Status',
        '',
        f'- Result manifest hit: {bool(result_cache_hit)}',
        f'- Channel result cache hits: {sum(bool(x) for x in channel_cache_hits)} / {len(channel_cache_hits)}',
        (
            f'- Spectrogram cache hits: {sum(bool(x) for x in spectrogram_cache_hits)} / {len(spectrogram_cache_hits)}'
            if spectrogram_cache_hits else '- Spectrogram cache hits: reused via lightweight result cache'
        ),
        (
            '- Spectrogram cache files: '
            + ', '.join(f'`{Path(path).name}`' for path in spectrogram_cache_paths)
            if spectrogram_cache_paths else '- Spectrogram cache files: reused via lightweight result cache'
        ),
    ]
    fpath_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    """Run the band-event workflow on one or many selected channels."""
    # Load the raw simulation only if the shared LFP cache is missing.
    sim_result = None
    lfp_cache_path = get_lfp_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    if not lfp_cache_path.exists():
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = load_sim_result(FPATH_SIM_RESULT)

    # Resolve the requested bands and channels before touching OEvent.
    detection_params = _make_detection_params()
    bands_of_interest = _normalize_bands_of_interest(BANDS_OF_INTEREST, detection_params)
    signal_kind = str(SIGNAL_KIND).strip().lower()
    raw_selected, proc_selected, time_s, sampr, n_interpolated = _load_selected_signals(
        sim_result,
        signal_kind=signal_kind,
        channel_mode=CHANNEL_MODE,
        y=Y,
        y_values=Y_VALUES,
        y_range=Y_RANGE,
        y_step=Y_STEP,
    )
    spectrogram_params = _make_spectrogram_params(sampr=sampr)
    channel_y = np.asarray(raw_selected.coords['channel_y'].values, dtype=float)
    dirpath_out = _get_output_dir(DIRPATH_RESULTS_ROOT, SRC_EXP_GROUP, signal_kind, bands_of_interest)
    result_cache_path = _get_result_cache_path(
        DIRPATH_PROC,
        signal_kind=signal_kind,
        channel_y=channel_y,
    )
    fpath_mask = _get_mask_path(DIRPATH_PROC, signal_kind=signal_kind, bands_of_interest=bands_of_interest)

    # Create the results folder and guard it with the copied config file.
    dirpath_out.mkdir(parents=True, exist_ok=True)
    fpath_cfg_copy = _write_or_validate_result_cfg_copy(dirpath_out)
    print(
        f'Running band-event analysis for {len(channel_y)} channel(s) in {dirpath_out} '
        f'with bands={bands_of_interest}, cfg={OEVENT_CFG_NAME}, exp={EXP_LABEL}'
    )

    # Build or reload the lightweight per-channel caches before producing outputs.
    (
        result_cache_path,
        result_cache_hit,
        channel_cache_paths,
        channel_cache_hits,
        spectrogram_cache_hits,
        spectrogram_cache_paths,
    ) = _ensure_channel_result_caches(
            raw_selected,
            proc_selected,
            time_s=time_s,
            sampr=sampr,
            signal_kind=signal_kind,
            bands_of_interest=bands_of_interest,
            spectrogram_params=spectrogram_params,
            detection_params=detection_params,
        )
    del raw_selected, proc_selected
    gc.collect()

    # Reconstruct event tables and plots by streaming one channel cache at a time.
    band_colors = _get_band_colors(bands_of_interest)
    resolved_bands = detection_params.resolved_bands()

    # Prepare the output layout and write the combined CSV summary first.
    dirpath_csv = dirpath_out / 'csv'
    dirpath_overview = dirpath_out / 'overview_pngs'
    dirpath_spect = dirpath_out / 'spectrogram_pngs'
    for path in [dirpath_csv]:
        path.mkdir(parents=True, exist_ok=True)
    if MAKE_PER_CHANNEL_OVERVIEW_PLOTS:
        dirpath_overview.mkdir(parents=True, exist_ok=True)
    if MAKE_SPECTROGRAM_PLOTS:
        dirpath_spect.mkdir(parents=True, exist_ok=True)

    # Stream channel caches for plots and combine only the small event tables in memory.
    event_tables = []
    passed_event_tables = []
    stacked_plot_signals = []
    burst_mask = np.zeros((len(channel_y), len(time_s)), dtype=np.uint8)
    plot_suffix = _get_plot_suffix()
    for channel_cache_path in channel_cache_paths:
        channel_ds = load_xr(channel_cache_path, data_type='dataset', load=False)
        channel_index = int(channel_ds.attrs['channel_index'])
        resolved_y = float(channel_ds.attrs['channel_y'])
        proc_trace = np.asarray(channel_ds['signal_proc'].values, dtype=float)
        plot_trace = _build_plot_signal(proc_trace, sampr)
        event_table_channel = event_table_from_dataset(channel_ds)
        passed_channel_events = event_table_channel.loc[event_table_channel['event_passed'].astype(bool)].copy()
        for event in passed_channel_events.itertuples(index=False):
            mask = (time_s >= float(event.start_time_s)) & (time_s <= float(event.stop_time_s))
            burst_mask[channel_index, mask] = 1
        event_tables.append(event_table_channel)
        passed_event_tables.append(passed_channel_events)
        if MAKE_STACKED_PLOT:
            stacked_plot_signals.append(plot_trace)
        channel_tag = f'y_{_format_tag_value(resolved_y)}'

        if MAKE_PER_CHANNEL_OVERVIEW_PLOTS:
            _make_overview_plot(
                dirpath_overview / f'{channel_tag}{plot_suffix}.png',
                time_s,
                plot_trace,
                passed_channel_events,
                resolved_y=resolved_y,
                signal_kind=signal_kind,
                band_colors=band_colors,
                bands_of_interest=bands_of_interest,
                xlim=PLOT_XLIM,
            )
        if MAKE_SPECTROGRAM_PLOTS:
            spec_time_s = np.asarray(channel_ds.coords['spec_time'].values, dtype=float)
            freq_hz = np.asarray(channel_ds.coords['freq'].values, dtype=float)
            spec_trace = np.asarray(channel_ds['spectrogram_norm'].values, dtype=float)
            _make_spectrogram_plot(
                dirpath_spect / f'{channel_tag}{plot_suffix}.png',
                spec_time_s,
                freq_hz,
                spec_trace,
                event_table_channel,
                passed_channel_events,
                resolved_bands=resolved_bands,
                band_colors=band_colors,
                resolved_y=resolved_y,
                signal_kind=signal_kind,
                bands_of_interest=bands_of_interest,
                xlim=PLOT_XLIM,
            )
        channel_ds.close()
        del channel_ds

    event_table = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    passed_events = pd.concat(passed_event_tables, ignore_index=True) if passed_event_tables else pd.DataFrame()
    csv_table = prepare_csv_event_table(event_table, round_digits=CSV_ROUND_DIGITS)
    fpath_csv = dirpath_csv / 'events.csv'
    csv_table.to_csv(fpath_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    mask_da = xr.DataArray(
        burst_mask,
        dims=('y', 'time'),
        coords={'y': channel_y, 'time': time_s},
        name='oevent_in_burst_mask',
        attrs={
            'signal_kind': signal_kind,
            'bands_of_interest': json.dumps(list(bands_of_interest)),
            'cfg_name': OEVENT_CFG_NAME,
            'exp_label': EXP_LABEL,
            'value_meaning': '1=in passed burst, 0=out of burst',
        },
    )
    save_xr(mask_da, fpath_mask)

    # Export one stacked multi-channel plot when more than one trace is shown.
    if MAKE_STACKED_PLOT and stacked_plot_signals:
        stacked_suffix = plot_suffix + _get_stacked_plot_suffix()
        _make_all_channels_plot(
            dirpath_out / f'all_channels{stacked_suffix}.png',
            time_s=time_s,
            signal_plot=np.stack(stacked_plot_signals, axis=0),
            channel_y=channel_y,
            passed_events=passed_events,
            signal_kind=signal_kind,
            bands_of_interest=bands_of_interest,
            trace_amp_scale=STACK_TRACE_AMP_SCALE,
            t_range=STACK_PLOT_T_RANGE,
            y_range=STACK_PLOT_Y_RANGE,
            xlim=PLOT_XLIM,
        )

    # Save a short metadata summary alongside the figures and CSV table.
    _write_metadata(
        dirpath_out / 'README.md',
        dirpath_out=dirpath_out,
        fpath_cfg_copy=fpath_cfg_copy,
        result_cache_path=result_cache_path,
        fpath_mask=fpath_mask,
        spectrogram_cache_paths=spectrogram_cache_paths,
        channel_y=channel_y,
        bands_of_interest=bands_of_interest,
        signal_kind=signal_kind,
        result_cache_hit=result_cache_hit,
        spectrogram_cache_hits=spectrogram_cache_hits,
        channel_cache_hits=channel_cache_hits,
    )

    # Print a compact terminal summary for the current run.
    print(f'Signal kind: {signal_kind}')
    print(f'Config name: {OEVENT_CFG_NAME}')
    print(f'Experiment label: {EXP_LABEL}')
    print(f'Channels analyzed: {len(channel_y)} at depths {channel_y.tolist()}')
    print(f'Sampling rate: {sampr:g} Hz')
    print(f'Interpolated samples: {n_interpolated}')
    print(f'Result manifest: {"hit" if result_cache_hit else "miss"} at {result_cache_path}')
    print(f'Channel result cache hits: {sum(bool(x) for x in channel_cache_hits)} / {len(channel_cache_hits)}')
    if spectrogram_cache_hits:
        print(
            f'Spectrogram cache hits: {sum(bool(x) for x in spectrogram_cache_hits)} / '
            f'{len(spectrogram_cache_hits)}'
        )
    print(f'Selected-band events: {len(event_table)} total, {len(passed_events)} passed')
    print(f'Copied config: {fpath_cfg_copy}')
    print(f'Saved in-burst mask: {fpath_mask}')
    print(f'Saved CSV: {fpath_csv}')
    if MAKE_STACKED_PLOT:
        print(f'Saved stacked channels PNG: {dirpath_out / f"all_channels{plot_suffix + _get_stacked_plot_suffix()}.png"}')
    if MAKE_PER_CHANNEL_OVERVIEW_PLOTS:
        print(f'Saved overview PNGs to: {dirpath_overview}')
    if MAKE_SPECTROGRAM_PLOTS:
        print(f'Saved spectrogram PNGs to: {dirpath_spect}')


if __name__ == '__main__':
    main()
