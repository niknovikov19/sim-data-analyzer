"""Spike-triggered LFP/CSD analysis with one cached depth x time STA average."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
from sim_data_analyzer.spike_data import SpikeData
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_signal import interp_time_outliers
from sim_data_analyzer.xr_spike_triggered import calc_xr_sta


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
FPATH_LAYER_CONFIG = DIR_PACKAGE / 'dev_scratch' / 'analysis' / 'configs' / 'layers' / 'default.json'

RESULT_GROUP = 'sta'
TRIGGER_POP = 'ITS4'
SIGNAL_TYPE = 'csd'
SPIKE_T_LIMITS_S = (5.0, 30.0)
TIME_WIN_MS = (-100.0, 100.0)
SUBTRACT_CHAN_MEAN = True

MAKE_PLOT_1D = 0
MAKE_PLOT_2D = 1
PLOT_Y = None
SHOW_ZERO_LINE = False
SHOW_LAYER_BORDERS = True

FPATH_SPIKES = None


def _format_tag_value(value: float) -> str:
    """Format a numeric value into a compact filesystem-safe tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _round_ms_tag(value_ms: float) -> int:
    """Round one millisecond value for output tags."""
    return int(round(float(value_ms)))


def _get_sta_output_dirname(trigger_pop: str, signal_type: str, time_win_ms) -> str:
    """Build the user-facing output directory name."""
    win_start, win_stop = time_win_ms
    return (
        f'{trigger_pop}_{signal_type}_'
        f'{_round_ms_tag(win_start)}_{_round_ms_tag(win_stop)}'
    )


def _get_sta_output_dir(results_root: Path, exp_label: str, trigger_pop: str, signal_type: str, time_win_ms) -> Path:
    """Return the final results directory for one STA configuration."""
    return (
        Path(results_root)
        / exp_label
        / RESULT_GROUP
        / _get_sta_output_dirname(trigger_pop, signal_type, time_win_ms)
    )


def _get_sta_cache_dir(
        dirpath_proc: Path,
        trigger_pop: str,
        signal_type: str,
        spike_t_limits_s,
        time_win_ms,
        subtract_chan_mean: bool,
        ) -> Path:
    """Return the processing-cache directory for one STA configuration."""
    t0_s, t1_s = spike_t_limits_s
    folder = _get_sta_output_dirname(trigger_pop, signal_type, time_win_ms)
    return (
        Path(dirpath_proc)
        / 'spike_triggered_cache'
        / (
            f'{folder}'
            f'__t_{_format_tag_value(t0_s)}_{_format_tag_value(t1_s)}'
            f'__meansub_{int(bool(subtract_chan_mean))}'
        )
    )


def _get_default_trigger_spike_cache_path(dirpath_proc: Path, trigger_pop: str, spike_t_limits_s) -> Path:
    """Return the default trigger-spike cache path for one population/time span."""
    t0_s, t1_s = spike_t_limits_s
    return (
        Path(dirpath_proc)
        / 'spike_triggered_cache'
        / (
            f'spikes_{trigger_pop.lower()}_combined_ms_abs'
            f'__t_{_format_tag_value(t0_s)}_{_format_tag_value(t1_s)}.npz'
        )
    )


def _get_spike_cache_candidates(dirpath_proc: Path, trigger_pop: str, spike_t_limits_s, fpath_spikes=None) -> list[Path]:
    """Return candidate spike-cache paths in priority order."""
    if fpath_spikes is not None:
        return [Path(fpath_spikes)]
    return [
        _get_default_trigger_spike_cache_path(dirpath_proc, trigger_pop, spike_t_limits_s),
        Path(dirpath_proc) / 'spike_data_demo' / 'spikes_combined_ms.npz',
    ]


def _time_win_s_from_ms(time_win_ms) -> tuple[float, float]:
    """Convert an STA window from milliseconds to seconds."""
    time_win_ms = np.asarray(time_win_ms, dtype=float)
    if time_win_ms.shape != (2,):
        raise ValueError('TIME_WIN_MS should be a length-2 sequence')
    return tuple((time_win_ms * 1e-3).tolist())


def _normalize_signal_type(signal_type: str) -> str:
    """Validate the configured signal type."""
    signal_type = str(signal_type).strip().lower()
    if signal_type not in {'lfp', 'csd'}:
        raise ValueError("SIGNAL_TYPE should be either 'lfp' or 'csd'")
    return signal_type


def _encode_dataset_attr(value):
    """Convert one xarray attr value into a NetCDF-friendly scalar."""
    if value is None:
        return 'null'
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    return json.dumps(value, sort_keys=True)


def _make_netcdf_safe_dataarray(X: xr.DataArray) -> xr.DataArray:
    """Return a copy with attrs converted to NetCDF-friendly scalar values."""
    X_out = X.copy(deep=False)
    X_out.attrs = {
        str(key): _encode_dataset_attr(value)
        for key, value in dict(X.attrs).items()
    }
    return X_out


def _get_signal_for_analysis(
        fpath_sim_result,
        dirpath_proc_root,
        signal_type: str,
        spike_t_limits_s,
        subtract_chan_mean: bool = True,
        ) -> tuple[xr.DataArray, Path]:
    """Load the cached LFP, preprocess it, and optionally derive CSD."""
    signal_type = _normalize_signal_type(signal_type)
    fpath_lfp_cache = get_lfp_cache_path(fpath_sim_result, dirpath_proc_root)
    sim_result = None
    if not fpath_lfp_cache.exists():
        sim_result = load_sim_result(fpath_sim_result)
    lfp = load_or_extract_lfp(sim_result, fpath_sim_result, dirpath_proc_root)
    lfp = interp_time_outliers(lfp).sel(time=slice(*spike_t_limits_s))
    if signal_type == 'lfp':
        signal = lfp
    else:
        signal = calc_xr_csd(lfp)
    signal = _maybe_subtract_channel_mean(signal, enabled=subtract_chan_mean)
    return signal, fpath_lfp_cache


def _maybe_subtract_channel_mean(
        signal: xr.DataArray,
        enabled: bool = True,
        time_dim: str = 'time',
        ) -> xr.DataArray:
    """Optionally subtract each channel's mean over time."""
    if not enabled:
        return signal
    if time_dim not in signal.dims:
        raise ValueError(f'Time dimension {time_dim!r} is not present in signal')

    source_attrs = dict(signal.attrs)
    signal_out = signal - signal.mean(dim=time_dim)
    signal_out.attrs = source_attrs
    signal_out.attrs['channel_mean_subtracted'] = True
    signal_out.attrs['channel_mean_subtract_time_dim'] = str(time_dim)
    return signal_out


def _is_usable_trigger_spikes(spikes: SpikeData, trigger_pop: str) -> bool:
    """Check whether a loaded SpikeData cache matches the required trigger setup."""
    if trigger_pop not in spikes.get_pop_names():
        return False
    if bool(spikes.metadata['subtract_t0']):
        return False
    return True


def _load_or_extract_trigger_spikes(
        fpath_sim_result,
        dirpath_proc,
        trigger_pop: str,
        spike_t_limits_s,
        fpath_spikes=None,
        ) -> tuple[SpikeData, Path, bool]:
    """Load a trigger-spike cache or extract one when no usable cache exists."""
    candidates = _get_spike_cache_candidates(
        dirpath_proc,
        trigger_pop,
        spike_t_limits_s,
        fpath_spikes=fpath_spikes,
    )
    explicit_path = None if fpath_spikes is None else Path(fpath_spikes)
    for candidate in candidates:
        if not candidate.exists():
            continue
        spikes = SpikeData.load(candidate)
        if not spikes.combine_mode:
            spikes = spikes.combine()
        if _is_usable_trigger_spikes(spikes, trigger_pop):
            return spikes, candidate, True
        if explicit_path is not None and candidate == explicit_path:
            raise ValueError(
                f'Spike cache {candidate} is not usable for {trigger_pop}: '
                'it should contain the population and preserve subtract_t0=False'
            )

    fpath_out = explicit_path
    if fpath_out is None:
        fpath_out = _get_default_trigger_spike_cache_path(dirpath_proc, trigger_pop, spike_t_limits_s)
    sim_result = load_sim_result(fpath_sim_result)
    spikes = SpikeData.from_sim_result(
        sim_result,
        pop_names=[trigger_pop],
        combine=True,
        t0=float(spike_t_limits_s[0]),
        tmax=float(spike_t_limits_s[1]),
        subtract_t0=False,
        ms=True,
        ndigits=3,
    )
    spikes.save(fpath_out)
    return spikes, fpath_out, False


def _build_sta_avg_2d(signal: xr.DataArray, spikes: SpikeData, trigger_pop: str, time_win_s) -> xr.DataArray:
    """Compute a depth x relative-time STA average using one channel at a time."""
    if 'y' not in signal.dims or 'time' not in signal.dims:
        raise ValueError('signal should contain y and time dimensions')

    signal = signal.transpose('y', 'time')
    y_values = np.asarray(signal.coords['y'].values, dtype=float)
    if y_values.size == 0:
        raise ValueError('signal should contain at least one depth channel')

    avg_values = None
    time_rel = None
    n_spikes_used = []
    for idx, _resolved_y in enumerate(y_values):
        signal_chan = signal.isel(y=idx)
        sta = calc_xr_sta(
            signal_chan,
            spikes,
            time_win_s,
            pop_name=trigger_pop,
            time_units='s',
            return_mode='both',
        )
        avg_chan = np.asarray(sta['avg'].values, dtype=float)
        if avg_values is None:
            time_rel = np.asarray(sta['avg'].coords['time_rel'].values, dtype=float)
            avg_values = np.full((len(y_values), len(time_rel)), np.nan, dtype=float)
        avg_values[idx, :] = avg_chan
        n_spikes_used.append(int(sta['epochs'].sizes['spike']))
        del signal_chan, sta, avg_chan
        gc.collect()

    attrs = dict(signal.attrs)
    attrs.update({
        'analysis': 'spike_triggered_sta',
        'trigger_pop': str(trigger_pop),
        'signal_type': str(signal.name or ''),
        'time_win_s': [float(time_win_s[0]), float(time_win_s[1])],
        'time_win_ms': [float(time_win_s[0] * 1e3), float(time_win_s[1] * 1e3)],
        'n_spikes_used_by_channel': [int(value) for value in n_spikes_used],
        'source_dims': list(signal.dims),
    })
    return xr.DataArray(
        avg_values,
        dims=['y', 'time_rel'],
        coords={'y': y_values, 'time_rel': time_rel},
        attrs=attrs,
        name='sta_avg',
    )


def _resolve_plot_depths(y_values, selected_y=None) -> list[float]:
    """Resolve requested plot depths to available channels."""
    y_values = np.asarray(y_values, dtype=float)
    if y_values.ndim != 1 or y_values.size == 0:
        raise ValueError('y_values should be a non-empty 1D coordinate')
    if selected_y is None:
        return [float(value) for value in y_values.tolist()]

    resolved = []
    seen = set()
    for value in np.asarray(selected_y, dtype=float).tolist():
        idx = int(np.argmin(np.abs(y_values - float(value))))
        resolved_y = float(y_values[idx])
        if resolved_y in seen:
            continue
        seen.add(resolved_y)
        resolved.append(resolved_y)
    return resolved


def _build_run_manifest(
        trigger_pop: str,
        signal_type: str,
        spike_t_limits_s,
        time_win_ms,
        subtract_chan_mean: bool,
        avg_sta: xr.DataArray,
        fpath_sim_result,
        fpath_lfp_cache,
        fpath_spikes,
        ) -> dict:
    """Build a compact JSON manifest for one STA cache."""
    return {
        'analysis': 'spike_triggered_sta',
        'trigger_pop': str(trigger_pop),
        'signal_type': str(signal_type),
        'spike_t_limits_s': [float(spike_t_limits_s[0]), float(spike_t_limits_s[1])],
        'time_win_ms': [float(time_win_ms[0]), float(time_win_ms[1])],
        'subtract_chan_mean': bool(subtract_chan_mean),
        'source_sim_result': str(Path(fpath_sim_result).resolve()),
        'source_lfp_cache': str(Path(fpath_lfp_cache).resolve()),
        'source_spike_cache': str(Path(fpath_spikes).resolve()),
        'source_layer_config': str(Path(FPATH_LAYER_CONFIG).resolve()),
        'resolved_y': [float(value) for value in np.asarray(avg_sta.coords['y'].values, dtype=float).tolist()],
        'time_rel_ms': [
            float(value)
            for value in (np.asarray(avg_sta.coords['time_rel'].values, dtype=float) * 1e3).tolist()
        ],
        'n_spikes_used_by_channel': _decode_int_list_attr(avg_sta.attrs.get('n_spikes_used_by_channel', [])),
    }


def _decode_int_list_attr(value) -> list[int]:
    """Decode a list-like attr that may round-trip through NetCDF as JSON text."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return []


def _load_layer_config(fpath_layer_config) -> dict:
    """Load a lightweight JSON layer config without touching the raw sim pickle."""
    fpath_layer_config = Path(fpath_layer_config)
    with fpath_layer_config.open('r', encoding='utf-8') as fobj:
        payload = json.load(fobj)
    if 'layers' not in payload or 'y_size_um' not in payload:
        raise ValueError('Layer config should define y_size_um and layers')
    return payload


def _get_layer_spans_um(layer_config: dict) -> list[dict]:
    """Convert normalized layer spans to absolute depth spans in micrometers."""
    y_size_um = float(layer_config['y_size_um'])
    spans = []
    for item in layer_config['layers']:
        y0_norm, y1_norm = item['y_norm']
        spans.append({
            'name': str(item['name']),
            'y0_um': float(y0_norm) * y_size_um,
            'y1_um': float(y1_norm) * y_size_um,
        })
    return spans


def _get_visible_layer_spans(layer_spans, y_values) -> list[dict]:
    """Return layer spans that overlap the plotted depth range."""
    y_values = np.asarray(y_values, dtype=float)
    if y_values.ndim != 1 or y_values.size == 0:
        return []
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    visible = []
    for span in layer_spans:
        if float(span['y1_um']) < y_min or float(span['y0_um']) > y_max:
            continue
        visible.append(dict(span))
    return visible


def _add_layer_borders(ax, layer_spans, y_values, x_min_ms, x_max_ms) -> None:
    """Overlay horizontal layer borders and left-side layer labels."""
    visible_spans = _get_visible_layer_spans(layer_spans, y_values)
    if not visible_spans:
        return

    border_values = set()
    for span in visible_spans:
        border_values.add(float(span['y0_um']))
        border_values.add(float(span['y1_um']))
    for y_border in sorted(border_values):
        ax.axhline(y_border, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    x_label = x_min_ms + 0.04 * (x_max_ms - x_min_ms)
    for span in visible_spans:
        y_mid = 0.5 * (float(span['y0_um']) + float(span['y1_um']))
        ax.text(
            x_label,
            y_mid,
            str(span['name']),
            color='k',
            fontsize=9,
            ha='left',
            va='center',
            bbox=dict(facecolor='white', alpha=0.35, edgecolor='none', pad=1.0),
        )


def _save_sta_avg_cache(avg_sta: xr.DataArray, cache_dir: Path, result_dir: Path, manifest: dict) -> tuple[Path, Path, Path]:
    """Save the average cache and manifest to their cache/results locations."""
    cache_dir = Path(cache_dir)
    result_dir = Path(result_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cache_nc = cache_dir / 'avg_2d.nc'
    result_nc = result_dir / 'avg_2d.nc'
    manifest_path = cache_dir / 'manifest.json'
    tmp_nc = cache_dir / 'avg_2d.tmp.nc'

    avg_sta_safe = _make_netcdf_safe_dataarray(avg_sta)
    if tmp_nc.exists():
        tmp_nc.unlink()
    try:
        save_xr(avg_sta_safe, tmp_nc)
        os.replace(tmp_nc, cache_nc)
    finally:
        if tmp_nc.exists():
            tmp_nc.unlink()
    shutil.copyfile(cache_nc, result_nc)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return cache_nc, result_nc, manifest_path


def _is_valid_sta_cache(avg_sta: xr.DataArray) -> bool:
    """Return whether a loaded STA cache looks usable for plotting/reuse."""
    if avg_sta.dims != ('y', 'time_rel'):
        return False
    if avg_sta.sizes.get('y', 0) == 0 or avg_sta.sizes.get('time_rel', 0) == 0:
        return False
    values = np.asarray(avg_sta.values, dtype=float)
    return bool(np.isfinite(values).any())


def _load_or_compute_sta_avg_cache(
        signal: xr.DataArray,
        spikes: SpikeData,
        trigger_pop: str,
        signal_type: str,
        spike_t_limits_s,
        time_win_ms,
        subtract_chan_mean: bool,
        cache_dir: Path,
        result_dir: Path,
        fpath_sim_result,
        fpath_lfp_cache,
        fpath_spikes,
        ) -> tuple[xr.DataArray, Path, Path, Path, bool]:
    """Load the cached all-channel STA or compute and save it."""
    cache_dir = Path(cache_dir)
    result_dir = Path(result_dir)
    cache_nc = cache_dir / 'avg_2d.nc'
    manifest_path = cache_dir / 'manifest.json'
    result_nc = result_dir / 'avg_2d.nc'

    if cache_nc.exists():
        try:
            avg_sta = load_xr(cache_nc, load=True)
        except Exception:
            avg_sta = None
        if avg_sta is not None and _is_valid_sta_cache(avg_sta):
            manifest = _build_run_manifest(
                trigger_pop,
                signal_type,
                spike_t_limits_s,
                time_win_ms,
                subtract_chan_mean,
                avg_sta,
                fpath_sim_result,
                fpath_lfp_cache,
                fpath_spikes,
            )
            if not manifest_path.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8',
                )
            if not result_nc.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_nc, result_nc)
            return avg_sta, cache_nc, result_nc, manifest_path, True
        print(f'Ignoring invalid STA cache and recomputing: {cache_nc}')

    avg_sta = _build_sta_avg_2d(signal, spikes, trigger_pop, _time_win_s_from_ms(time_win_ms))
    avg_sta.attrs.update({
        'trigger_pop': str(trigger_pop),
        'signal_type': str(signal_type),
        'spike_t_limits_s': [float(spike_t_limits_s[0]), float(spike_t_limits_s[1])],
        'time_win_ms': [float(time_win_ms[0]), float(time_win_ms[1])],
        'subtract_chan_mean': bool(subtract_chan_mean),
        'source_sim_result': str(Path(fpath_sim_result).resolve()),
        'source_lfp_cache': str(Path(fpath_lfp_cache).resolve()),
        'source_spike_cache': str(Path(fpath_spikes).resolve()),
        'source_layer_config': str(Path(FPATH_LAYER_CONFIG).resolve()),
        'resolved_y': [float(value) for value in np.asarray(avg_sta.coords['y'].values, dtype=float).tolist()],
    })
    manifest = _build_run_manifest(
        trigger_pop,
        signal_type,
        spike_t_limits_s,
        time_win_ms,
        subtract_chan_mean,
        avg_sta,
        fpath_sim_result,
        fpath_lfp_cache,
        fpath_spikes,
    )
    cache_nc, result_nc, manifest_path = _save_sta_avg_cache(avg_sta, cache_dir, result_dir, manifest)
    return avg_sta, cache_nc, result_nc, manifest_path, False


def _plot_sta_2d(
        avg_sta: xr.DataArray,
        fpath_out: Path,
        trigger_pop: str,
        signal_type: str,
        show_zero_line: bool = False,
        layer_spans=None,
        ) -> None:
    """Render the all-channel depth x time STA image."""
    time_rel_ms = np.asarray(avg_sta.coords['time_rel'].values, dtype=float) * 1e3
    y_values = np.asarray(avg_sta.coords['y'].values, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        np.asarray(avg_sta.values, dtype=float),
        aspect='auto',
        origin='upper',
        extent=[time_rel_ms[0], time_rel_ms[-1], y_values[-1], y_values[0]],
        cmap='coolwarm' if signal_type == 'csd' else 'viridis',
    )
    fig.colorbar(image, ax=ax, label=signal_type.upper())
    if show_zero_line:
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1)
    if layer_spans is not None:
        _add_layer_borders(ax, layer_spans, y_values, time_rel_ms[0], time_rel_ms[-1])
    ax.set_xlabel('Time relative to spike (ms)')
    ax.set_ylabel('Depth (um)')
    ax.set_title(f'{signal_type.upper()} STA, trigger pop {trigger_pop}')
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _plot_sta_1d(
        avg_sta: xr.DataArray,
        fpath_out: Path,
        resolved_y: float,
        trigger_pop: str,
        signal_type: str,
        show_zero_line: bool = False,
        ) -> None:
    """Render one single-channel STA trace from the cached all-channel average."""
    row = avg_sta.sel(y=resolved_y)
    time_rel_ms = np.asarray(row.coords['time_rel'].values, dtype=float) * 1e3
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(time_rel_ms, np.asarray(row.values, dtype=float), color='k', linewidth=2)
    if show_zero_line:
        ax.axvline(0.0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Time relative to spike (ms)')
    ax.set_ylabel(signal_type.upper())
    ax.set_title(f'{signal_type.upper()} STA @ y={resolved_y:g}, trigger pop {trigger_pop}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the spike-triggered LFP/CSD analysis."""
    exp_label = get_exp_label(FPATH_SIM_RESULT)
    dirpath_proc = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    signal_type = _normalize_signal_type(SIGNAL_TYPE)
    layer_spans = None
    dirpath_out = _get_sta_output_dir(
        DIRPATH_RESULTS_ROOT,
        exp_label,
        TRIGGER_POP,
        signal_type,
        TIME_WIN_MS,
    )
    dirpath_cache = _get_sta_cache_dir(
        dirpath_proc,
        TRIGGER_POP,
        signal_type,
        SPIKE_T_LIMITS_S,
        TIME_WIN_MS,
        SUBTRACT_CHAN_MEAN,
    )

    signal, fpath_lfp_cache = _get_signal_for_analysis(
        FPATH_SIM_RESULT,
        DIRPATH_PROC_ROOT,
        signal_type,
        SPIKE_T_LIMITS_S,
        subtract_chan_mean=SUBTRACT_CHAN_MEAN,
    )
    signal.name = signal_type

    spikes, fpath_spikes, spike_cache_hit = _load_or_extract_trigger_spikes(
        FPATH_SIM_RESULT,
        dirpath_proc,
        TRIGGER_POP,
        SPIKE_T_LIMITS_S,
        fpath_spikes=FPATH_SPIKES,
    )

    if SHOW_LAYER_BORDERS:
        layer_spans = _get_layer_spans_um(_load_layer_config(FPATH_LAYER_CONFIG))

    avg_sta, cache_nc, result_nc, manifest_path, sta_cache_hit = _load_or_compute_sta_avg_cache(
        signal,
        spikes,
        TRIGGER_POP,
        signal_type,
        SPIKE_T_LIMITS_S,
        TIME_WIN_MS,
        SUBTRACT_CHAN_MEAN,
        dirpath_cache,
        dirpath_out,
        FPATH_SIM_RESULT,
        fpath_lfp_cache,
        fpath_spikes,
    )

    if MAKE_PLOT_2D:
        dirpath_out.mkdir(parents=True, exist_ok=True)
        _plot_sta_2d(
            avg_sta,
            dirpath_out / 'sta_2d.png',
            TRIGGER_POP,
            signal_type,
            show_zero_line=SHOW_ZERO_LINE,
            layer_spans=layer_spans,
        )

    if MAKE_PLOT_1D:
        dirpath_single = dirpath_out / 'single_chans'
        dirpath_single.mkdir(parents=True, exist_ok=True)
        for resolved_y in _resolve_plot_depths(avg_sta.coords['y'].values, selected_y=PLOT_Y):
            _plot_sta_1d(
                avg_sta,
                dirpath_single / f'sta_y_{resolved_y:g}.png',
                resolved_y,
                TRIGGER_POP,
                signal_type,
                show_zero_line=SHOW_ZERO_LINE,
            )

    print(f'Output dir: {dirpath_out}')
    print(f'STA cache file: {cache_nc}')
    print(f'Results cache copy: {result_nc}')
    print(f'Manifest: {manifest_path}')
    print(f'Spike cache: {fpath_spikes} ({"hit" if spike_cache_hit else "miss"})')
    print(f'STA cache: {"hit" if sta_cache_hit else "miss"} at {cache_nc}')


if __name__ == '__main__':
    main()
