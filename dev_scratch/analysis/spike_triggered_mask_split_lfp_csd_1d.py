"""Masked 1D spike-triggered LFP/CSD analysis with cached channel-wise STA splits."""

from __future__ import annotations

import gc
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
from sim_data_analyzer.netpyne_res_parse_utils import get_pop_names
from sim_data_analyzer.spike_data import SpikeData, _SpikeMeta
from sim_data_analyzer.xr_diff import calc_xr_csd
from sim_data_analyzer.xr_io import load_xr, save_xr
from sim_data_analyzer.xr_signal import interp_time_outliers
from sim_data_analyzer.xr_spike_triggered import calc_xr_sta


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
FPATH_MASK = (
    DIR_PACKAGE
    / 'dev_scratch'
    / 'data_proc'
    / 'a1_lfp_30s_0'
    / 'oevent_mask'
    / 'exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default.nc'
)

RESULT_GROUP = 'sta_mask_split'
TRIGGER_POPS = ['ITS4']
POP_GROUP_NAME = 'example_group'
SIGNAL_TYPE = 'csd'
SPIKE_T_LIMITS_S = (5.0, 30.0)
TIME_WIN_MS = (-100.0, 100.0)
CSD_SUBTRACTION_MODE = 'shared'
PLOT_Y = None
SHOW_ZERO_LINE = False
ZERO_LINE_ALPHA = 0.3
FPATH_SPIKES = None

MASK_STATES = ('mask1', 'mask0')
MASK_STATE_LABELS = {
    'mask1': 'mask=1',
    'mask0': 'mask=0',
}
MASK_STATE_COLORS = {
    'mask1': '#d62728',
    'mask0': '#1f77b4',
}


def _format_tag_value(value: float) -> str:
    """Format a numeric value into a compact filesystem-safe tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _round_ms_tag(value_ms: float) -> int:
    """Round one millisecond value for output tags."""
    return int(round(float(value_ms)))


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


def _normalize_csd_subtraction_mode(mode: str) -> str:
    """Validate the configured CSD subtraction mode."""
    mode = str(mode).strip().lower()
    if mode not in {'shared', 'separate'}:
        raise ValueError("CSD_SUBTRACTION_MODE should be either 'shared' or 'separate'")
    return mode


def _resolve_all_trigger_pops(all_pop_names) -> list[str]:
    """Return all non-frozen population names in stable order."""
    pop_names = [str(pop_name) for pop_name in list(all_pop_names) if 'frz' not in str(pop_name)]
    if not pop_names:
        raise ValueError('No non-frozen populations are available for analysis')
    return pop_names


def _needs_all_trigger_pops(trigger_pops) -> bool:
    """Return whether the configured trigger-pop spec requests all non-frozen populations."""
    if isinstance(trigger_pops, str):
        return trigger_pops.strip().lower() == 'all'
    trigger_pops = list(trigger_pops)
    return len(trigger_pops) == 1 and str(trigger_pops[0]).strip().lower() == 'all'


def _resolve_trigger_pops(trigger_pops, pop_group_name: str | None = None, all_pop_names=None) -> list[str]:
    """Validate and normalize the configured trigger-pop selection."""
    if _needs_all_trigger_pops(trigger_pops):
        if all_pop_names is None:
            raise ValueError('all_pop_names is required when TRIGGER_POPS requests all populations')
        pop_names = _resolve_all_trigger_pops(all_pop_names)
    else:
        pop_names = [str(pop_name) for pop_name in list(trigger_pops)]
    if not pop_names:
        raise ValueError('TRIGGER_POPS should contain at least one population name')
    if any(not pop_name.strip() for pop_name in pop_names):
        raise ValueError('TRIGGER_POPS should not contain empty population names')
    if len(set(pop_names)) != len(pop_names):
        raise ValueError(f'TRIGGER_POPS should not contain duplicates: {pop_names}')
    if len(pop_names) > 1 and not str(pop_group_name or '').strip():
        raise ValueError('POP_GROUP_NAME is required when TRIGGER_POPS contains multiple populations')
    return pop_names


def _try_load_pop_names_from_spike_cache(fpath_spikes) -> list[str] | None:
    """Return population names from one spike cache when it is available and readable."""
    if fpath_spikes is None:
        return None
    fpath_spikes = Path(fpath_spikes)
    if not fpath_spikes.exists():
        return None
    try:
        spikes = SpikeData.load(fpath_spikes)
    except Exception:
        return None
    return [str(pop_name) for pop_name in spikes.get_pop_names()]


def _load_available_trigger_pop_names(fpath_sim_result, dirpath_proc, fpath_spikes=None) -> list[str]:
    """Load available trigger populations, preferring spike caches over the raw sim pickle."""
    if isinstance(fpath_spikes, dict):
        cache_pop_names = [str(pop_name) for pop_name in fpath_spikes]
        if cache_pop_names:
            return _resolve_all_trigger_pops(cache_pop_names)
    else:
        cache_pop_names = _try_load_pop_names_from_spike_cache(fpath_spikes)
        if cache_pop_names:
            return _resolve_all_trigger_pops(cache_pop_names)

    combined_spike_cache = Path(dirpath_proc) / 'spike_data_demo' / 'spikes_combined_ms.npz'
    cache_pop_names = _try_load_pop_names_from_spike_cache(combined_spike_cache)
    if cache_pop_names:
        return _resolve_all_trigger_pops(cache_pop_names)

    sim_result = load_sim_result(fpath_sim_result)
    return _resolve_all_trigger_pops(get_pop_names(sim_result))


def _get_mask_tag(fpath_mask) -> str:
    """Return a compact tag derived from the mask filename."""
    return Path(fpath_mask).stem


def _get_result_tag(trigger_pops, signal_type: str, time_win_ms, fpath_mask, csd_subtraction_mode: str, pop_group_name: str | None = None) -> str:
    """Build the user-facing result tag for one masked STA configuration."""
    pop_names = _resolve_trigger_pops(trigger_pops, pop_group_name=pop_group_name)
    pop_tag = pop_names[0] if len(pop_names) == 1 else str(pop_group_name).strip()
    return (
        f'{pop_tag}_{signal_type}_{_round_ms_tag(time_win_ms[0])}_{_round_ms_tag(time_win_ms[1])}'
        f'__mask_{_get_mask_tag(fpath_mask)}'
        f'__csdsub_{_normalize_csd_subtraction_mode(csd_subtraction_mode)}'
    )


def _get_output_dir(results_root: Path, exp_label: str, trigger_pops, signal_type: str, time_win_ms, fpath_mask, csd_subtraction_mode: str, pop_group_name: str | None = None) -> Path:
    """Return the final results directory for one masked STA configuration."""
    return (
        Path(results_root)
        / exp_label
        / RESULT_GROUP
        / _get_result_tag(
            trigger_pops,
            signal_type,
            time_win_ms,
            fpath_mask,
            csd_subtraction_mode,
            pop_group_name=pop_group_name,
        )
        / '1d'
    )


def _get_cache_dir(
        dirpath_proc: Path,
        trigger_pop: str,
        signal_type: str,
        spike_t_limits_s,
        time_win_ms,
        fpath_mask,
        csd_subtraction_mode: str,
        ) -> Path:
    """Return the processing-cache directory for one masked STA configuration."""
    t0_s, t1_s = spike_t_limits_s
    return (
        Path(dirpath_proc)
        / 'spike_triggered_mask_split_cache'
        / (
            f'{trigger_pop}_{signal_type}_{_round_ms_tag(time_win_ms[0])}_{_round_ms_tag(time_win_ms[1])}'
            f'__mask_{_get_mask_tag(fpath_mask)}'
            f'__t_{_format_tag_value(t0_s)}_{_format_tag_value(t1_s)}'
            f'__csdsub_{_normalize_csd_subtraction_mode(csd_subtraction_mode)}'
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


def _resolve_configured_spike_cache_path(fpath_spikes, trigger_pop: str, trigger_pops, pop_group_name: str | None = None) -> Path | None:
    """Resolve an optional configured spike cache path for one population."""
    if fpath_spikes is None:
        return None
    if isinstance(fpath_spikes, dict):
        resolved = fpath_spikes.get(trigger_pop)
        return None if resolved is None else Path(resolved)
    trigger_pops = _resolve_trigger_pops(trigger_pops, pop_group_name=pop_group_name)
    if len(trigger_pops) > 1:
        raise ValueError(
            'FPATH_SPIKES should be None or a {pop_name: path} mapping when TRIGGER_POPS has multiple populations'
        )
    return Path(fpath_spikes)


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


def _maybe_subtract_channel_mean(signal: xr.DataArray, enabled: bool = True, time_dim: str = 'time') -> xr.DataArray:
    """Optionally subtract each channel's mean over time."""
    if not enabled:
        return signal
    signal_out = signal - signal.mean(dim=time_dim)
    signal_out.attrs = dict(signal.attrs)
    signal_out.attrs['channel_mean_subtracted'] = True
    return signal_out


def _get_signal_for_analysis(
        fpath_sim_result,
        dirpath_proc_root,
        signal_type: str,
        spike_t_limits_s,
        csd_subtraction_mode: str,
        ) -> tuple[xr.DataArray, Path]:
    """Load cached LFP, preprocess it, and optionally derive CSD."""
    signal_type = _normalize_signal_type(signal_type)
    csd_subtraction_mode = _normalize_csd_subtraction_mode(csd_subtraction_mode)
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
        signal = _maybe_subtract_channel_mean(signal, enabled=(csd_subtraction_mode == 'shared'))
    signal.name = signal_type
    return signal, fpath_lfp_cache


def _load_mask(fpath_mask) -> xr.DataArray:
    """Load and validate one saved binary event mask."""
    mask = load_xr(fpath_mask, data_type='dataarray', load=True)
    if mask.dims != ('y', 'time'):
        raise ValueError(f'Expected mask dims ("y", "time"), got {mask.dims}')
    return mask


def _resolve_plot_depths(y_values, selected_y=None) -> list[float]:
    """Resolve requested plot depths to available channels."""
    y_values = np.asarray(y_values, dtype=float)
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


def _get_spike_time_scale_s(spikes: SpikeData) -> float:
    """Return the scale factor from stored spike units to seconds."""
    return 1e-3 if bool(spikes.metadata['ms']) else 1.0


def _build_subset_spike_data(spikes: SpikeData, trigger_pop: str, raw_times_subset: np.ndarray) -> SpikeData:
    """Build one temporary combined SpikeData for a filtered trigger subset."""
    meta = _SpikeMeta(
        combine=True,
        t0=float(spikes.metadata['t0']),
        tmax=float(spikes.metadata['tmax']),
        subtract_t0=bool(spikes.metadata['subtract_t0']),
        ms=bool(spikes.metadata['ms']),
        ndigits=int(spikes.metadata['ndigits']),
    )
    return SpikeData(
        {str(trigger_pop): [np.asarray(raw_times_subset, dtype=float)]},
        meta=meta,
        pop_sizes={str(trigger_pop): int(spikes.get_pop_size(trigger_pop))},
    )


def _split_raw_spikes_by_mask(mask_chan: xr.DataArray, raw_spike_times, spike_time_scale_s: float) -> dict[str, np.ndarray]:
    """Split one combined spike train by mask value at the same channel/time."""
    raw_spike_times = np.asarray(raw_spike_times, dtype=float)
    if raw_spike_times.size == 0:
        return {state: np.array([], dtype=float) for state in MASK_STATES}

    spike_times_s = raw_spike_times * float(spike_time_scale_s)
    tmin = float(mask_chan.coords['time'].values[0])
    tmax = float(mask_chan.coords['time'].values[-1])
    in_range = (spike_times_s >= tmin) & (spike_times_s <= tmax)
    if not np.any(in_range):
        return {state: np.array([], dtype=float) for state in MASK_STATES}

    raw_times_valid = raw_spike_times[in_range]
    spike_times_valid_s = spike_times_s[in_range]
    mask_values = np.asarray(
        mask_chan.sel(time=spike_times_valid_s, method='nearest').values,
        dtype=float,
    )
    is_mask1 = mask_values >= 0.5
    return {
        'mask1': np.asarray(raw_times_valid[is_mask1], dtype=float),
        'mask0': np.asarray(raw_times_valid[~is_mask1], dtype=float),
    }


def _build_mask_split_sta_dataset(
        signal: xr.DataArray,
        mask: xr.DataArray,
        spikes: SpikeData,
        trigger_pop: str,
        time_win_s,
        signal_type: str,
        csd_subtraction_mode: str,
        ) -> xr.Dataset:
    """Compute channel-wise STA averages split by same-channel mask value."""
    signal = signal.transpose('y', 'time')
    mask = mask.transpose('y', 'time')
    y_values = np.asarray(signal.coords['y'].values, dtype=float)
    if not np.array_equal(y_values, np.asarray(mask.coords['y'].values, dtype=float)):
        raise ValueError('Signal and mask y coordinates should match exactly')

    raw_spike_times = np.asarray(spikes.get_pop_spikes(trigger_pop)[0], dtype=float)
    spike_time_scale_s = _get_spike_time_scale_s(spikes)
    time_rel = None
    avg_values = None
    n_spikes = np.zeros((len(MASK_STATES), len(y_values)), dtype=np.int64)

    for y_idx, y_value in enumerate(y_values):
        signal_chan = signal.isel(y=y_idx)
        mask_chan = mask.sel(y=float(y_value))
        split_times = _split_raw_spikes_by_mask(mask_chan, raw_spike_times, spike_time_scale_s)

        for state_idx, state in enumerate(MASK_STATES):
            subset_spikes = _build_subset_spike_data(spikes, trigger_pop, split_times[state])
            sta = calc_xr_sta(
                signal_chan,
                subset_spikes,
                time_win_s,
                pop_name=trigger_pop,
                time_units='s',
                return_mode='both',
            )
            avg_chan = np.asarray(sta['avg'].values, dtype=float)
            if (signal_type == 'csd') and (_normalize_csd_subtraction_mode(csd_subtraction_mode) == 'separate'):
                avg_chan = avg_chan - np.nanmean(avg_chan)
            if avg_values is None:
                time_rel = np.asarray(sta['avg'].coords['time_rel'].values, dtype=float)
                avg_values = np.full((len(MASK_STATES), len(y_values), len(time_rel)), np.nan, dtype=float)
            avg_values[state_idx, y_idx, :] = avg_chan
            n_spikes[state_idx, y_idx] = int(sta['epochs'].sizes['spike'])
            del subset_spikes, sta, avg_chan
            gc.collect()

        del signal_chan, mask_chan
        gc.collect()

    attrs = {
        'analysis': 'spike_triggered_mask_split_1d',
        'trigger_pop': str(trigger_pop),
        'signal_type': str(signal_type),
        'time_win_ms': [float(time_win_s[0] * 1e3), float(time_win_s[1] * 1e3)],
        'csd_subtraction_mode': str(csd_subtraction_mode),
    }
    return xr.Dataset(
        data_vars={
            'sta_avg': (
                ('state', 'y', 'time_rel'),
                avg_values,
            ),
            'n_spikes': (
                ('state', 'y'),
                n_spikes,
            ),
        },
        coords={
            'state': list(MASK_STATES),
            'y': y_values,
            'time_rel': time_rel,
        },
        attrs=attrs,
    )


def _is_valid_sta_cache(ds: xr.Dataset) -> bool:
    """Return whether a loaded masked STA cache looks usable."""
    if 'sta_avg' not in ds or 'n_spikes' not in ds:
        return False
    if ds['sta_avg'].dims != ('state', 'y', 'time_rel'):
        return False
    values = np.asarray(ds['sta_avg'].values, dtype=float)
    return bool(np.isfinite(values).any())


def _load_or_compute_sta_cache(
        signal: xr.DataArray,
        mask: xr.DataArray,
        spikes: SpikeData,
        trigger_pop: str,
        signal_type: str,
        time_win_ms,
        csd_subtraction_mode: str,
        cache_dir: Path,
        ) -> tuple[xr.Dataset, Path, bool]:
    """Load or compute one cached mask-split STA dataset."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / 'sta_mask_split.nc'
    if cache_path.exists():
        ds = load_xr(cache_path, data_type='dataset', load=True)
        if _is_valid_sta_cache(ds):
            return ds, cache_path, True
    ds = _build_mask_split_sta_dataset(
        signal,
        mask,
        spikes,
        trigger_pop,
        _time_win_s_from_ms(time_win_ms),
        signal_type,
        csd_subtraction_mode,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_xr(ds, cache_path)
    return ds, cache_path, False


def _plot_mask_split_sta_1d(
        sta_ds: xr.Dataset,
        resolved_y: float,
        fpath_out: Path,
        trigger_pop: str,
        signal_type: str,
        show_zero_line: bool = False,
        zero_line_alpha: float = 0.3,
        ) -> None:
    """Render one channel overlay plot for mask=1 versus mask=0 STAs."""
    time_rel_ms = np.asarray(sta_ds.coords['time_rel'].values, dtype=float) * 1e3
    fig, ax = plt.subplots(figsize=(10, 4))
    for state in MASK_STATES:
        row = sta_ds['sta_avg'].sel(state=state, y=resolved_y)
        n_spikes = int(sta_ds['n_spikes'].sel(state=state, y=resolved_y).item())
        ax.plot(
            time_rel_ms,
            np.asarray(row.values, dtype=float),
            color=MASK_STATE_COLORS[state],
            linewidth=2,
            label=f'{MASK_STATE_LABELS[state]} (n={n_spikes})',
        )
    if show_zero_line:
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1, alpha=float(zero_line_alpha))
    ax.set_xlabel('Time relative to spike (ms)')
    ax.set_ylabel(signal_type.upper())
    ax.set_title(f'{signal_type.upper()} STA split by mask @ y={resolved_y:g}, trigger pop {trigger_pop}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run masked 1D spike-triggered LFP/CSD analysis."""
    exp_label = get_exp_label(FPATH_SIM_RESULT)
    dirpath_proc = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
    signal_type = _normalize_signal_type(SIGNAL_TYPE)
    csd_subtraction_mode = _normalize_csd_subtraction_mode(CSD_SUBTRACTION_MODE)
    all_pop_names = None
    if _needs_all_trigger_pops(TRIGGER_POPS):
        all_pop_names = _load_available_trigger_pop_names(
            FPATH_SIM_RESULT,
            dirpath_proc,
            fpath_spikes=FPATH_SPIKES,
        )
    trigger_pops = _resolve_trigger_pops(
        TRIGGER_POPS,
        pop_group_name=POP_GROUP_NAME,
        all_pop_names=all_pop_names,
    )
    dirpath_out = _get_output_dir(
        DIRPATH_RESULTS_ROOT,
        exp_label,
        trigger_pops,
        signal_type,
        TIME_WIN_MS,
        FPATH_MASK,
        csd_subtraction_mode,
        pop_group_name=POP_GROUP_NAME,
    )

    signal, fpath_lfp_cache = _get_signal_for_analysis(
        FPATH_SIM_RESULT,
        DIRPATH_PROC_ROOT,
        signal_type,
        SPIKE_T_LIMITS_S,
        csd_subtraction_mode,
    )
    mask = _load_mask(FPATH_MASK).sel(time=slice(*SPIKE_T_LIMITS_S))

    print(f'Output dir: {dirpath_out}')
    print(f'LFP cache: {fpath_lfp_cache}')
    print(f'Mask: {FPATH_MASK}')

    for trigger_pop in trigger_pops:
        spikes, fpath_spikes, spike_cache_hit = _load_or_extract_trigger_spikes(
            FPATH_SIM_RESULT,
            dirpath_proc,
            trigger_pop,
            SPIKE_T_LIMITS_S,
            fpath_spikes=_resolve_configured_spike_cache_path(
                FPATH_SPIKES,
                trigger_pop,
                trigger_pops,
                pop_group_name=POP_GROUP_NAME,
            ),
        )
        cache_dir = _get_cache_dir(
            dirpath_proc,
            trigger_pop,
            signal_type,
            SPIKE_T_LIMITS_S,
            TIME_WIN_MS,
            FPATH_MASK,
            csd_subtraction_mode,
        )
        sta_ds, cache_path, cache_hit = _load_or_compute_sta_cache(
            signal,
            mask,
            spikes,
            trigger_pop,
            signal_type,
            TIME_WIN_MS,
            csd_subtraction_mode,
            cache_dir,
        )

        dirpath_pop = dirpath_out / str(trigger_pop)
        dirpath_pop.mkdir(parents=True, exist_ok=True)
        for resolved_y in _resolve_plot_depths(sta_ds.coords['y'].values, selected_y=PLOT_Y):
            _plot_mask_split_sta_1d(
                sta_ds,
                resolved_y,
                dirpath_pop / f'sta_masksplit_y_{resolved_y:g}.png',
                trigger_pop,
                signal_type,
                show_zero_line=SHOW_ZERO_LINE,
                zero_line_alpha=ZERO_LINE_ALPHA,
            )

        print(f'[{trigger_pop}] Spike cache: {fpath_spikes} ({"hit" if spike_cache_hit else "miss"})')
        print(f'[{trigger_pop}] STA cache: {"hit" if cache_hit else "miss"} at {cache_path}')
        print(f'[{trigger_pop}] Plot dir: {dirpath_pop}')


if __name__ == '__main__':
    main()
