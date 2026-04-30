import copy

import numpy as np
import xarray as xr

from sim_data_analyzer.spike_data import SpikeData


def _maybe_store_proc_info(X_out, func_name, params, store_proc_info):
    """Store processing metadata in a simple JSON-serializable attr. """
    if not store_proc_info:
        return X_out

    proc_steps = copy.deepcopy(X_out.attrs.get('proc_steps', []))
    if not isinstance(proc_steps, list):
        proc_steps = []
    proc_steps.append({'name': func_name, 'params': params})
    X_out.attrs['proc_steps'] = proc_steps
    return X_out


def _attach_proc_info_to_epochs(epochs, func_name, params, store_proc_info):
    """Attach processing metadata to one epoch output or a dict of outputs. """
    if isinstance(epochs, dict):
        return {
            pop_name: _maybe_store_proc_info(X_out, func_name, params, store_proc_info)
            for pop_name, X_out in epochs.items()
        }
    return _maybe_store_proc_info(epochs, func_name, params, store_proc_info)


def _validate_time_units(time_units):
    """Validate supported time units. """
    if time_units not in {'s', 'ms'}:
        raise ValueError(f'Unsupported time_units {time_units!r}')


def _validate_time_coord(time_coord):
    """Validate that the time coordinate is 1D, monotonic, and regular. """
    if time_coord.ndim != 1:
        raise ValueError('Time coordinate should be 1-dimensional')
    tt = np.asarray(time_coord.values, dtype=float)
    if tt.size < 2:
        raise ValueError('Time coordinate should contain at least 2 points')
    dt = np.diff(tt)
    if np.any(dt <= 0):
        raise ValueError('Time coordinate should be strictly increasing')
    if not np.allclose(dt, dt[0]):
        raise ValueError('Time coordinate should be regularly sampled')
    return tt, float(dt[0])


def _time_scale_factor(spikes: SpikeData, time_units: str) -> float:
    """Get a conversion factor from SpikeData units to signal units. """
    spike_ms = bool(spikes.metadata['ms'])
    if spike_ms and time_units == 's':
        return 1e-3
    if (not spike_ms) and time_units == 'ms':
        return 1e3
    return 1.0


def _resolve_pop_names(X_in, spikes: SpikeData, pop_name, pop_dim):
    """Resolve the populations to process. """
    spike_pops = spikes.get_pop_names()
    if pop_name is not None:
        if pop_name not in spike_pops:
            raise ValueError(f'Population {pop_name!r} is not present in SpikeData')
        if pop_dim in X_in.dims and pop_name not in X_in.coords[pop_dim].values.tolist():
            raise ValueError(f'Population {pop_name!r} is not present in signal {pop_dim!r}')
        return [pop_name]

    if pop_dim in X_in.dims:
        signal_pops = X_in.coords[pop_dim].values.tolist()
        pop_names = [name for name in signal_pops if name in spike_pops]
        if not pop_names:
            raise ValueError('No overlapping populations between signal and SpikeData')
        return pop_names

    raise ValueError(f'pop_name should be provided when signal has no {pop_dim!r} dimension')


def _get_pop_trigger_times(spikes: SpikeData, pop_name: str, time_scale: float) -> np.ndarray:
    """Get pooled trigger times for one population in signal units. """
    spike_list = spikes.get_pop_spikes(pop_name)
    if spikes.combine_mode:
        trigger_times = spike_list[0]
    else:
        if spike_list:
            trigger_times = np.sort(np.concatenate(spike_list))
        else:
            trigger_times = np.array([], dtype=np.float64)
    return np.asarray(trigger_times, dtype=float) * time_scale


def _extract_epochs_for_slice(X_slice, trigger_times, time_win, time_dim):
    """Extract spike-triggered epochs for one xarray slice. """
    tt, dt = _validate_time_coord(X_slice.coords[time_dim])
    n_before = int(round(time_win[0] / dt))
    n_after = int(round(time_win[1] / dt))
    time_rel = np.arange(n_before, n_after + 1, dtype=np.float64) * dt

    values = np.asarray(X_slice.transpose(*[dim for dim in X_slice.dims if dim != time_dim], time_dim).values)
    base_dims = [dim for dim in X_slice.dims if dim != time_dim]
    base_shape = values.shape[:-1]

    epochs = []
    for spike_time in trigger_times:
        idx_center = int(round((spike_time - tt[0]) / dt))
        idx_start = idx_center + n_before
        idx_stop = idx_center + n_after + 1
        if (idx_start < 0) or (idx_stop > values.shape[-1]):
            continue
        epochs.append(values[..., idx_start:idx_stop])

    if epochs:
        epoch_values = np.stack(epochs, axis=0)
    else:
        epoch_values = np.full((0,) + base_shape + (len(time_rel),), np.nan, dtype=values.dtype)

    dims = ['spike'] + base_dims + ['time_rel']
    coords = {
        'spike': np.arange(epoch_values.shape[0]),
        'time_rel': time_rel,
    }
    for dim in base_dims:
        coords[dim] = X_slice.coords[dim]
    return xr.DataArray(epoch_values, dims=dims, coords=coords)


def extract_xr_spike_triggered_epochs(
        X_in: xr.DataArray,
        spikes: SpikeData,
        time_win,
        pop_name=None,
        time_dim='time',
        pop_dim='pop',
        cell_dim='cell_gid',
        time_units='s',
        store_proc_info=False
        ):
    """Extract spike-triggered epochs from an xarray signal."""
    _validate_time_units(time_units)
    if time_dim not in X_in.dims:
        raise ValueError(f'Time dimension {time_dim!r} is not present')

    pop_names = _resolve_pop_names(X_in, spikes, pop_name, pop_dim)
    time_scale = _time_scale_factor(spikes, time_units)
    source_attrs = copy.deepcopy(X_in.attrs)
    params = {
        'time_win': list(time_win),
        'time_dim': time_dim,
        'pop_dim': pop_dim,
        'cell_dim': cell_dim,
        'time_units': time_units,
        'pop_name': pop_name,
        'resolved_pops': pop_names,
    }

    if pop_dim in X_in.dims:
        epochs = {}
        for pop_name_ in pop_names:
            X_slice = X_in.sel({pop_dim: pop_name_})
            trigger_times = _get_pop_trigger_times(spikes, pop_name_, time_scale)
            X_epochs = _extract_epochs_for_slice(X_slice, trigger_times, time_win, time_dim)
            X_epochs.attrs = copy.deepcopy(source_attrs)
            epochs[pop_name_] = X_epochs
        return _attach_proc_info_to_epochs(
            epochs, 'extract_xr_spike_triggered_epochs', params, store_proc_info
        )

    trigger_times = _get_pop_trigger_times(spikes, pop_names[0], time_scale)
    X_epochs = _extract_epochs_for_slice(X_in, trigger_times, time_win, time_dim)
    X_epochs.attrs = copy.deepcopy(source_attrs)
    return _attach_proc_info_to_epochs(
        X_epochs, 'extract_xr_spike_triggered_epochs', params, store_proc_info
    )


def _average_epoch_dict(epoch_dict, pop_dim):
    """Average a dict of epoch arrays into one xarray with a pop dimension. """
    avg_list = []
    pop_names = []
    for pop_name, X_epochs in epoch_dict.items():
        X_avg = X_epochs.mean(dim='spike')
        X_avg = X_avg.expand_dims({pop_dim: [pop_name]})
        avg_list.append(X_avg)
        pop_names.append(pop_name)
    if avg_list:
        return xr.concat(avg_list, dim=pop_dim)
    raise ValueError('No populations were available for averaging')


def calc_xr_sta(
        X_in: xr.DataArray,
        spikes: SpikeData,
        time_win,
        pop_name=None,
        time_dim='time',
        pop_dim='pop',
        cell_dim='cell_gid',
        time_units='s',
        return_mode='avg',
        store_proc_info=False
        ):
    """Calculate spike-triggered averages from an xarray signal."""
    if return_mode not in {'avg', 'epochs', 'both'}:
        raise ValueError(f'Unsupported return_mode {return_mode!r}')

    params = {
        'time_win': list(time_win),
        'time_dim': time_dim,
        'pop_dim': pop_dim,
        'cell_dim': cell_dim,
        'time_units': time_units,
        'pop_name': pop_name,
        'return_mode': return_mode,
    }
    epochs = extract_xr_spike_triggered_epochs(
        X_in,
        spikes,
        time_win,
        pop_name=pop_name,
        time_dim=time_dim,
        pop_dim=pop_dim,
        cell_dim=cell_dim,
        time_units=time_units,
        store_proc_info=False,
    )

    if isinstance(epochs, dict):
        avg = _average_epoch_dict(epochs, pop_dim)
        avg.attrs = copy.deepcopy(X_in.attrs)
        epochs = _attach_proc_info_to_epochs(
            epochs, 'calc_xr_sta', params, store_proc_info
        )
    else:
        avg = epochs.mean(dim='spike')
        avg.attrs = copy.deepcopy(X_in.attrs)
        epochs = _attach_proc_info_to_epochs(
            epochs, 'calc_xr_sta', params, store_proc_info
        )

    avg = _maybe_store_proc_info(avg, 'calc_xr_sta', params, store_proc_info)

    if return_mode == 'avg':
        return avg
    if return_mode == 'epochs':
        return epochs
    return {'avg': avg, 'epochs': epochs}
