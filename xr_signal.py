import copy

import numpy as np
import xarray as xr

from sim_data_analyzer.signal_filters import filter_signal


def _maybe_store_proc_info(X_out, func_name, params, store_proc_info):
    """Store processing metadata in a simple JSON-serializable attr. """
    if not store_proc_info:
        return X_out

    # Extend an existing processing history if it is already present.
    proc_steps = copy.deepcopy(X_out.attrs.get('proc_steps', []))
    if not isinstance(proc_steps, list):
        proc_steps = []
    proc_steps.append({'name': func_name, 'params': params})
    X_out.attrs['proc_steps'] = proc_steps
    return X_out


def _finalize_result(X_out, source_attrs, func_name, params, compute, store_proc_info):
    """Realize deferred computation only when requested by the caller. """
    if compute:
        X_out = X_out.compute()

    # Restore input attrs before appending the current processing step.
    X_out.attrs = copy.deepcopy(source_attrs)
    X_out = _maybe_store_proc_info(X_out, func_name, params, store_proc_info)
    return X_out


def _validate_regular_time_coord(time_coord):
    """Validate that a time coordinate is 1D, monotonic, and regular. """
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


def _resolve_sampling_rate(dt, fs):
    """Resolve sampling rate and validate it against the time step. """
    if fs is None:
        return round(1.0 / dt, 5)

    fs = float(fs)
    dt_fs = 1.0 / fs
    if not np.isclose(dt_fs, dt, rtol=1e-5, atol=1e-12):
        raise ValueError('fs is inconsistent with the time coordinate spacing')
    return fs


def _get_crosscorr_sample_lags(n_time, dt, lag_window):
    """Convert a lag window in time units into integer sample lags. """
    min_lag = -(n_time - 1)
    max_lag = n_time - 1

    if lag_window is None:
        return np.arange(min_lag, max_lag + 1, dtype=np.int64)

    lag_window_arr = np.asarray(lag_window, dtype=float)
    if lag_window_arr.shape != (2,):
        raise ValueError('lag_window should be a length-2 sequence')
    if not np.all(np.isfinite(lag_window_arr)):
        raise ValueError('lag_window values should be finite')

    lag_min, lag_max = lag_window_arr.tolist()
    if lag_min > lag_max:
        raise ValueError('lag_window lower bound should not exceed upper bound')

    tol = 1e-12
    sample_min = int(np.ceil((lag_min / dt) - tol))
    sample_max = int(np.floor((lag_max / dt) + tol))
    sample_min = max(sample_min, min_lag)
    sample_max = min(sample_max, max_lag)
    if sample_min > sample_max:
        return np.array([], dtype=np.int64)
    return np.arange(sample_min, sample_max + 1, dtype=np.int64)


def _calc_crosscorr_1d(x1, x2, sample_lags, subtract_mean, normalize, use_full):
    """Cross-correlation for one 1D signal pair over selected lags. """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    sample_lags = np.asarray(sample_lags, dtype=np.int64)

    if subtract_mean:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)

    if normalize:
        denom = np.sqrt(np.sum(x1 * x1) * np.sum(x2 * x2))
        if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
            return np.full(sample_lags.shape, np.nan, dtype=np.float64)
    else:
        denom = float(x1.size)

    if sample_lags.size == 0:
        return np.empty(0, dtype=np.float64)

    if use_full:
        corr = np.correlate(x1, x2, mode='full')
        return corr / denom

    out = np.empty(sample_lags.shape, dtype=np.float64)
    for idx, lag in enumerate(sample_lags):
        if lag == 0:
            numer = np.sum(x1 * x2)
        elif lag > 0:
            numer = np.sum(x2[:-lag] * x1[lag:])
        else:
            numer = np.sum(x2[-lag:] * x1[:lag])
        out[idx] = numer / denom
    return out


def _interp_isolated_outliers_1d(values, time_values, z_thresh, rel_neighbor_thresh):
    """Interpolate isolated one-bin outliers in a 1D time series. """
    values = np.asarray(values, dtype=float)
    time_values = np.asarray(time_values, dtype=float)
    cleaned = values.copy()

    if values.size < 3:
        return cleaned

    left = values[:-2]
    center = values[1:-1]
    right = values[2:]

    finite = np.isfinite(left) & np.isfinite(center) & np.isfinite(right)
    same_side = ((center - left) * (center - right)) > 0

    neighbor_mean = 0.5 * (left + right)
    center_dev = np.abs(center - neighbor_mean)
    neighbor_span = np.abs(right - left)

    valid_diff = np.abs(np.diff(values))
    valid_diff = valid_diff[np.isfinite(valid_diff)]
    if valid_diff.size == 0:
        return cleaned

    diff_median = np.median(valid_diff)
    diff_mad = np.median(np.abs(valid_diff - diff_median))
    diff_scale = 1.4826 * diff_mad
    abs_thresh = diff_median + z_thresh * diff_scale
    abs_thresh = max(abs_thresh, np.finfo(float).eps)

    rel_thresh = rel_neighbor_thresh * neighbor_span
    core_mask = finite & same_side & (center_dev > abs_thresh) & (center_dev > rel_thresh)

    full_mask = np.zeros(values.shape, dtype=bool)
    full_mask[1:-1] = core_mask
    isolated_mask = full_mask.copy()
    isolated_mask[1:] &= ~full_mask[:-1]
    isolated_mask[:-1] &= ~full_mask[1:]

    outlier_idx = np.flatnonzero(isolated_mask)
    for idx in outlier_idx:
        dt = time_values[idx + 1] - time_values[idx - 1]
        if dt == 0 or not np.isfinite(dt):
            continue
        weight = (time_values[idx] - time_values[idx - 1]) / dt
        cleaned[idx] = values[idx - 1] + weight * (values[idx + 1] - values[idx - 1])

    return cleaned


def interp_time_outliers(
        X_in: xr.DataArray,
        time_dim: str = 'time',
        z_thresh: float = 8.0,
        rel_neighbor_thresh: float = 5.0
        ) -> xr.DataArray:
    """Interpolate isolated one-bin outliers along a time dimension.

    A sample is treated as an outlier only when it is an isolated one-bin
    excursion away from both immediate neighbors, and its deviation is large
    relative to both the local background and the neighbor-to-neighbor change.
    """
    if time_dim not in X_in.dims:
        raise ValueError(f'Time dimension {time_dim!r} is not present')

    time_coord = X_in.coords[time_dim]
    if time_coord.ndim != 1:
        raise ValueError('Time coordinate should be 1-dimensional')

    X_out = xr.apply_ufunc(
        _interp_isolated_outliers_1d,
        X_in,
        time_coord,
        kwargs={
            'z_thresh': z_thresh,
            'rel_neighbor_thresh': rel_neighbor_thresh,
        },
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
    )

    X_out = X_out.transpose(*X_in.dims)
    X_out.attrs = copy.deepcopy(X_in.attrs)
    X_out.attrs['outlier_interp'] = {
        'name': 'interp_time_outliers',
        'params': {
            'time_dim': time_dim,
            'z_thresh': z_thresh,
            'rel_neighbor_thresh': rel_neighbor_thresh,
        }
    }
    return X_out


def filter_xr_signal(
        X_in: xr.DataArray,
        fband,
        order: int = 3,
        btype: str = 'bandpass',
        fs: float | None = None,
        time_dim: str = 'time',
        compute: bool = True,
        store_proc_info: bool = True
        ) -> xr.DataArray:
    """Filter an xarray signal along a time dimension.

    If compute=False, return the xarray result without forcing computation.
    If compute=True, compute the result before returning it. Deferred behavior
    only matters for dask-backed or chunked inputs.
    """

    # The code below assumes that time is the last dimension
    if X_in.dims[-1] != time_dim:
        raise ValueError('Time should be the last dimension')

    tt0 = X_in.coords[time_dim].values
    if fs is None:
        fs = round(1. / (tt0[1] - tt0[0]), 5)  # Round to correct for numerical errors

    source_attrs = copy.deepcopy(X_in.attrs)
    Y = xr.apply_ufunc(
        filter_signal,
        X_in,
        kwargs={'fband': fband, 'order': order, 'btype': btype, 'fs': fs},
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
    )
    Y = Y.transpose(*X_in.dims)

    # Compute the result if needed, write the params to Y.attrs
    if np.ndim(fband) == 0:
        fband_attr = float(fband)
    else:
        fband_attr = np.asarray(fband, dtype=float).tolist()
    params = {
        'fband': fband_attr,
        'order': order,
        'btype': btype,
        'fs': fs,
        'time_dim': time_dim,
    }
    return _finalize_result(
        Y, source_attrs, 'filter_xr_signal', params, compute, store_proc_info
    )


def calc_xr_crosscorr(
        X1_in: xr.DataArray,
        X2_in: xr.DataArray,
        time_dim: str = 'time',
        fs: float | None = None,
        lag_window=None,
        subtract_mean: bool = False,
        normalize: bool = False,
        compute: bool = True,
        store_proc_info: bool = True
        ) -> xr.DataArray:
    """Calculate a cross-correlogram along a time dimension.

    By default the output follows the legacy analyzer convention and divides by
    the full trace length `N`. When `normalize=True`, the output is scaled by
    the L2 energy of the processed signals instead.
    """
    if time_dim not in X1_in.dims or time_dim not in X2_in.dims:
        raise ValueError(f'Time dimension {time_dim!r} is not present')
    if X1_in.dims[-1] != time_dim or X2_in.dims[-1] != time_dim:
        raise ValueError('Time should be the last dimension')

    tt1, dt1 = _validate_regular_time_coord(X1_in.coords[time_dim])
    tt2, dt2 = _validate_regular_time_coord(X2_in.coords[time_dim])
    if tt1.size != tt2.size:
        raise ValueError('Signals should have matching time lengths')
    if not np.isclose(dt1, dt2, rtol=1e-9, atol=1e-12):
        raise ValueError('Signals should have matching sample spacing')
    if not np.allclose(tt1, tt2, rtol=1e-9, atol=1e-12):
        raise ValueError('Signals should have matching time coordinates')

    fs = _resolve_sampling_rate(dt1, fs)
    sample_lags = _get_crosscorr_sample_lags(tt1.size, dt1, lag_window)
    lag_values = sample_lags.astype(np.float64) / fs
    use_full = lag_window is None

    # Reuse the same time coordinate object so xarray broadcast/alignment
    # does not fail on equivalent-but-not-identical float indexes.
    X2_work = X2_in.assign_coords({time_dim: X1_in.coords[time_dim]})

    source_attrs = copy.deepcopy(X1_in.attrs)
    Y = xr.apply_ufunc(
        _calc_crosscorr_1d,
        X1_in,
        X2_work,
        xr.DataArray(sample_lags, dims=['lag']),
        kwargs={
            'subtract_mean': subtract_mean,
            'normalize': normalize,
            'use_full': use_full,
        },
        input_core_dims=[[time_dim], [time_dim], ['lag']],
        output_core_dims=[['lag']],
        dask_gufunc_kwargs={'output_sizes': {'lag': len(sample_lags)}},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
    )
    Y = Y.assign_coords({'lag': ('lag', lag_values)})

    lag_window_attr = None
    if lag_window is not None:
        lag_window_attr = np.asarray(lag_window, dtype=float).tolist()
    params = {
        'time_dim': time_dim,
        'fs': fs,
        'lag_window': lag_window_attr,
        'subtract_mean': subtract_mean,
        'normalize': normalize,
    }
    return _finalize_result(
        Y, source_attrs, 'calc_xr_crosscorr', params, compute, store_proc_info
    )
