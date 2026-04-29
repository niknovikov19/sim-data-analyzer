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
