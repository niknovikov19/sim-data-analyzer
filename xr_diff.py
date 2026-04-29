import copy

import numpy as np
import xarray as xr


def _diff_keepdims(X, n, axis):
    """Calculate derivative with initial padding to preserve dimensions. """
    npad1 = int(n / 2)
    npad2 = n - npad1
    X = np.concatenate(
        [np.take(X, np.arange(npad1), axis),
         X,
         np.take(X, np.arange(-npad2, 0), axis)],
        axis=axis
        )
    return np.diff(X, n, axis)


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


def calc_xr_diff(
        X: xr.DataArray,
        n: int = 1,
        ydim='y',
        compute=False,
        store_proc_info=False
        ):
    """Calculate diff of an xarray along the y-dimension, supports dask.

    If compute=False, return the xarray result without forcing computation.
    If compute=True, compute the result before returning it. Deferred behavior
    only matters for dask-backed or chunked inputs.
    """
    source_attrs = copy.deepcopy(X.attrs)
    Y = xr.apply_ufunc(
        _diff_keepdims, X,
        input_core_dims=[[ydim]], output_core_dims=[[ydim]],
        kwargs={'n': n, 'axis': -1},  # Because the core dim becomes last
        dask='parallelized', output_dtypes=[X.dtype]
    )
    Y = Y.assign_coords({ydim: X.coords[ydim]})
    Y = Y.transpose(*X.dims)
    params = {
        'n': n,
        'ydim': ydim,
    }
    return _finalize_result(
        Y, source_attrs, 'calc_xr_diff', params, compute, store_proc_info
    )


def calc_xr_bipolar(
        X: xr.DataArray,
        ydim='y',
        compute=False,
        store_proc_info=False
        ):
    """Calculate bipolar reference as the 1st y-derivative. """
    Y = calc_xr_diff(X, n=1, ydim=ydim, compute=compute, store_proc_info=False)
    params = {
        'ydim': ydim,
    }
    return _finalize_result(
        Y, X.attrs, 'calc_xr_bipolar', params, False, store_proc_info
    )


def calc_xr_csd(
        X: xr.DataArray,
        ydim='y',
        compute=False,
        store_proc_info=False
        ):
    """Calculate CSD as the 2nd y-derivative. """
    Y = calc_xr_diff(X, n=2, ydim=ydim, compute=compute, store_proc_info=False)
    params = {
        'ydim': ydim,
    }
    return _finalize_result(
        Y, X.attrs, 'calc_xr_csd', params, False, store_proc_info
    )
