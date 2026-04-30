import copy

import numpy as np
import scipy.signal as sig
import xarray as xr


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


def calc_xr_welch(
        X_in: xr.DataArray,
        win_len=4,
        win_overlap=0.75,
        fmin=2,
        fmax=30,
        fs=None,
        time_dim='time',
        window='hann',
        detrend='constant',
        scaling='density',
        average='median',
        compute=True,
        store_proc_info=True):
    """Calculate PSD using Welch method."""

    if X_in.dims[-1] != time_dim:
        raise ValueError('Time should be the last dimension')

    # Sampling rate
    tt0 = X_in.coords[time_dim].values
    if fs is None:
        fs = round(1. / (tt0[1] - tt0[0]), 5)

    # Window and overlap in samples
    nperseg = round(win_len * fs)
    noverlap = round(win_overlap * nperseg)

    # Frequencies from dummy signal
    xz = np.zeros(len(tt0))
    ff, _ = sig.welch(
        xz,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
        average=average,
        axis=-1,
    )

    def f(X, fs, window, nperseg, noverlap, detrend, scaling, average):
        _, S = sig.welch(
            X,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling,
            average=average,
            axis=-1,
        )
        return S

    source_attrs = copy.deepcopy(X_in.attrs)
    W = xr.apply_ufunc(
        f, X_in,
        kwargs={
            'fs': fs,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'detrend': detrend,
            'scaling': scaling,
            'average': average,
        },
        input_core_dims=[[time_dim]],
        output_core_dims=[['freq']],
        dask_gufunc_kwargs={'output_sizes': {'freq': len(ff)}} ,
        vectorize=False,
        dask='parallelized',
        output_dtypes=[np.float64],
    )

    W = W.assign_coords({'freq': ('freq', ff)})
    W = W.sel(freq=slice(fmin, fmax))

    params = {
        'win_len': win_len,
        'win_overlap': win_overlap,
        'fmin': fmin,
        'fmax': fmax,
        'fs': fs,
        'time_dim': time_dim,
        'window': window,
        'detrend': detrend,
        'scaling': scaling,
        'average': average,
    }

    return _finalize_result(
        W, source_attrs, 'calc_xr_welch', params, compute, store_proc_info
    )
    

def calc_xr_cpsd(X1_in: xr.DataArray, X2_in: xr.DataArray, 
                 win_len=0.5, win_overlap=0.5, fmax=100,
                 fs=None, time_dim='time', compute=True,
                 store_proc_info=True):
    """Cross-power spectral density between two signals, complex-valued.

    If compute=False, return the xarray result without forcing computation.
    If compute=True, compute the result before returning it. Deferred behavior
    only matters for dask-backed or chunked inputs.
    """

    # The code below assumes that time is the last dimension
    if (X1_in.dims[-1] != time_dim) or (X2_in.dims[-1] != time_dim):
        raise ValueError('Time should be the last dimension')
    
    # Sampling rate
    tt0 = X1_in.coords[time_dim].values
    if fs is None:
        fs = round(1. / (tt0[1] - tt0[0]), 5)  # Round to correct for numerical errors

    # Window and overlap in samples
    nperseg = round(win_len * fs)
    noverlap = round(win_overlap * win_len * fs)

    # Call sig.csd() on a surrogate array to get the output frequencies
    # Function name csd() is misleading and means cross-spetral density
    xz = np.zeros(len(tt0))
    ff, _ = sig.csd(xz, xz, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)

    # Wrapping function for sig.csd() that returns a single variable
    def f(X1, X2, fs, nperseg, noverlap):
        _, S = sig.csd(
            X1, X2, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
        return S
    
    # Apply sig.csd() to xr.DataArray's (with dask support)
    source_attrs = copy.deepcopy(X1_in.attrs)
    W = xr.apply_ufunc(
        f, X1_in, X2_in,
        kwargs={'fs': fs, 'nperseg': nperseg, 'noverlap': noverlap},
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[['freq']],
        dask_gufunc_kwargs={'output_sizes': {'freq': len(ff)}},
        vectorize=False, dask='parallelized',
        output_dtypes=[np.complex128]
    )
    W = W.assign_coords({'freq': ('freq', ff)})
    
    # Select freq. range of interest
    W = W.sel(freq=slice(None, fmax))

    # Compute the result if needed, write the params to W.attrs
    params = {
        'win_len': win_len,
        'win_overlap': win_overlap,
        'fmax': fmax,
        'fs': fs,
        'time_dim': time_dim,
    }
    return _finalize_result(
        W, source_attrs, 'calc_xr_cpsd', params, compute, store_proc_info
    )


def calc_xr_tf(X_in, win_len=0.5, win_overlap=0.5, fmax=100,
               fs=None, time_dim='time', compute=True,
               store_proc_info=True):
    """Calculate complex-valued time-frequency transform.

    If compute=False, return the xarray result without forcing computation.
    If compute=True, compute the result before returning it. Deferred behavior
    only matters for dask-backed or chunked inputs.
    """
    
    # The code below assumes that time is the last dimension
    if (X_in.dims[-1] != time_dim):
        raise ValueError('Time should be the last dimension')
    
    # Sampling rate
    tt0 = X_in.coords[time_dim].values
    if fs is None:
        fs = round(1. / (tt0[1] - tt0[0]), 5)  # Round to correct for numerical errors

    # Window and overlap in samples
    nperseg = round(win_len * fs)
    noverlap = round(win_overlap * win_len * fs)

    # Call sig.spectrogram() on a surrogate array to get the output
    # frequencies and time bins
    xz = np.zeros(len(tt0))
    ff, tt, _ = sig.spectrogram(
        xz, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Shift positions of W_ time bins to the closest X_in time bins
    idx = np.round(tt * fs).astype(int)
    tt = tt0[idx]

    # Wrapping function for sig.spectrogram() that returns a single variable
    def f(X, fs, nperseg, noverlap):
        _, _, S = sig.spectrogram(
            X, fs=fs, nperseg=nperseg, noverlap=noverlap,
            mode='complex', axis=-1)
        return S
    
    # Apply sig.spectrogram() to xr.DataArray's (with dask support)
    source_attrs = copy.deepcopy(X_in.attrs)
    W = xr.apply_ufunc(
        f, X_in,
        kwargs={'fs': fs, 'nperseg': nperseg, 'noverlap': noverlap},
        input_core_dims=[[time_dim]],
        output_core_dims=[['freq', 'time1']],
        dask_gufunc_kwargs={'output_sizes': {'freq': len(ff), 'time1': len(tt)}},
        vectorize=False, dask='parallelized',
        output_dtypes=[np.complex128]
    )
    W = W.rename({'time1': 'time'})    
    W = W.assign_coords(
         {'freq': ('freq', ff), 'time': ('time', tt)})
    
    # Select freq. range of interest
    W = W.sel(freq=slice(0, fmax))

    # Compute the result if needed, write the params to W.attrs
    params = {
        'win_len': win_len,
        'win_overlap': win_overlap,
        'fmax': fmax,
        'fs': fs,
        'time_dim': time_dim,
    }
    return _finalize_result(
        W, source_attrs, 'calc_xr_tf', params, compute, store_proc_info
    )
