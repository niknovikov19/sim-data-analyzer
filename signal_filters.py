import numpy as np
import scipy.signal as sig


_SUPPORTED_BTYPES = {'bandpass', 'lowpass', 'highpass', 'bandstop'}


def _infer_fs(t):
    """Infer sampling rate from time bins. """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError('Time vector should be 1-dimensional')
    if t.size < 2:
        raise ValueError('Time vector should contain at least 2 samples')
    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError('Time vector should be strictly increasing')
    return 1.0 / dt


def _normalize_fband(fband, btype):
    """Normalize cutoff specification for scipy.signal.butter(). """
    if btype in {'bandpass', 'bandstop'}:
        ff = np.asarray(fband, dtype=float)
        if ff.shape != (2,):
            raise ValueError(f'{btype} expects a 2-element cutoff sequence')
        return ff

    if np.ndim(fband) != 0:
        raise ValueError(f'{btype} expects a scalar cutoff frequency')
    return float(fband)


def filter_signal(x, t=None, fband=None, order=3, btype='bandpass', fs=None):
    """Filter a 1D signal with a Butterworth filter and zero-phase application. """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError('Input signal should be 1-dimensional')
    if fband is None:
        raise ValueError('fband should be provided')
    if btype not in _SUPPORTED_BTYPES:
        raise ValueError(f'Unsupported btype {btype!r}')
    if fs is None:
        if t is None:
            raise ValueError('Either t or fs should be provided')
        fs = _infer_fs(t)

    cutoff = _normalize_fband(fband, btype)
    sos = sig.butter(order, cutoff, btype=btype, output='sos', fs=fs)
    return sig.sosfiltfilt(sos, x)


def filter_signal_bandpass(x, t, fband, order=3):
    """Filter a 1D signal with the legacy bandpass-style call shape. """
    return filter_signal(x, t=t, fband=fband, order=order, btype='bandpass')
