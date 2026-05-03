"""Reusable OEvent helpers with replay and lightweight result caching support."""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


DIR_PACKAGE = Path(__file__).resolve().parent
DIR_REPO = DIR_PACKAGE.parent
DIR_EXTERNAL_OEVENT = DIR_PACKAGE / 'external' / 'oevent'


def _bootstrap_oevent() -> None:
    """Expose the vendored OEvent checkout and optional stubs."""
    cache_dir = DIR_PACKAGE / '.mplcache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache_dir))
    for path in [DIR_REPO, DIR_EXTERNAL_OEVENT]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    try:
        __import__('hdf5storage')
    except ModuleNotFoundError:
        stub = types.ModuleType('hdf5storage')

        def _missing(*_args, **_kwargs):
            raise ModuleNotFoundError(
                'hdf5storage is required only for MATLAB/ECoG loading helpers in external/oevent'
            )

        stub.read = _missing
        sys.modules['hdf5storage'] = stub


_bootstrap_oevent()

from oevent import (  # noqa: E402
    GetDFrame,
    dbands,
    getCV2,
    getDynamicThresh,
    getFF,
    getLV,
    getblobIEI,
    getblobinrange,
    getmorletwin,
    getspecevents,
    mednorm,
    noiseampCSD,
    unitnorm,
)


NORMOPS = {
    'mednorm': mednorm,
    'unitnorm': unitnorm,
}


@dataclass(frozen=True)
class OEventSpectrogramParams:
    """Parameters that control Morlet spectrogram generation."""
    winsz: float
    sampr: float
    freqmin: float
    freqmax: float
    freqstep: float
    getphase: bool = True
    useloglfreq: bool = False
    mspecwidth: float = 7.0
    noiseamp: float = noiseampCSD
    normop_name: str = 'mednorm'

    def __post_init__(self) -> None:
        """Validate the configured normalization operator."""
        if self.normop_name not in NORMOPS:
            raise ValueError(
                f'Unsupported normop_name: {self.normop_name!r}. '
                f'Expected one of {sorted(NORMOPS)}'
            )

    def get_normop(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the configured spectrogram normalization function."""
        return NORMOPS[self.normop_name]

    def to_cache_dict(self) -> dict[str, float | bool | str]:
        """Return a stable cache-friendly representation of the settings."""
        return asdict(self)


@dataclass(frozen=True)
class OEventDetectionParams:
    """Parameters that can be replayed from a cached spectrogram."""
    medthresh: float
    overlapth: float = 0.5
    use_dyn_thresh: bool = False
    threshfctr: float = 2.0
    endfctr: float = 0.5
    band_overrides: Mapping[str, tuple[float, float]] | None = None

    def resolved_bands(self) -> dict[str, tuple[float, float]]:
        """Return OEvent bands merged with any per-run overrides."""
        bands = {name: (float(bounds[0]), float(bounds[1])) for name, bounds in dbands.items()}
        if self.band_overrides is None:
            return bands
        for name, bounds in self.band_overrides.items():
            if len(bounds) != 2:
                raise ValueError(f'Band override for {name!r} should have exactly 2 values')
            bands[name] = (float(bounds[0]), float(bounds[1]))
        return bands


@dataclass
class OEventSpectrogramBundle:
    """Cached Morlet windows and derived metadata for replay."""
    lms_raw: list
    lmsnorm: list[np.ndarray]
    lnoise: list[bool]
    lsidx: list[int]
    leidx: list[int]
    specsamp: int
    specdur: float
    scalex: float
    spectrogram_params: OEventSpectrogramParams

    def stacked_tfr(self, normalized: bool = True) -> np.ndarray:
        """Return the stitched spectrogram across all cached windows."""
        if normalized:
            return np.hstack(self.lmsnorm)
        return np.hstack([ms.TFR for ms in self.lms_raw])

    def freq_axis_hz(self) -> np.ndarray:
        """Return the shared frequency axis for the cached windows."""
        if not self.lms_raw:
            raise ValueError('Spectrogram bundle is empty')
        return np.asarray(self.lms_raw[0].f, dtype=float)

    def time_axis_s(self, time_offset_s: float = 0.0) -> np.ndarray:
        """Return the stitched time axis for the cached windows."""
        time_chunks = []
        for offidx, ms in zip(self.lsidx, self.lms_raw):
            chunk = time_offset_s + (
                float(offidx) + np.arange(ms.TFR.shape[1], dtype=float)
            ) / self.spectrogram_params.sampr
            time_chunks.append(chunk)
        return np.concatenate(time_chunks)


class OEventAnalyzer:
    """Thin wrapper around OEvent spectrogram construction and replay."""

    def __init__(self, spectrogram_params: OEventSpectrogramParams):
        """Store the spectrogram settings used by this analyzer."""
        self.spectrogram_params = spectrogram_params

    def build_bundle(self, dat: np.ndarray) -> OEventSpectrogramBundle:
        """Compute Morlet windows and normalized spectrograms for one trace."""
        signal = _as_1d_signal(dat)
        params = self.spectrogram_params
        lms, lnoise, lsidx, leidx = getmorletwin(
            signal,
            int(params.winsz * params.sampr),
            params.sampr,
            freqmin=params.freqmin,
            freqmax=params.freqmax,
            freqstep=params.freqstep,
            noiseamp=params.noiseamp,
            getphase=params.getphase,
            useloglfreq=params.useloglfreq,
            mspecwidth=params.mspecwidth,
        )
        if not lms:
            raise ValueError('OEvent returned no Morlet windows for the provided signal')
        normop = params.get_normop()
        lmsnorm = [normop(ms.TFR) for ms in lms]
        specsamp = int(lms[0].TFR.shape[1])
        specdur = float(specsamp / params.sampr)
        scalex = float(1e3 * specdur / specsamp)
        return OEventSpectrogramBundle(
            lms_raw=lms,
            lmsnorm=lmsnorm,
            lnoise=[bool(x) for x in lnoise],
            lsidx=[int(x) for x in lsidx],
            leidx=[int(x) for x in leidx],
            specsamp=specsamp,
            specdur=specdur,
            scalex=scalex,
            spectrogram_params=params,
        )

    def detect_from_signal(
            self,
            dat: np.ndarray,
            detection_params: OEventDetectionParams,
            MUA=None,
            ) -> dict:
        """Build a bundle from the signal and run OEvent detection."""
        bundle = self.build_bundle(dat)
        return self.detect_from_bundle(bundle, dat, detection_params=detection_params, MUA=MUA)

    def detect_from_bundle(
            self,
            bundle: OEventSpectrogramBundle,
            dat: np.ndarray,
            detection_params: OEventDetectionParams,
            MUA=None,
            ) -> dict:
        """Replay OEvent detection from a cached spectrogram bundle."""
        if bundle.spectrogram_params != self.spectrogram_params:
            raise ValueError('Spectrogram bundle parameters do not match the analyzer configuration')

        signal = _as_1d_signal(dat)
        params = self.spectrogram_params
        bands = detection_params.resolved_bands()
        if detection_params.use_dyn_thresh:
            evthresh = float(
                getDynamicThresh(
                    bundle.lmsnorm,
                    bundle.lnoise,
                    float(detection_params.threshfctr),
                    float(detection_params.medthresh),
                )
            )
        else:
            evthresh = float(detection_params.medthresh)

        llevent = getspecevents(
            bundle.lms_raw,
            bundle.lmsnorm,
            bundle.lnoise,
            evthresh,
            bundle.lsidx,
            bundle.leidx,
            signal,
            MUA,
            0,
            params.sampr,
            overlapth=float(detection_params.overlapth),
            endfctr=float(detection_params.endfctr),
            getphase=params.getphase,
        )

        dout = {
            'sampr': float(params.sampr),
            'medthresh': float(detection_params.medthresh),
            'winsz': float(params.winsz),
            'freqmin': float(params.freqmin),
            'freqmax': float(params.freqmax),
            'freqstep': float(params.freqstep),
            'overlapth': float(detection_params.overlapth),
            'threshfctr': float(detection_params.threshfctr),
            'useDynThresh': bool(detection_params.use_dyn_thresh),
            'mspecwidth': float(params.mspecwidth),
            'noiseamp': float(params.noiseamp),
            'endfctr': float(detection_params.endfctr),
            'lsidx': list(bundle.lsidx),
            'leidx': list(bundle.leidx),
            'specsamp': int(bundle.specsamp),
            'specdur': float(bundle.specdur),
            'scalex': float(bundle.scalex),
            'lchan': [0],
        }
        dout[0] = _empty_channel_result(bands)
        dout[0]['lnoise'] = list(bundle.lnoise)
        dout[0]['evthresh'] = evthresh

        for levent in llevent:
            for band, (minf, maxf) in bands.items():
                lband = getblobinrange(levent, minf, maxf)
                count = len(lband)
                dout[0][band]['Count'].append(count)
                dout[0][band]['levent'].append(lband)
                if count > 2:
                    lband_iei = getblobIEI(lband, bundle.scalex)
                    dout[0][band]['IEI'].append(lband_iei)
                    dout[0][band]['CV'].append(getCV2(lband_iei))
                    if count > 3:
                        dout[0][band]['LV'].append(getLV(lband_iei))
                else:
                    dout[0][band]['IEI'].append([])
            for band in bands:
                dout[0][band]['FF'] = getFF(dout[0][band]['Count'])

        return dout

    def to_dataframe(
            self,
            dout: dict,
            dat: np.ndarray,
            MUA=None,
            alignby: str = 'bywaveletpeak',
            haveMUA: bool = False,
            ):
        """Convert an OEvent result dict into the standard dataframe."""
        signal_2d = _as_signal_2d(dat)
        return GetDFrame(
            dout,
            self.spectrogram_params.sampr,
            signal_2d,
            MUA,
            alignby=alignby,
            haveMUA=haveMUA,
        )


def _empty_channel_result(
        bands: Mapping[str, tuple[float, float]],
        ) -> dict[str, dict[str, list | float | None]]:
    """Build the upstream-like per-band output skeleton."""
    return {
        band: {'LV': [], 'CV': [], 'Count': [], 'FF': None, 'levent': [], 'IEI': []}
        for band in bands
    }


def _as_1d_signal(dat: np.ndarray) -> np.ndarray:
    """Convert a user signal into the 1-D form OEvent expects here."""
    signal = np.asarray(dat, dtype=float)
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2 and 1 in signal.shape:
        return signal.reshape(-1)
    raise ValueError(f'Expected a 1-D signal or a single-channel 2-D array, got shape={signal.shape}')


def _as_signal_2d(dat: np.ndarray) -> np.ndarray:
    """Convert a signal into the 2-D shape expected by GetDFrame."""
    signal = np.asarray(dat, dtype=float)
    if signal.ndim == 1:
        return signal[np.newaxis, :]
    if signal.ndim == 2:
        return signal
    raise ValueError(f'Expected a 1-D or 2-D signal array, got shape={signal.shape}')


def normalize_band_event_table(
        events: pd.DataFrame,
        bands_of_interest: Sequence[str],
        sampr: float,
        time_offset_s: float,
        resolved_y: float,
        channel_index: int,
        min_ncycle: float | None = None,
        max_foct: float | None = None,
        min_filtsigcor: float | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select bands of interest, apply quality filters, and add canonical columns."""
    bands_of_interest = [str(band) for band in bands_of_interest]
    selected = events.loc[events['band'].isin(bands_of_interest)].copy()
    keep_mask = np.ones(len(selected), dtype=bool)
    if min_ncycle is not None:
        keep_mask &= selected['ncycle'].to_numpy(dtype=float) >= float(min_ncycle)
    if max_foct is not None:
        keep_mask &= selected['Foct'].to_numpy(dtype=float) <= float(max_foct)
    if min_filtsigcor is not None:
        keep_mask &= selected['filtsigcor'].to_numpy(dtype=float) >= float(min_filtsigcor)

    selected['event_channel'] = int(channel_index)
    selected['event_y'] = float(resolved_y)
    selected['event_band'] = selected['band'].astype(str)
    selected['event_passed'] = keep_mask
    selected['t_offset_s'] = float(time_offset_s)
    selected['peak_time_s'] = time_offset_s + selected['absPeakT'] / 1e3
    selected['start_time_s'] = time_offset_s + selected['absminT'] / 1e3
    selected['stop_time_s'] = time_offset_s + selected['absmaxT'] / 1e3
    selected['duration_s'] = selected['dur'] / 1e3
    selected['sampr_hz'] = float(sampr)
    selected = selected.drop(columns=['band', 'chan'], errors='ignore')

    summary_cols = [
        'event_channel',
        'event_y',
        'event_band',
        'event_passed',
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
        if col not in selected.columns:
            selected[col] = np.nan
    selected = selected.sort_values(['event_channel', 'peak_time_s']).reset_index(drop=True)
    selected = selected[summary_cols + [col for col in selected.columns if col not in summary_cols]]
    passed = selected.loc[selected['event_passed'].to_numpy(dtype=bool)].reset_index(drop=True)
    return selected, passed


def prepare_csv_event_table(
        events: pd.DataFrame,
        round_digits: int,
        drop_columns: Sequence[str] = ('CSDwvf', 'filtsig', 'selection_status'),
        ) -> pd.DataFrame:
    """Build a readable scalar-only CSV table from the normalized event table."""
    csv_table = events.copy()
    csv_table = csv_table.drop(columns=list(drop_columns), errors='ignore')
    csv_table = csv_table.drop(
        columns=[col for col in csv_table.columns if 'MUA' in col.upper()],
        errors='ignore',
    )
    keep_cols = [
        col for col in csv_table.columns
        if csv_table[col].map(is_scalar_event_value).all()
    ]
    csv_table = csv_table.loc[:, keep_cols]
    numeric_cols = csv_table.select_dtypes(include=[np.number]).columns
    csv_table.loc[:, numeric_cols] = csv_table.loc[:, numeric_cols].round(int(round_digits))
    return csv_table


def is_scalar_event_value(value) -> bool:
    """Return whether one value can be stored as a one-line scalar field."""
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


def resolve_xr_channel_selection(
        signal_raw: xr.DataArray,
        signal_proc: xr.DataArray,
        channel_mode: str,
        y: float | None = None,
        y_values: Sequence[float] | str | None = None,
        y_range: tuple[float, float] | None = None,
        y_step: float | None = None,
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """Resolve a single- or multi-channel selection from y x time arrays."""
    _validate_signal_array(signal_raw)
    _validate_signal_array(signal_proc)
    if signal_raw.dims != signal_proc.dims:
        raise ValueError('signal_raw and signal_proc should have identical dims')

    channel_mode = str(channel_mode).strip().lower()
    if channel_mode not in {'single', 'multi'}:
        raise ValueError("channel_mode should be either 'single' or 'multi'")

    if channel_mode == 'single':
        if y is None:
            raise ValueError('y should be provided in single-channel mode')
        raw_trace = signal_raw.sel(y=float(y), method='nearest').load()
        proc_trace = signal_proc.sel(y=float(y), method='nearest').load()
        resolved_y = float(raw_trace.coords['y'].item())
        return _stack_channel_traces([raw_trace], [proc_trace], [resolved_y])

    requested_y = _resolve_requested_y_values(signal_raw.coords['y'].values, y_values, y_range, y_step)
    if requested_y is None:
        raw_selected = signal_raw.load()
        proc_selected = signal_proc.load()
        if 'y' not in raw_selected.dims:
            raise ValueError('Expected a y dimension in multi-channel mode')
        return _rename_y_to_channel(raw_selected), _rename_y_to_channel(proc_selected)

    raw_traces = []
    proc_traces = []
    resolved_y_values = []
    seen = set()
    for requested in requested_y:
        raw_trace = signal_raw.sel(y=float(requested), method='nearest').load()
        proc_trace = signal_proc.sel(y=float(requested), method='nearest').load()
        resolved_y = float(raw_trace.coords['y'].item())
        if resolved_y in seen:
            continue
        seen.add(resolved_y)
        raw_traces.append(raw_trace)
        proc_traces.append(proc_trace)
        resolved_y_values.append(resolved_y)
    if not raw_traces:
        raise ValueError('No channels were selected for multi-channel mode')
    return _stack_channel_traces(raw_traces, proc_traces, resolved_y_values)


def build_band_event_result_dataset(
        signal_raw: np.ndarray,
        signal_proc: np.ndarray,
        spectrogram_norm: np.ndarray,
        time_s: np.ndarray,
        spec_time_s: np.ndarray,
        freq_hz: np.ndarray,
        channel_y: Sequence[float],
        event_table: pd.DataFrame,
        attrs: Mapping[str, object] | None = None,
        ) -> xr.Dataset:
    """Pack traces, spectrograms, and scalar event fields into a lightweight dataset."""
    signal_raw = np.asarray(signal_raw, dtype=float)
    signal_proc = np.asarray(signal_proc, dtype=float)
    spectrogram_norm = np.asarray(spectrogram_norm, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    spec_time_s = np.asarray(spec_time_s, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    channel_y = np.asarray(channel_y, dtype=float)
    if signal_raw.shape != signal_proc.shape:
        raise ValueError('signal_raw and signal_proc should have matching shapes')
    if signal_raw.ndim != 2:
        raise ValueError('signal_raw and signal_proc should have shape (channel, time)')
    if spectrogram_norm.ndim != 3:
        raise ValueError('spectrogram_norm should have shape (channel, freq, spec_time)')
    if signal_raw.shape[0] != spectrogram_norm.shape[0]:
        raise ValueError('Trace and spectrogram channel counts should match')
    if signal_raw.shape[1] != time_s.size:
        raise ValueError('time_s length should match the trace time axis')
    if spectrogram_norm.shape[1] != freq_hz.size:
        raise ValueError('freq_hz length should match the spectrogram frequency axis')
    if spectrogram_norm.shape[2] != spec_time_s.size:
        raise ValueError('spec_time_s length should match the spectrogram time axis')
    if signal_raw.shape[0] != channel_y.size:
        raise ValueError('channel_y length should match the channel axis')

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
            arr = np.asarray([_stringify_scalar_event_value(value) for value in values], dtype=str)
        event_vars[col] = ('event', arr)

    dataset = xr.Dataset(
        data_vars={
            'signal_raw': (['channel', 'time'], signal_raw),
            'signal_proc': (['channel', 'time'], signal_proc),
            'spectrogram_norm': (['channel', 'freq', 'spec_time'], spectrogram_norm),
            **event_vars,
        },
        coords={
            'channel': np.arange(signal_raw.shape[0], dtype=int),
            'time': time_s,
            'freq': freq_hz,
            'spec_time': spec_time_s,
            'event': np.arange(len(event_table), dtype=int),
            'channel_y': ('channel', channel_y),
        },
    )
    if attrs is not None:
        dataset.attrs.update({key: _encode_xr_attr_value(value) for key, value in attrs.items()})
    return dataset


def event_table_from_dataset(dataset: xr.Dataset) -> pd.DataFrame:
    """Reconstruct the scalar event table from a lightweight result dataset."""
    event_cols = {}
    for name, data in dataset.data_vars.items():
        if tuple(data.dims) != ('event',):
            continue
        values = data.values
        if values.dtype.kind in {'U', 'S'}:
            event_cols[name] = values.astype(str)
        else:
            event_cols[name] = values
    if not event_cols:
        return pd.DataFrame(index=np.arange(dataset.sizes.get('event', 0), dtype=int))
    return pd.DataFrame(event_cols)


def stack_bundle_spectrograms(
        bundles: Sequence[OEventSpectrogramBundle],
        time_offset_s: float = 0.0,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack cached normalized spectrograms across channels."""
    if not bundles:
        raise ValueError('At least one spectrogram bundle is required')
    freq_hz = bundles[0].freq_axis_hz()
    spec_time_s = bundles[0].time_axis_s(time_offset_s=time_offset_s)
    stacked = []
    for bundle in bundles:
        bundle_freq = bundle.freq_axis_hz()
        bundle_time = bundle.time_axis_s(time_offset_s=time_offset_s)
        if not np.array_equal(bundle_freq, freq_hz):
            raise ValueError('All bundles should share the same frequency axis')
        if not np.array_equal(bundle_time, spec_time_s):
            raise ValueError('All bundles should share the same spectrogram time axis')
        stacked.append(bundle.stacked_tfr(normalized=True))
    return np.stack(stacked, axis=0), freq_hz, spec_time_s


def _validate_signal_array(signal: xr.DataArray) -> None:
    """Check that one xarray signal uses y x time style dimensions."""
    if not isinstance(signal, xr.DataArray):
        raise TypeError('signal should be an xarray DataArray')
    if 'time' not in signal.dims or 'y' not in signal.dims:
        raise ValueError(f'Expected signal dims including time and y, got {signal.dims}')


def _resolve_requested_y_values(
        available_y,
        y_values: Sequence[float] | str | None,
        y_range: tuple[float, float] | None,
        y_step: float | None,
        ) -> list[float] | None:
    """Resolve the requested y targets for multi-channel selection."""
    if isinstance(y_values, str):
        if y_values.strip().lower() == 'all':
            return None
        raise ValueError("y_values should be a sequence, None, or the string 'all'")
    if y_values is not None:
        return [float(value) for value in y_values]
    if y_range is None:
        return None
    y0, y1 = [float(value) for value in y_range]
    if y1 < y0:
        y0, y1 = y1, y0
    if y_step is None:
        available_y = np.asarray(available_y, dtype=float)
        return available_y[(available_y >= y0) & (available_y <= y1)].tolist()
    if float(y_step) <= 0:
        raise ValueError('y_step should be positive when provided')
    return np.arange(y0, y1 + 0.5 * float(y_step), float(y_step), dtype=float).tolist()


def _stack_channel_traces(
        raw_traces: Sequence[xr.DataArray],
        proc_traces: Sequence[xr.DataArray],
        resolved_y_values: Sequence[float],
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """Stack per-y traces into DataArrays with channel and time dimensions."""
    raw_values = np.stack([np.asarray(trace.values, dtype=float) for trace in raw_traces], axis=0)
    proc_values = np.stack([np.asarray(trace.values, dtype=float) for trace in proc_traces], axis=0)
    time_values = np.asarray(raw_traces[0].coords['time'].values, dtype=float)
    coords = {
        'channel': np.arange(len(raw_traces), dtype=int),
        'time': time_values,
        'channel_y': ('channel', np.asarray(resolved_y_values, dtype=float)),
    }
    raw_stacked = xr.DataArray(raw_values, dims=('channel', 'time'), coords=coords, name='signal_raw')
    proc_stacked = xr.DataArray(proc_values, dims=('channel', 'time'), coords=coords, name='signal_proc')
    return raw_stacked, proc_stacked


def _rename_y_to_channel(signal: xr.DataArray) -> xr.DataArray:
    """Rename a y x time signal to channel x time while preserving depths."""
    signal = signal.transpose('y', 'time').rename({'y': 'channel'})
    return signal.assign_coords(channel=np.arange(signal.sizes['channel'], dtype=int)).assign_coords(
        channel_y=('channel', np.asarray(signal.coords['channel'].values, dtype=float))
    )


def _stringify_scalar_event_value(value) -> str:
    """Convert one scalar event value into a stable string representation."""
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value)


def _encode_xr_attr_value(value):
    """Convert one attribute value into a NetCDF-friendly scalar."""
    if value is None:
        return 'null'
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return value
    return json.dumps(value, sort_keys=True)
