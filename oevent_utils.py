"""Reusable OEvent helpers with spectrogram replay support."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


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
            chunk = time_offset_s + (float(offidx) + np.arange(ms.TFR.shape[1])) / self.spectrogram_params.sampr
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
    if signal.ndim == 2:
        if 1 in signal.shape:
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
