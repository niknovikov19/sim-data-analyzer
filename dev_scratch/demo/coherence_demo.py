"""Demo for band-averaged xr coherence on one pair of rate traces."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.scratch_data import (
    get_exp_label,
    get_proc_dir,
    get_rates_cache_path,
    load_or_extract_rates,
    load_sim_result,
)
from sim_data_analyzer.xr_signal import filter_xr_signal
from sim_data_analyzer.xr_spect import calc_xr_cpsd


FPATH_SIM_RESULT = (
    DIR_PACKAGE / 'dev_scratch' / 'data_src' / 'a1_lfp_30s' / 'data_00000_seed_1000.pkl'
)
DIRPATH_PROC_ROOT = DIR_PACKAGE / 'dev_scratch' / 'data_proc'
DIRPATH_RESULTS_ROOT = DIR_PACKAGE / 'dev_scratch' / 'results'
EXP_LABEL = get_exp_label(FPATH_SIM_RESULT)
DIRPATH_PROC = get_proc_dir(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT)
DIRPATH_OUT = DIRPATH_RESULTS_ROOT / EXP_LABEL / 'demo'

T_LIMITS = (10.0, 30.0)
RATE_DT = 1e-3
POP1 = 'IT3'
POP2 = 'PV3'

FILTER_FBAND = None
FILTER_ORDER = 3

WIN_LEN = 1.0
WIN_OVERLAP = 0.5
FMAX = 100.0
FBAND = (8.0, 14.0)


def _format_tag_value(value: float) -> str:
    """Format a numeric value into a compact filesystem-safe tag."""
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _get_fband_tag(fband) -> str:
    """Build the selected coherence-band tag."""
    return f'{_format_tag_value(fband[0])}_{_format_tag_value(fband[1])}'


def _select_pair(rates, pop1: str, pop2: str):
    """Validate and select the requested population pair."""
    pop_names = rates.pop.values.tolist()
    missing = [pop_name for pop_name in [pop1, pop2] if pop_name not in pop_names]
    if missing:
        raise ValueError(f'Requested populations are not present in the rates data: {missing}')
    return rates.sel(pop=pop1), rates.sel(pop=pop2)


def _maybe_filter_trace(trace):
    """Optionally bandpass-filter one rate trace before spectral analysis."""
    if FILTER_FBAND is None:
        return trace
    return filter_xr_signal(
        trace,
        fband=tuple(float(x) for x in FILTER_FBAND),
        order=FILTER_ORDER,
        btype='bandpass',
        compute=True,
        store_proc_info=True,
    )


def _compute_coherence(rate1, rate2):
    """Compute CPSD, coherence spectrum, and the complex band-mean summary."""
    auto1 = calc_xr_cpsd(
        rate1,
        rate1,
        win_len=WIN_LEN,
        win_overlap=WIN_OVERLAP,
        fmax=FMAX,
        compute=True,
        store_proc_info=True,
    )
    auto2 = calc_xr_cpsd(
        rate2,
        rate2,
        win_len=WIN_LEN,
        win_overlap=WIN_OVERLAP,
        fmax=FMAX,
        compute=True,
        store_proc_info=True,
    )
    cpsd = calc_xr_cpsd(
        rate1,
        rate2,
        win_len=WIN_LEN,
        win_overlap=WIN_OVERLAP,
        fmax=FMAX,
        compute=True,
        store_proc_info=True,
    )

    denom = np.sqrt(np.abs(auto1.values) * np.abs(auto2.values))
    coherence_vals = np.full(cpsd.shape, np.nan + 0j, dtype=np.complex128)
    valid = np.isfinite(denom) & (denom > np.finfo(float).eps)
    coherence_vals[valid] = cpsd.values[valid] / denom[valid]
    coherence = cpsd.copy(data=coherence_vals)

    fmask = (coherence.freq >= FBAND[0]) & (coherence.freq <= FBAND[1])
    if int(np.count_nonzero(fmask.values)) == 0:
        raise ValueError(f'No frequencies fall inside FBAND={FBAND}')
    band_mean = complex(coherence.where(fmask, drop=True).mean('freq').item())
    return cpsd, coherence, band_mean


def _make_plot(fpath_out: Path, coherence, band_mean: complex) -> None:
    """Render the coherence magnitude and phase spectra for one pair."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    freq = coherence.freq.values
    coh_mag = np.abs(coherence.values)
    coh_phase = np.angle(coherence.values)

    # Show the magnitude spectrum and highlight the averaging band.
    axes[0].plot(freq, coh_mag, color='#1f6feb', lw=2)
    axes[0].axvspan(FBAND[0], FBAND[1], color='#f2cc60', alpha=0.25)
    axes[0].set_ylabel('|C(f)|')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Coherence magnitude')

    # Show the phase spectrum on the same frequency grid.
    axes[1].plot(freq, coh_phase, color='#d29922', lw=2)
    axes[1].axvspan(FBAND[0], FBAND[1], color='#f2cc60', alpha=0.25)
    axes[1].set_ylabel('phase (rad)')
    axes[1].set_xlabel('frequency (Hz)')
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Coherence phase')

    fig.suptitle(
        f'Coherence demo: {POP1} vs {POP2}, '
        f'band=[{FBAND[0]:g}, {FBAND[1]:g}] Hz, '
        f'|mean|={abs(band_mean):.3f}, angle={np.angle(band_mean):+.3f} rad'
    )
    fig.tight_layout()
    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the single-pair firing-rate coherence demo."""
    sim_result = None
    rate_cache = get_rates_cache_path(FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    if not rate_cache.exists():
        print(f'Loading simulation result: {FPATH_SIM_RESULT}')
        sim_result = load_sim_result(FPATH_SIM_RESULT)

    rates = load_or_extract_rates(sim_result, FPATH_SIM_RESULT, DIRPATH_PROC_ROOT, RATE_DT)
    rates = rates.sel(time=slice(*T_LIMITS)).load()
    rate1, rate2 = _select_pair(rates, POP1, POP2)
    rate1 = _maybe_filter_trace(rate1)
    rate2 = _maybe_filter_trace(rate2)

    cpsd, coherence, band_mean = _compute_coherence(rate1, rate2)
    print(f'CPSD shape: {cpsd.shape}, freq bins: {cpsd.sizes["freq"]}')
    print(
        f'Band-mean coherence for {POP1} vs {POP2}: '
        f'complex={band_mean.real:+.6f}{band_mean.imag:+.6f}j, '
        f'|mean|={abs(band_mean):.6f}, angle={np.angle(band_mean):+.6f} rad'
    )

    DIRPATH_OUT.mkdir(parents=True, exist_ok=True)
    fpath_out = DIRPATH_OUT / f'coherence__{POP1}__{POP2}__band_{_get_fband_tag(FBAND)}.png'
    _make_plot(fpath_out, coherence, band_mean)
    print(f'Saved plot: {fpath_out}')


if __name__ == '__main__':
    main()
