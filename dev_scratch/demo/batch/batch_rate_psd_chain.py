"""Minimal demo: batch rates in one file, then cached PSD on top."""

from __future__ import annotations

import sys
from pathlib import Path

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.batch_xr import (
    extract_batch_params_to_xr,
    collect_batch_rates_from_pkl,
)
from sim_data_analyzer.xr_cache import load_or_run_xr
from sim_data_analyzer.xr_spect import calc_xr_welch


def run_batch_rate_psd_chain(
        dirpath_cfg,
        dirpath_batch_data,
        dirpath_cache,
        cfg_param_fields,
        *,
        fname_cfg_templ: str = "*_cfg.json",
        job_pos_in_fname: int = -2,
        fname_data_templ: str = "grid_{job:05d}_data.pkl",
        rates_cache_name: str = "batch_rates.nc",
        psd_cache_name: str = "batch_rates_psd.nc",
        t_limits=(0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        avg_cells: bool = True,
        rate_chunks: dict[str, int] | None = None,
        rate_open_kwargs: dict | None = None,
        psd_open_kwargs: dict | None = None,
        win_len: float = 2.0,
        win_overlap: float = 0.75,
        fmin: float = 2.0,
        fmax: float = 30.0,
        average: str = "median",
        compute_psd: bool = False,
        ):
    """Run a plain batch workflow: collect rates once, then cache PSD."""
    dirpath_cfg = Path(dirpath_cfg)
    dirpath_batch_data = Path(dirpath_batch_data)
    dirpath_cache = Path(dirpath_cache)
    dirpath_cache.mkdir(parents=True, exist_ok=True)

    # Build the batch index from cfg files.
    job_idx_xr = extract_batch_params_to_xr(
        dirpath_cfg,
        cfg_param_fields=cfg_param_fields,
        fname_cfg_templ=fname_cfg_templ,
        job_pos_in_fname=job_pos_in_fname,
    )

    # Pick explicit filenames for the cached artifacts.
    # Change rates_cache_name when you want to keep another batch-rates variant.
    rates_cache_path = dirpath_cache / rates_cache_name
    psd_cache_path = dirpath_cache / psd_cache_name

    # Collect the batch rates, reusing or building the batch file internally.
    rates_xr = collect_batch_rates_from_pkl(
        job_idx_xr,
        dirpath_batch_data,
        fname_templ=fname_data_templ,
        t_limits=t_limits,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
        avg_cells=avg_cells,
        cache_path=rates_cache_path,
        lazy=True,
        chunks=rate_chunks,
        open_kwargs=rate_open_kwargs,
    )

    # Cache the PSD step with parameter checking handled by load_or_run_xr().
    psd_xr, _ = load_or_run_xr(
        psd_cache_path,
        calc_xr_welch,
        rates_xr,
        win_len=win_len,
        win_overlap=win_overlap,
        fmin=fmin,
        fmax=fmax,
        average=average,
        compute=compute_psd,
        open_kwargs=psd_open_kwargs,
    )

    return {
        "job_idx_xr": job_idx_xr,
        "rates_xr": rates_xr,
        "rates_cache_path": rates_cache_path,
        "psd_xr": psd_xr,
        "psd_cache_path": psd_cache_path,
    }
