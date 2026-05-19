"""Demo: cache per-job spikes, then collect batch rates and PSD from SpikeData."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DIR_PACKAGE = Path(__file__).resolve().parents[3]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.batch_xr import (
    collect_batch_rates_from_spike_data,
    extract_batch_params_to_xr,
    extract_batch_spike_data_from_pkl,
)
from sim_data_analyzer import netpyne_res_parse_utils as parse_utils
from sim_data_analyzer.xr_cache import load_or_run_xr
from sim_data_analyzer.xr_spect import calc_xr_welch


DIRPATH_BATCH = DIR_PACKAGE / "dev_scratch" / "data_src" / "hpc_remote" / "grid_5x5_thal"
DIRPATH_CFG = DIRPATH_BATCH / "cfg"
DIRPATH_PKL = DIRPATH_BATCH / "pkl"
DIRPATH_CACHE = DIR_PACKAGE / "dev_scratch" / "data_proc" / "grid_5x5_thal"
DIRPATH_SPIKES = DIRPATH_CACHE / "spike_cache__source-pkl__pops-all-fixed-order__t-1-end__abs-s"
DIRPATH_RESULTS = DIR_PACKAGE / "dev_scratch" / "results" / "grid_5x5_thal"

CFG_PARAM_FIELDS = {
    "rxe": "rxe",
    "rxi": "rxi",
}

SPIKE_T_LIMITS = (1.0, 5.0)
DT_BIN = 5e-3
TAU_SMOOTH = 20e-3
PSD_WIN_LEN = 2.0
PSD_WIN_OVERLAP = 0.75
PSD_FMIN = 2.0
PSD_FMAX = 80.0

RAW_RATES_CACHE_NAME = "batch_rates__source-spike-data__var-rates__pops-all-fixed-order__dt-5ms__tau-20ms__lazy.nc"
RAW_PSD_CACHE_NAME = "batch_rates_psd__source-spike-data__var-rates__pops-all-fixed-order__dt-5ms__tau-20ms__f-2-80__lazy.nc"
RATES_CACHE_NAME = "batch_rates__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__lazy.nc"
PSD_CACHE_NAME = "batch_rates_psd__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__f-2-80__lazy.nc"
SUMMARY_NAME = "batch_spike_rates_psd_summary.md"
RATE_MEAN_PNG_NAME = "batch_spike_rates__mean_by_pop.png"
RATE_HEATMAP_PNG_NAME = "batch_spike_rates__heatmaps_top_pops.png"
PSD_MEAN_PNG_NAME = "batch_spike_rates_psd__mean_by_pop.png"


def _drop_frz_pops(X):
    """Keep only populations whose names do not contain 'frz'."""
    pop_names = [str(pop_name) for pop_name in X.coords["pop"].values if "frz" not in str(pop_name)]
    return X.sel(pop=pop_names)


def _get_pop_names_from_first_pkl(dirpath_pkl: Path) -> list[str]:
    """Read one batch pkl and reuse its population order for every job."""
    fpath_first = sorted(dirpath_pkl.glob("data_*.pkl"))[0]
    with fpath_first.open("rb") as fobj:
        sim_result = pickle.load(fobj)
    return [str(pop_name) for pop_name in parse_utils.get_pop_names(sim_result)]


def run_batch_spike_rate_psd_chain(
        dirpath_cfg,
        dirpath_batch_data,
        dirpath_spikes,
        dirpath_cache,
        cfg_param_fields,
        *,
        fname_cfg_templ: str = "*_cfg.json",
        job_pos_in_fname: int = -2,
        fname_data_templ: str = "grid_{job:05d}_data.pkl",
        fname_spikes_templ: str = "spikes_{job:05d}.npz",
        rates_cache_name: str = "batch_spike_rates.nc",
        psd_cache_name: str | None = "batch_spike_rates_psd.nc",
        pop_names=None,
        spike_t_limits=(0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
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
    """Run a plain batch workflow using cached SpikeData as the rate source."""
    dirpath_cfg = Path(dirpath_cfg)
    dirpath_batch_data = Path(dirpath_batch_data)
    dirpath_spikes = Path(dirpath_spikes)
    dirpath_cache = Path(dirpath_cache)
    dirpath_spikes.mkdir(parents=True, exist_ok=True)
    dirpath_cache.mkdir(parents=True, exist_ok=True)

    # Build the batch index from cfg files.
    job_idx_xr = extract_batch_params_to_xr(
        dirpath_cfg,
        cfg_param_fields=cfg_param_fields,
        fname_cfg_templ=fname_cfg_templ,
        job_pos_in_fname=job_pos_in_fname,
    )

    # Extract one reusable SpikeData cache file per job.
    extract_batch_spike_data_from_pkl(
        job_idx_xr,
        dirpath_batch_data,
        dirpath_spikes,
        fname_data_templ=fname_data_templ,
        fname_spikes_templ=fname_spikes_templ,
        pop_names=pop_names,
        t_limits=spike_t_limits,
        combine=True,
        subtract_t0=False,
        ms=False,
    )

    # Collect the batch rates from the per-job SpikeData cache.
    rates_cache_path = dirpath_cache / rates_cache_name
    rates_xr = collect_batch_rates_from_spike_data(
        job_idx_xr,
        dirpath_spikes,
        fname_templ=fname_spikes_templ,
        t_limits=spike_t_limits if spike_t_limits[1] is not None else None,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
        cache_path=rates_cache_path,
        lazy=True,
        chunks=rate_chunks,
        open_kwargs=rate_open_kwargs,
    )

    # Optionally cache PSD on the raw all-pop batch rates.
    psd_cache_path = None
    psd_xr = None
    if psd_cache_name is not None:
        psd_cache_path = dirpath_cache / psd_cache_name
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
        "spike_dir": dirpath_spikes,
        "rates_xr": rates_xr,
        "rates_cache_path": rates_cache_path,
        "psd_xr": psd_xr,
        "psd_cache_path": psd_cache_path,
    }


def main() -> None:
    DIRPATH_CACHE.mkdir(parents=True, exist_ok=True)
    DIRPATH_SPIKES.mkdir(parents=True, exist_ok=True)
    DIRPATH_RESULTS.mkdir(parents=True, exist_ok=True)

    # Build the batch grid from the job cfg files.
    job_idx_xr = extract_batch_params_to_xr(
        DIRPATH_CFG,
        cfg_param_fields=CFG_PARAM_FIELDS,
        fname_cfg_templ="cfg_*.json",
        job_pos_in_fname=1,
    )
    pop_names = _get_pop_names_from_first_pkl(DIRPATH_PKL)

    # Extract one reusable SpikeData cache file per job.
    extract_batch_spike_data_from_pkl(
        job_idx_xr,
        DIRPATH_PKL,
        DIRPATH_SPIKES,
        fname_data_templ="data_{job:05d}_*.pkl",
        fname_spikes_templ="spikes_{job:05d}.npz",
        pop_names=pop_names,
        t_limits=SPIKE_T_LIMITS,
        combine=True,
        subtract_t0=False,
        ms=False,
    )

    # Collect the raw all-pop batch rates from the per-job SpikeData cache once.
    fpath_raw_rates_cache = DIRPATH_CACHE / RAW_RATES_CACHE_NAME
    if fpath_raw_rates_cache.exists():
        raw_rates_xr = xr.open_dataarray(fpath_raw_rates_cache)
    else:
        raw_rates_xr = collect_batch_rates_from_spike_data(
            job_idx_xr,
            DIRPATH_SPIKES,
            fname_templ="spikes_{job:05d}.npz",
            t_limits=SPIKE_T_LIMITS,
            dt_bin=DT_BIN,
            tau_smooth=TAU_SMOOTH,
            cache_path=fpath_raw_rates_cache,
            lazy=True,
            chunks={"time": 2000},
            open_kwargs={"chunks": {"rxe": 1, "rxi": 1, "time": 2000}},
        )

    # Cache the no-frz subset as the public rates artifact used below.
    fpath_rates_cache = DIRPATH_CACHE / RATES_CACHE_NAME
    rates_xr, _ = load_or_run_xr(
        fpath_rates_cache,
        _drop_frz_pops,
        raw_rates_xr.load(),
        load=True,
    )

    # Recompute or reuse the PSD that corresponds to the filtered public rates artifact.
    fpath_psd_cache = DIRPATH_CACHE / PSD_CACHE_NAME
    psd_xr, _ = load_or_run_xr(
        fpath_psd_cache,
        calc_xr_welch,
        rates_xr,
        win_len=PSD_WIN_LEN,
        win_overlap=PSD_WIN_OVERLAP,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        average="median",
        compute=True,
        load=True,
    )

    # Summarize the collected rates and PSD with small derived arrays.
    rate_mean_by_pop = rates_xr.mean(dim=("rxe", "rxi", "time"), skipna=True).compute()
    rate_mean_series = rate_mean_by_pop.to_series().sort_values(ascending=False)
    psd_mean_by_pop = psd_xr.mean(dim=("rxe", "rxi"), skipna=True).compute()
    psd_band_power = psd_mean_by_pop.mean(dim="freq", skipna=True)
    psd_band_power_series = psd_band_power.to_series().sort_values(ascending=False)
    psd_peak_freq = psd_mean_by_pop.idxmax(dim="freq", skipna=True).to_series()

    # Write one markdown summary for quick inspection in the IDE.
    summary_lines = [
        "# Batch SpikeData Rates + PSD Extraction",
        "",
        f"- Batch root: `{DIRPATH_BATCH}`",
        f"- Spike cache dir: `{DIRPATH_SPIKES}`",
        f"- Raw rates cache file: `{fpath_raw_rates_cache}`",
        f"- Public rates cache file: `{fpath_rates_cache}`",
        f"- Public PSD cache file: `{fpath_psd_cache}`",
        f"- Source type: `SpikeData`",
        f"- Upstream raw source: `pkl`",
        f"- Extracted signal: `population rates`",
        f"- PSD method: `Welch`",
        "- Population filter: exclude names containing `frz`",
        f"- Populations: `{rates_xr.sizes['pop']}`",
        f"- Rate dims: `{rates_xr.dims}`",
        f"- Rate shape: `{tuple(rates_xr.shape)}`",
        f"- PSD dims: `{psd_xr.dims}`",
        f"- PSD shape: `{tuple(psd_xr.shape)}`",
        "",
        "## Batch Coordinates",
        "",
        f"- `rxe`: {list(map(float, rates_xr.coords['rxe'].values.tolist()))}",
        f"- `rxi`: {list(map(float, rates_xr.coords['rxi'].values.tolist()))}",
        f"- time range: `{float(rates_xr.coords['time'].values[0]):.6g}` .. `{float(rates_xr.coords['time'].values[-1]):.6g}` s",
        f"- freq range: `{float(psd_xr.coords['freq'].values[0]):.6g}` .. `{float(psd_xr.coords['freq'].values[-1]):.6g}` Hz",
        "",
        "## Top Populations By Mean Rate",
        "",
    ]
    for pop_name, value in rate_mean_series.head(12).items():
        summary_lines.append(f"- `{pop_name}`: `{float(value):.6g}`")
    summary_lines.extend([
        "",
        "## Top Populations By Mean PSD Band Power",
        "",
    ])
    for pop_name, value in psd_band_power_series.head(12).items():
        summary_lines.append(
            f"- `{pop_name}`: `{float(value):.6g}` at peak `{float(psd_peak_freq[pop_name]):.6g}` Hz"
        )
    (DIRPATH_RESULTS / SUMMARY_NAME).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Plot the mean rate of each population across the whole batch and time axis.
    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(rate_mean_series))))
    yy = np.arange(len(rate_mean_series))
    ax.barh(yy, rate_mean_series.values, color="#1f6feb")
    ax.set_yticks(yy)
    ax.set_yticklabels(rate_mean_series.index.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean rate")
    ax.set_title("Mean population rate across the batch from SpikeData cache")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIRPATH_RESULTS / RATE_MEAN_PNG_NAME, dpi=150)
    plt.close(fig)

    # Plot rxe x rxi heatmaps for the most active populations.
    top_pops = rate_mean_series.head(6).index.tolist()
    rate_mean_grid = rates_xr.mean(dim="time", skipna=True).sel(pop=top_pops).compute()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True, squeeze=False)
    image = None
    vmin = float(np.nanmin(rate_mean_grid.values))
    vmax = float(np.nanmax(rate_mean_grid.values))
    for idx, pop_name in enumerate(top_pops):
        row = idx // 3
        col = idx % 3
        ax = axes[row][col]
        X_pop = rate_mean_grid.sel(pop=pop_name)
        image = ax.imshow(
            X_pop.values,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(pop_name, fontsize=9)
        ax.set_xticks(range(X_pop.sizes["rxi"]))
        ax.set_xticklabels([f"{float(x):g}" for x in X_pop.coords["rxi"].values], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(X_pop.sizes["rxe"]))
        ax.set_yticklabels([f"{float(x):g}" for x in X_pop.coords["rxe"].values], fontsize=7)
        if row == 1:
            ax.set_xlabel("rxi")
        if col == 0:
            ax.set_ylabel("rxe")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="mean rate")
    fig.suptitle("Mean rates across the rxe x rxi batch from SpikeData cache", fontsize=14)
    fig.savefig(DIRPATH_RESULTS / RATE_HEATMAP_PNG_NAME, dpi=150)
    plt.close(fig)

    # Plot mean PSD curves for the populations with the strongest band power.
    top_psd_pops = psd_band_power_series.head(8).index.tolist()
    psd_plot = psd_mean_by_pop.sel(pop=top_psd_pops)
    fig, ax = plt.subplots(figsize=(10, 6))
    for pop_name in top_psd_pops:
        ax.plot(
            psd_plot.coords["freq"].values,
            psd_plot.sel(pop=pop_name).values,
            label=pop_name,
            linewidth=1.5,
        )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Mean population PSD across the batch from SpikeData cache")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DIRPATH_RESULTS / PSD_MEAN_PNG_NAME, dpi=150)
    plt.close(fig)

    print(f"Saved spike cache dir: {DIRPATH_SPIKES}")
    print(f"Saved raw rates cache: {fpath_raw_rates_cache}")
    print(f"Saved public rates cache: {fpath_rates_cache}")
    print(f"Saved public PSD cache: {fpath_psd_cache}")
    print(f"Saved summary: {DIRPATH_RESULTS / SUMMARY_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / RATE_MEAN_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / RATE_HEATMAP_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / PSD_MEAN_PNG_NAME}")


if __name__ == "__main__":
    main()
