"""Demo: collect batch rates from raw job pickles, then cache PSD on top."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR_PACKAGE = Path(__file__).resolve().parents[3]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.batch_xr import (
    collect_batch_rates_from_pkl,
    extract_batch_params_to_xr,
)
from sim_data_analyzer.xr_cache import load_or_run_xr
from sim_data_analyzer.xr_spect import calc_xr_welch


DIRPATH_BATCH = DIR_PACKAGE / "dev_scratch" / "data_src" / "hpc_remote" / "grid_5x5_thal"
DIRPATH_CFG = DIRPATH_BATCH / "cfg"
DIRPATH_PKL = DIRPATH_BATCH / "pkl"
DIRPATH_CACHE = DIR_PACKAGE / "dev_scratch" / "data_proc" / "grid_5x5_thal"
DIRPATH_RESULTS = DIR_PACKAGE / "dev_scratch" / "results" / "grid_5x5_thal"

CFG_PARAM_FIELDS = {
    "rxe": "rxe",
    "rxi": "rxi",
}

T_LIMITS = (1.0, None)
DT_BIN = 5e-3
TAU_SMOOTH = 20e-3
PSD_WIN_LEN = 2.0
PSD_WIN_OVERLAP = 0.75
PSD_FMIN = 2.0
PSD_FMAX = 80.0

RAW_RATES_CACHE_NAME = "batch_rates__source-pkl__var-rates__pops-all__dt-5ms__tau-20ms__lazy.nc"
RATES_CACHE_NAME = "batch_rates__source-pkl__var-rates__pops-no-frz__dt-5ms__tau-20ms__lazy.nc"
PSD_CACHE_NAME = "batch_rates_psd__source-pkl__var-rates__pops-no-frz__dt-5ms__tau-20ms__f-2-80__lazy.nc"
SUMMARY_NAME = "batch_pkl_rates_psd_summary.md"
RATE_MEAN_PNG_NAME = "batch_pkl_rates__mean_by_pop.png"
RATE_HEATMAP_PNG_NAME = "batch_pkl_rates__heatmaps_top_pops.png"
PSD_MEAN_PNG_NAME = "batch_pkl_rates_psd__mean_by_pop.png"


def _drop_frz_pops(X):
    """Keep only populations whose names do not contain 'frz'."""
    pop_names = [str(pop_name) for pop_name in X.coords["pop"].values if "frz" not in str(pop_name)]
    return X.sel(pop=pop_names)


def main() -> None:
    DIRPATH_CACHE.mkdir(parents=True, exist_ok=True)
    DIRPATH_RESULTS.mkdir(parents=True, exist_ok=True)

    # Build the batch grid from the job cfg files.
    job_idx_xr = extract_batch_params_to_xr(
        DIRPATH_CFG,
        cfg_param_fields=CFG_PARAM_FIELDS,
        fname_cfg_templ="cfg_*.json",
        job_pos_in_fname=1,
    )

    # Collect all populations into one lazy batch file once.
    fpath_raw_rates_cache = DIRPATH_CACHE / RAW_RATES_CACHE_NAME
    raw_rates_xr = collect_batch_rates_from_pkl(
        job_idx_xr,
        DIRPATH_PKL,
        fname_templ="data_{job:05d}_*.pkl",
        t_limits=T_LIMITS,
        dt_bin=DT_BIN,
        tau_smooth=TAU_SMOOTH,
        avg_cells=True,
        cache_path=fpath_raw_rates_cache,
        lazy=True,
        open_kwargs={"chunks": {"rxe": 1, "rxi": 1, "time": 2000}},
        chunks={"time": 2000},
    )

    # Cache the no-frz subset as the public rates artifact used below.
    fpath_rates_cache = DIRPATH_CACHE / RATES_CACHE_NAME
    rates_xr, _ = load_or_run_xr(
        fpath_rates_cache,
        _drop_frz_pops,
        raw_rates_xr,
        open_kwargs={"chunks": {"rxe": 1, "rxi": 1, "time": 2000}},
    )

    # Compute or reuse PSD from the cached batch rates artifact.
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
        compute=False,
        open_kwargs={"chunks": {"rxe": 1, "rxi": 1, "freq": 128}},
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
        "# Batch PKL Rates + PSD Extraction",
        "",
        f"- Batch root: `{DIRPATH_BATCH}`",
        f"- Rates cache file: `{fpath_rates_cache}`",
        f"- PSD cache file: `{fpath_psd_cache}`",
        f"- Source type: `pkl`",
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
    ax.set_title("Mean population rate across the batch")
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
    fig.suptitle("Mean rates across the rxe x rxi batch", fontsize=14)
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
    ax.set_title("Mean population PSD across the batch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DIRPATH_RESULTS / PSD_MEAN_PNG_NAME, dpi=150)
    plt.close(fig)

    print(f"Saved rates cache: {fpath_rates_cache}")
    print(f"Saved PSD cache: {fpath_psd_cache}")
    print(f"Saved summary: {DIRPATH_RESULTS / SUMMARY_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / RATE_MEAN_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / RATE_HEATMAP_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / PSD_MEAN_PNG_NAME}")


if __name__ == "__main__":
    main()
