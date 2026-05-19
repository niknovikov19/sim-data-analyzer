"""Demo: collect job-produced JSON rates into one batch xarray artifact."""

from __future__ import annotations

import json
import math
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
    collect_batch_json,
    extract_batch_params_to_xr,
)


DIRPATH_BATCH = DIR_PACKAGE / "dev_scratch" / "data_src" / "hpc_remote" / "grid_5x5_rate_pv"
DIRPATH_CFG = DIRPATH_BATCH / "cfg"
DIRPATH_JSON = DIRPATH_BATCH / "results"
DIRPATH_CACHE = DIR_PACKAGE / "dev_scratch" / "data_proc" / "grid_5x5_rate_pv"
DIRPATH_RESULTS = DIR_PACKAGE / "dev_scratch" / "results" / "grid_5x5_rate_pv"

CFG_PARAM_FIELDS = {
    "rxe": "rxe",
    "rxi": "rxi",
}

RATES_CACHE_NAME = "batch_rates__source-json__var-rates__pops-no-frz__lazy.nc"
SUMMARY_NAME = "batch_json_rates_summary.md"
POP_MEAN_PNG_NAME = "batch_json_rates__mean_by_pop.png"
POP_HEATMAPS_PNG_NAME = "batch_json_rates__heatmaps_all_pops.png"


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

    # Read the first JSON file to keep the population order from the job output.
    fpath_first_json = sorted(DIRPATH_JSON.glob("result_*.json"))[0]
    payload_first = json.loads(fpath_first_json.read_text(encoding="utf-8"))
    pop_names = [pop_name for pop_name in payload_first["rates"] if "frz" not in pop_name]

    # Collect all per-job rates into one cached batch Dataset.
    fpath_cache = DIRPATH_CACHE / RATES_CACHE_NAME
    X = collect_batch_json(
        job_idx_xr,
        DIRPATH_JSON,
        var_mappings={"rate": "rates"},
        fname_templ="result_{job:05d}_*.json",
        dict_dims={"pop": pop_names},
        cache_path=fpath_cache,
        lazy=True,
    )
    rate_xr = X["rate"].load()

    # Summarize the collected batch in one markdown file.
    rate_mean_by_pop = rate_xr.mean(dim=("rxe", "rxi"), skipna=True).to_series().sort_values(ascending=False)
    summary_lines = [
        "# Batch JSON Rates Extraction",
        "",
        f"- Batch root: `{DIRPATH_BATCH}`",
        f"- Cache file: `{fpath_cache}`",
        f"- Source type: `json`",
        f"- Extracted variable: `rates -> rate`",
        f"- Populations: `{len(pop_names)}`",
        "- Population filter: exclude names containing `frz`",
        f"- Batch dims: `{rate_xr.dims}`",
        f"- Batch shape: `{tuple(rate_xr.shape)}`",
        "",
        "## Batch Coordinates",
        "",
        f"- `rxe`: {list(map(float, rate_xr.coords['rxe'].values.tolist()))}",
        f"- `rxi`: {list(map(float, rate_xr.coords['rxi'].values.tolist()))}",
        "",
        "## Rate Range",
        "",
        f"- min: `{float(np.nanmin(rate_xr.values)):.6g}`",
        f"- max: `{float(np.nanmax(rate_xr.values)):.6g}`",
        "",
        "## Top Populations By Mean Rate",
        "",
    ]
    for pop_name, value in rate_mean_by_pop.head(12).items():
        summary_lines.append(f"- `{pop_name}`: `{float(value):.6g}`")
    (DIRPATH_RESULTS / SUMMARY_NAME).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Plot the mean rate of each population across the whole batch.
    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(pop_names))))
    yy = np.arange(len(rate_mean_by_pop))
    ax.barh(yy, rate_mean_by_pop.values, color="#1f6feb")
    ax.set_yticks(yy)
    ax.set_yticklabels(rate_mean_by_pop.index.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean rate")
    ax.set_title("Mean population rate across the batch")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIRPATH_RESULTS / POP_MEAN_PNG_NAME, dpi=150)
    plt.close(fig)

    # Plot one rxe x rxi heatmap for every population.
    n_pop = len(pop_names)
    ncols = 4
    nrows = math.ceil(n_pop / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    vmin = float(np.nanmin(rate_xr.values))
    vmax = float(np.nanmax(rate_xr.values))
    image = None
    for idx, pop_name in enumerate(pop_names):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        X_pop = rate_xr.sel(pop=pop_name)
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
        if row == nrows - 1:
            ax.set_xlabel("rxi")
        if col == 0:
            ax.set_ylabel("rxe")

    for idx in range(n_pop, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.75, label="rate")
    fig.suptitle("Population rates across the rxe x rxi batch", fontsize=14)
    fig.savefig(DIRPATH_RESULTS / POP_HEATMAPS_PNG_NAME, dpi=150)
    plt.close(fig)

    print(f"Saved batch cache: {fpath_cache}")
    print(f"Saved summary: {DIRPATH_RESULTS / SUMMARY_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / POP_MEAN_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / POP_HEATMAPS_PNG_NAME}")


if __name__ == "__main__":
    main()
