"""Demo: compare fixed-order batch rates and PSD from direct PKL vs SpikeData."""

from __future__ import annotations

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


DIRPATH_CACHE = DIR_PACKAGE / "dev_scratch" / "data_proc" / "grid_5x5_thal"
DIRPATH_RESULTS = DIR_PACKAGE / "dev_scratch" / "results" / "grid_5x5_thal"

RATES_PKL_NAME = "batch_rates__source-pkl__var-rates__pops-no-frz-fixed-order-windowed__dt-5ms__tau-20ms__lazy.nc"
RATES_SPIKE_NAME = "batch_rates__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__lazy.nc"
PSD_PKL_NAME = "batch_rates_psd__source-pkl__var-rates__pops-no-frz-fixed-order-windowed__dt-5ms__tau-20ms__f-2-80__lazy.nc"
PSD_SPIKE_NAME = "batch_rates_psd__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__f-2-80__lazy.nc"

SUMMARY_NAME = "batch_pkl_vs_spike_equivalence_summary.md"
RATE_DIFF_PNG_NAME = "batch_pkl_vs_spike_rates__mean_diff_heatmaps.png"
PSD_DIFF_PNG_NAME = "batch_pkl_vs_spike_psd__mean_absdiff_curves.png"


def _max_abs_diff_stats(X_diff: xr.DataArray) -> dict[str, float]:
    """Compute simple absolute-difference summary statistics."""
    abs_vals = np.abs(np.asarray(X_diff.values))
    finite_vals = abs_vals[np.isfinite(abs_vals)]
    return {
        "max_abs_diff": float(finite_vals.max()),
        "mean_abs_diff": float(finite_vals.mean()),
        "p95_abs_diff": float(np.percentile(finite_vals, 95)),
        "p99_abs_diff": float(np.percentile(finite_vals, 99)),
    }


def main() -> None:
    DIRPATH_RESULTS.mkdir(parents=True, exist_ok=True)

    # Load the current fixed-order direct and SpikeData cache artifacts.
    rates_pkl = xr.open_dataarray(DIRPATH_CACHE / RATES_PKL_NAME).load()
    rates_spike = xr.open_dataarray(DIRPATH_CACHE / RATES_SPIKE_NAME).load()
    psd_pkl = xr.open_dataarray(DIRPATH_CACHE / PSD_PKL_NAME).load()
    psd_spike = xr.open_dataarray(DIRPATH_CACHE / PSD_SPIKE_NAME).load()

    # Compare rates and PSD only after verifying aligned coordinates.
    if not rates_pkl.coords.equals(rates_spike.coords):
        raise ValueError("Rate caches are not aligned on the same coordinates")
    if not psd_pkl.coords.equals(psd_spike.coords):
        raise ValueError("PSD caches are not aligned on the same coordinates")

    rates_diff = rates_pkl - rates_spike
    psd_diff = psd_pkl - psd_spike
    rate_stats = _max_abs_diff_stats(rates_diff)
    psd_stats = _max_abs_diff_stats(psd_diff)

    # Summarize equivalence in a small markdown artifact.
    rate_mask_equal = bool(np.array_equal(np.isfinite(rates_pkl.values), np.isfinite(rates_spike.values)))
    psd_mask_equal = bool(np.array_equal(np.isfinite(psd_pkl.values), np.isfinite(psd_spike.values)))
    rate_mean_direct = rates_pkl.mean(dim=("rxe", "rxi", "time"), skipna=True).to_series()
    rate_mean_spike = rates_spike.mean(dim=("rxe", "rxi", "time"), skipna=True).to_series()
    rate_mean_absdiff = np.abs(rate_mean_direct - rate_mean_spike).sort_values(ascending=False)
    psd_band_direct = psd_pkl.mean(dim=("rxe", "rxi", "freq"), skipna=True).to_series()
    psd_band_spike = psd_spike.mean(dim=("rxe", "rxi", "freq"), skipna=True).to_series()
    psd_band_absdiff = np.abs(psd_band_direct - psd_band_spike).sort_values(ascending=False)

    summary_lines = [
        "# PKL vs SpikeData Equivalence",
        "",
        "## Rates",
        "",
        f"- direct cache: `{DIRPATH_CACHE / RATES_PKL_NAME}`",
        f"- spike-data cache: `{DIRPATH_CACHE / RATES_SPIKE_NAME}`",
        f"- finite-mask identical: `{rate_mask_equal}`",
        f"- max abs diff: `{rate_stats['max_abs_diff']:.6g}`",
        f"- mean abs diff: `{rate_stats['mean_abs_diff']:.6g}`",
        f"- p95 abs diff: `{rate_stats['p95_abs_diff']:.6g}`",
        f"- p99 abs diff: `{rate_stats['p99_abs_diff']:.6g}`",
        "",
        "### Mean-rate difference by population",
        "",
    ]
    for pop_name, value in rate_mean_absdiff.items():
        summary_lines.append(
            f"- `{pop_name}`: direct `{float(rate_mean_direct[pop_name]):.6g}`, "
            f"spike `{float(rate_mean_spike[pop_name]):.6g}`, "
            f"|diff| `{float(value):.6g}`"
        )

    summary_lines.extend([
        "",
        "## PSD",
        "",
        f"- direct cache: `{DIRPATH_CACHE / PSD_PKL_NAME}`",
        f"- spike-data cache: `{DIRPATH_CACHE / PSD_SPIKE_NAME}`",
        f"- finite-mask identical: `{psd_mask_equal}`",
        f"- max abs diff: `{psd_stats['max_abs_diff']:.6g}`",
        f"- mean abs diff: `{psd_stats['mean_abs_diff']:.6g}`",
        f"- p95 abs diff: `{psd_stats['p95_abs_diff']:.6g}`",
        f"- p99 abs diff: `{psd_stats['p99_abs_diff']:.6g}`",
        "",
        "### Mean-PSD difference by population",
        "",
    ])
    for pop_name, value in psd_band_absdiff.items():
        summary_lines.append(
            f"- `{pop_name}`: direct `{float(psd_band_direct[pop_name]):.6g}`, "
            f"spike `{float(psd_band_spike[pop_name]):.6g}`, "
            f"|diff| `{float(value):.6g}`"
        )
    (DIRPATH_RESULTS / SUMMARY_NAME).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Plot batch-mean rate-difference heatmaps for every no-frz population.
    mean_rate_diff = rates_diff.mean(dim="time", skipna=True).compute()
    vmax_rate = float(np.nanmax(np.abs(mean_rate_diff.values)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, squeeze=False)
    for idx, pop_name in enumerate(mean_rate_diff.coords["pop"].values):
        row = idx // 2
        col = idx % 2
        ax = axes[row][col]
        X_pop = mean_rate_diff.sel(pop=pop_name)
        image = ax.imshow(
            X_pop.values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax_rate,
            vmax=vmax_rate,
        )
        ax.set_title(f"{pop_name}: direct - spike", fontsize=10)
        ax.set_xticks(range(X_pop.sizes["rxi"]))
        ax.set_xticklabels([f"{float(x):g}" for x in X_pop.coords["rxi"].values], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(X_pop.sizes["rxe"]))
        ax.set_yticklabels([f"{float(x):g}" for x in X_pop.coords["rxe"].values], fontsize=8)
        if row == 1:
            ax.set_xlabel("rxi")
        if col == 0:
            ax.set_ylabel("rxe")
    fig.colorbar(image, ax=axes, shrink=0.8, label="mean rate diff")
    fig.suptitle("Batch-mean rate difference between direct PKL and SpikeData", fontsize=14)
    fig.savefig(DIRPATH_RESULTS / RATE_DIFF_PNG_NAME, dpi=150)
    plt.close(fig)

    # Plot mean absolute PSD-difference curves for every no-frz population.
    mean_abs_psd_diff = np.abs(psd_diff).mean(dim=("rxe", "rxi"), skipna=True).compute()
    fig, ax = plt.subplots(figsize=(10, 6))
    for pop_name in mean_abs_psd_diff.coords["pop"].values:
        ax.plot(
            mean_abs_psd_diff.coords["freq"].values,
            mean_abs_psd_diff.sel(pop=pop_name).values,
            label=str(pop_name),
            linewidth=1.5,
        )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean |PSD diff|")
    ax.set_title("Mean absolute PSD difference: direct PKL vs SpikeData")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(DIRPATH_RESULTS / PSD_DIFF_PNG_NAME, dpi=150)
    plt.close(fig)

    print(f"Saved summary: {DIRPATH_RESULTS / SUMMARY_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / RATE_DIFF_PNG_NAME}")
    print(f"Saved figure: {DIRPATH_RESULTS / PSD_DIFF_PNG_NAME}")


if __name__ == "__main__":
    main()
