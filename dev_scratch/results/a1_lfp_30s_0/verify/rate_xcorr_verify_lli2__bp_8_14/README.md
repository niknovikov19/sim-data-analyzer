# Rate Cross-Correlation Analysis

Pairwise population-rate cross-correlations for populations whose names do not contain `frz`,
optionally restricted by `POP_NAMES` and optionally bandpass-filtered before correlation.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/rate_crosscorr.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- Rate cache used: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_rates_dt_0.001.nc`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Cross-correlation cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/crosscorr_cache`
- Cross-correlation cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/crosscorr_cache/rate_xcorr_verify_lli2__npops_3__dt_0p001__t_10_30__lag_m0p5_0p5__bp_8_14__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/verify/rate_xcorr_verify_lli2__bp_8_14`

## Parameters

```json
{
  "ANALYSIS_LABEL": "rate_xcorr_verify_lli2",
  "T_LIMITS": [
    10.0,
    30.0
  ],
  "RATE_DT": 0.001,
  "LAG_WINDOW": [
    -0.5,
    0.5
  ],
  "POP_NAMES": [
    "IT3",
    "PV3",
    "SOM3"
  ],
  "DO_PLOT": false,
  "DO_PLOT_MATRICES": false,
  "DO_PLOT_LLI_MATRICES": true,
  "FILTER_FBAND": [
    8.0,
    14.0
  ],
  "FILTER_ORDER": 3,
  "CSV_ROUND_DIGITS": 3,
  "MATRIX_THRESHOLD": 0.5,
  "PLOT_AMP_THRESHOLD": 0.08,
  "LLI_WINDOW": [
    -0.02,
    0.02
  ],
  "LLI_EPS": 1e-12,
  "pair_enumeration": "self pairs plus unordered cross-pop pairs in filtered pop order",
  "correlation_views": [
    "raw_over_N",
    "demeaned_over_N",
    "demeaned_normalized"
  ],
  "summary_metric": "largest-absolute normalized mean-subtracted cross-correlation peak (amplitude and lag)",
  "lli_metrics": [
    "bounded_diff_over_abs_sum_from_demeaned_corr",
    "signed_lead_minus_lag_area_from_demeaned_corr"
  ]
}
```

## Output Naming

- PNG naming convention: `<pop_i>__<pop_j>.png`
- Peak-amplitude CSV: `rate_xcorr_verify_lli2__amp.csv`
- Peak-lag CSV: `rate_xcorr_verify_lli2__lag.csv`
- LLI bounded CSV: `rate_xcorr_verify_lli2__lli_bounded.csv`
- LLI area-diff CSV: `rate_xcorr_verify_lli2__lli_area_diff.csv`
- LLI matrix PNG: `lli_matrices.png`
- Pair-PNG subfolder: not generated
- Matrix-summary PNGs: not generated
- CSV peak metrics come from the normalized, mean-subtracted cross-correlation peak
- Positive LLI means the row population leads the column population
- LLI bounded = `(A_lead - A_lag) / (|A_lead| + |A_lag| + eps)` from demeaned `/N` correlograms
- LLI area diff = `A_lead - A_lag` from demeaned `/N` correlograms
- LLI is derived from cached correlograms using only the short `LLI_WINDOW`, not the full `LAG_WINDOW`

## Populations

- Included populations: IT3, PV3, SOM3
- Number of analyzed pairs: 6
- Plotting enabled: False
- Number of pair PNGs written: 0
- Matrix plotting enabled: False
- LLI matrix plotting enabled: True
- Matrix threshold: 0.5 (masked view uses amplitude hatching and white lag cells for smaller |amplitude| values)
- Pair plot threshold: 0.08 (pair PNGs use normalized, mean-subtracted peak amplitude gating)
- LLI window: (-0.02, 0.02) (bounded asymmetry over negative vs positive lags)
