# Rate Cross-Correlation Analysis

Pairwise population-rate cross-correlations for populations whose names do not contain `frz`,
optionally restricted by `POP_NAMES` and optionally bandpass-filtered before correlation.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/rate_crosscorr.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- Rate cache used: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_rates_dt_0.001.nc`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Cross-correlation cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/crosscorr_cache`
- Cross-correlation cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/crosscorr_cache/rate_xcorr_allpops__npops_43__dt_0p001__t_10_30__lag_m0p5_0p5__bp_8_14__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/rate_xcorr_allpops__bp_8_14`

## Parameters

```json
{
  "ANALYSIS_LABEL": "rate_xcorr_allpops",
  "T_LIMITS": [
    10.0,
    30.0
  ],
  "RATE_DT": 0.001,
  "LAG_WINDOW": [
    -0.5,
    0.5
  ],
  "POP_NAMES": null,
  "DO_PLOT": false,
  "DO_PLOT_MATRICES": true,
  "FILTER_FBAND": [
    8.0,
    14.0
  ],
  "FILTER_ORDER": 3,
  "CSV_ROUND_DIGITS": 3,
  "MATRIX_THRESHOLD": 0.5,
  "PLOT_AMP_THRESHOLD": 0.08,
  "pair_enumeration": "self pairs plus unordered cross-pop pairs in filtered pop order",
  "correlation_views": [
    "raw_over_N",
    "demeaned_over_N",
    "demeaned_normalized"
  ],
  "summary_metric": "largest-absolute normalized mean-subtracted cross-correlation peak (amplitude and lag)"
}
```

## Output Naming

- PNG naming convention: `<pop_i>__<pop_j>.png`
- Peak-amplitude CSV: `rate_xcorr_allpops__amp.csv`
- Peak-lag CSV: `rate_xcorr_allpops__lag.csv`
- Pair-PNG subfolder: not generated
- Matrix-summary PNGs: `rate_xcorr_allpops__matrices.png`, `rate_xcorr_allpops__matrices__thr_0p5.png`
- CSV metrics come from the normalized, mean-subtracted cross-correlation peak

## Populations

- Included populations: IT2, IT3, ITP4, ITS4, IT5A, IT5B, IT6, CT5A, CT5B, CT6, PT5B, PV2, PV3, PV4, PV5A, PV5B, PV6, SOM2, SOM3, SOM4, SOM5A, SOM5B, SOM6, VIP2, VIP3, VIP4, VIP5A, VIP5B, VIP6, NGF1, NGF2, NGF3, NGF4, NGF5A, NGF5B, NGF6, TC, HTC, TI, IRE, TCM, TIM, IREM
- Number of analyzed pairs: 946
- Plotting enabled: False
- Number of pair PNGs written: 0
- Matrix plotting enabled: True
- Matrix threshold: 0.5 (masked view uses amplitude hatching and white lag cells for smaller |amplitude| values)
- Pair plot threshold: 0.08 (pair PNGs use normalized, mean-subtracted peak amplitude gating)
