# Rate Coherence Analysis

Pairwise population-rate coherence and phase-difference matrices derived from cached CPSD spectra.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/rate_coherence.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- Rate cache used: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_rates_dt_0.001.nc`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Coherence cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache`
- Coherence cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache/rate_coher_verify_labels2__pops_n18_ac59217772__dt_0p001__t_10_30__win_2__ov_0p5__fmax_100__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/verify/rate_coher_verify_labels2__win_2_over_0p5_fband_8_14`

## Parameters

```json
{
  "ANALYSIS_LABEL": "rate_coher_verify_labels2",
  "T_LIMITS": [
    10.0,
    30.0
  ],
  "RATE_DT": 0.001,
  "WIN_LEN": 2,
  "WIN_OVERLAP": 0.5,
  "FMAX": 100,
  "FBAND": [
    8,
    14
  ],
  "POP_NAMES": [
    "IT3",
    "IT2",
    "ITP4",
    "ITS4",
    "PT5B",
    "VIP2",
    "VIP3",
    "IT5A",
    "NGF2",
    "IT5B",
    "PV5A",
    "SOM4",
    "PV5B",
    "SOM3",
    "SOM5A",
    "PV2",
    "PV4",
    "PV3"
  ],
  "DO_PLOT_VECTORS": true,
  "BASIS_POP": "IT3",
  "VECTOR_COLOR_SCHEME": "cell_type",
  "CSV_ROUND_DIGITS": 3,
  "COHERENCE_THRESHOLD": 0.5,
  "pair_enumeration": "self pairs plus unordered cross-pop pairs in filtered pop order",
  "band_summary": "complex mean of coherence over FBAND; magnitude and phase taken from that mean"
}
```

## Output Naming

- Coherence CSV: `coherence.csv`
- Phase CSV: `phase.csv`
- Matrix-summary PNGs: not generated
- Vector PNG: `vectors__basis_IT3__thr_0p5.png`

## Populations

- Included populations: IT3, IT2, ITP4, ITS4, PT5B, VIP2, VIP3, IT5A, NGF2, IT5B, PV5A, SOM4, PV5B, SOM3, SOM5A, PV2, PV4, PV3
- Number of analyzed pairs: 171
- Matrix plotting enabled: False
- Vector plotting enabled: True
- Coherence threshold: 0.5 (masked view hatches weak coherence cells and whitens their phase cells)
- Vector plot reuses the band-averaged complex coherence and the matrix coherence threshold.
