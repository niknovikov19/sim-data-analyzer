# Rate Coherence Analysis

Pairwise population-rate coherence and phase-difference matrices derived from cached CPSD spectra.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/rate_coherence.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- Rate cache used: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_rates_dt_0.001.nc`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Coherence cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache`
- Coherence cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache/rate_coherence_verify2__pops_n2_989521536a__dt_0p001__t_10_30__win_1__ov_0p5__fmax_100__bp_5_30__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/rate_coherence_verify2__bp_5_30`

## Parameters

```json
{
  "ANALYSIS_LABEL": "rate_coherence_verify2",
  "T_LIMITS": [
    10.0,
    30.0
  ],
  "RATE_DT": 0.001,
  "FILTER_FBAND": [
    5.0,
    30.0
  ],
  "FILTER_ORDER": 3,
  "WIN_LEN": 1.0,
  "WIN_OVERLAP": 0.5,
  "FMAX": 100.0,
  "FBAND": [
    8.0,
    14.0
  ],
  "POP_NAMES": [
    "IT3",
    "PV3"
  ],
  "CSV_ROUND_DIGITS": 3,
  "COHERENCE_THRESHOLD": 0.3,
  "pair_enumeration": "self pairs plus unordered cross-pop pairs in filtered pop order",
  "band_summary": "complex mean of coherence over FBAND; magnitude and phase taken from that mean"
}
```

## Output Naming

- Coherence CSV: `rate_coherence_verify2__fband_8_14__coherence.csv`
- Phase CSV: `rate_coherence_verify2__fband_8_14__phase.csv`
- Matrix-summary PNGs: `rate_coherence_verify2__fband_8_14__matrices.png`, `rate_coherence_verify2__fband_8_14__matrices__thr_0p3.png`

## Populations

- Included populations: IT3, PV3
- Number of analyzed pairs: 3
- Matrix plotting enabled: True
- Coherence threshold: 0.3 (masked view hatches weak coherence cells and whitens their phase cells)
