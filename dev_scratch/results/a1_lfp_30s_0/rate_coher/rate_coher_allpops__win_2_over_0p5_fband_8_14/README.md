# Rate Coherence Analysis

Pairwise population-rate coherence and phase-difference matrices derived from cached CPSD spectra.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/rate_coherence.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- Rate cache used: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_rates_dt_0.001.nc`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Coherence cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache`
- Coherence cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/coherence_cache/rate_coher_allpops__pops_n43_fa08c11cde__dt_0p001__t_10_30__win_2__ov_0p5__fmax_100__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/rate_coher/rate_coher_allpops__win_2_over_0p5_fband_8_14`

## Parameters

```json
{
  "ANALYSIS_LABEL": "rate_coher_allpops",
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
  "POP_NAMES": null,
  "CSV_ROUND_DIGITS": 3,
  "COHERENCE_THRESHOLD": 0.5,
  "pair_enumeration": "self pairs plus unordered cross-pop pairs in filtered pop order",
  "band_summary": "complex mean of coherence over FBAND; magnitude and phase taken from that mean"
}
```

## Output Naming

- Coherence CSV: `coherence.csv`
- Phase CSV: `phase.csv`
- Matrix-summary PNGs: `matrices.png`, `matrices__thr_0p5.png`

## Populations

- Included populations: IT2, IT3, ITP4, ITS4, IT5A, IT5B, IT6, CT5A, CT5B, CT6, PT5B, PV2, PV3, PV4, PV5A, PV5B, PV6, SOM2, SOM3, SOM4, SOM5A, SOM5B, SOM6, VIP2, VIP3, VIP4, VIP5A, VIP5B, VIP6, NGF1, NGF2, NGF3, NGF4, NGF5A, NGF5B, NGF6, TC, HTC, TI, IRE, TCM, TIM, IREM
- Number of analyzed pairs: 946
- Matrix plotting enabled: True
- Coherence threshold: 0.5 (masked view hatches weak coherence cells and whitens their phase cells)
