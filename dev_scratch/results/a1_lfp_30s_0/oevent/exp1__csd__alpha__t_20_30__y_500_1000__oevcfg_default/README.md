# OEvent Band-Event Analysis

Independent per-channel OEvent analysis over cached LFP/CSD traces with a lightweight result cache.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/oevent_band_events.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- LFP cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_lfp.nc`
- OEvent config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/configs/oevent/default.json`
- Copied config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/exp1__csd__alpha__t_20_30__y_500_1000__oevcfg_default/oevcfg_default.json`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Lightweight result cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/oevent_cache/result/result__exp1__csd__alpha__t_20_30__y_500_1000__oevcfg_default__kind_csd__t_20_30__outz_8__outrel_5__meansub_1__ncyc_3__foct_1p5__fsc_0__chs_n6_ffe92a2aca__d_95ce7bc900e8__v1.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/exp1__csd__alpha__t_20_30__y_500_1000__oevcfg_default`

## Parameters

```json
{
  "OEVENT_CFG_NAME": "default",
  "EXP_LABEL": "exp1",
  "SIGNAL_KIND": "csd",
  "T_LIMITS": [
    20.0,
    30.0
  ],
  "CHANNEL_MODE": "multi",
  "Y": null,
  "Y_VALUES": null,
  "Y_RANGE": [
    500.0,
    1000.0
  ],
  "Y_STEP": null,
  "BANDS_OF_INTEREST": [
    "alpha"
  ],
  "BAND_OVERRIDES": {
    "alpha": [
      7.0,
      15.0
    ]
  },
  "CSV_ROUND_DIGITS": 3,
  "PLOT_FILTER_FBAND": null,
  "PLOT_XLIM": null,
  "SPECT_EVENT_COLOR": "red"
}
```

## Channels

- Resolved channel depths: 500, 600, 700, 800, 900, 1000
- Number of channels: 6

## Cache Status

- Lightweight result cache hit: True
- Spectrogram cache hits: reused via lightweight result cache
- Spectrogram cache files: reused via lightweight result cache
