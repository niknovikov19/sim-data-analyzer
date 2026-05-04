# OEvent Band-Event Analysis

Independent per-channel OEvent analysis over cached LFP/CSD traces with a lightweight result cache.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/oevent_band_events.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- LFP cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_lfp.nc`
- OEvent config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/configs/oevent/default.json`
- Copied config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default/oevcfg_default.json`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Result-cache manifest: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/oevent_cache/result/result_manifest__exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default__kind_csd__t_5_30__outz_8__outrel_5__meansub_1__ncyc_3__foct_1p5__fsc_0__chs_n30_430d94c75d__d_5c8d92b5e3d4__v1.json`
- In-burst mask: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/oevent_mask/exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default.nc`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default`

## Parameters

```json
{
  "OEVENT_CFG_NAME": "default",
  "EXP_LABEL": "exp1",
  "SIGNAL_KIND": "csd",
  "T_LIMITS": [
    5,
    30
  ],
  "CHANNEL_MODE": "multi",
  "Y": null,
  "Y_VALUES": null,
  "Y_RANGE": [
    0,
    3000
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
  "PLOT_FILTER_FBAND": [
    4,
    25
  ],
  "PLOT_XLIM": null,
  "STACK_PLOT_T_RANGE": [
    10,
    20
  ],
  "STACK_PLOT_Y_RANGE": [
    2200,
    2800
  ],
  "MAKE_PER_CHANNEL_OVERVIEW_PLOTS": false,
  "MAKE_SPECTROGRAM_PLOTS": false,
  "MAKE_STACKED_PLOT": false,
  "STACK_TRACE_AMP_SCALE": 0.3,
  "SPECT_EVENT_COLOR": "red"
}
```

## Channels

- Resolved channel depths: 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900
- Number of channels: 30

## Cache Status

- Result manifest hit: True
- Channel result cache hits: 30 / 30
- Spectrogram cache hits: reused via lightweight result cache
- Spectrogram cache files: reused via lightweight result cache
