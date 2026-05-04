# OEvent Band-Event Analysis

Independent per-channel OEvent analysis over cached LFP/CSD traces with a lightweight result cache.

## Paths

- Script: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/oevent_band_events.py`
- Raw source: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/a1_lfp_30s/data_00000_seed_1000.pkl`
- LFP cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/a1_lfp_30s_0_lfp.nc`
- OEvent config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/analysis/configs/oevent/default.json`
- Copied config: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/streamcheck__csd__alpha__t_5_30__y_500_700__oevcfg_default/oevcfg_default.json`
- Intermediate/cache root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0`
- Result-cache manifest: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/a1_lfp_30s_0/oevent_cache/result/result_manifest__streamcheck__csd__alpha__t_5_30__y_500_700__oevcfg_default__kind_csd__t_5_30__outz_8__outrel_5__meansub_1__ncyc_3__foct_1p5__fsc_0__chs_n3_59ecb145bb__d_d07ec8cb709c__v1.json`
- Results folder: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/results/a1_lfp_30s_0/oevent/streamcheck__csd__alpha__t_5_30__y_500_700__oevcfg_default`

## Parameters

```json
{
  "OEVENT_CFG_NAME": "default",
  "EXP_LABEL": "streamcheck",
  "SIGNAL_KIND": "csd",
  "T_LIMITS": [
    5,
    30
  ],
  "CHANNEL_MODE": "multi",
  "Y": null,
  "Y_VALUES": null,
  "Y_RANGE": [
    500.0,
    700.0
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
    2,
    40
  ],
  "PLOT_XLIM": null,
  "MAKE_PER_CHANNEL_OVERVIEW_PLOTS": false,
  "MAKE_SPECTROGRAM_PLOTS": false,
  "MAKE_STACKED_PLOT": true,
  "STACK_TRACE_AMP_SCALE": 0.3,
  "SPECT_EVENT_COLOR": "red"
}
```

## Channels

- Resolved channel depths: 500, 600, 700
- Number of channels: 3

## Cache Status

- Result manifest hit: True
- Channel result cache hits: 3 / 3
- Spectrogram cache hits: reused via lightweight result cache
- Spectrogram cache files: reused via lightweight result cache
