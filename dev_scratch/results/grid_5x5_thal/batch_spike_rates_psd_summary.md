# Batch SpikeData Rates + PSD Extraction

- Batch root: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_src/hpc_remote/grid_5x5_thal`
- Spike cache dir: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/spike_cache__source-pkl__pops-all-fixed-order__t-1-end__abs-s`
- Raw rates cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates__source-spike-data__var-rates__pops-all-fixed-order__dt-5ms__tau-20ms__lazy.nc`
- Public rates cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__lazy.nc`
- Public PSD cache file: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates_psd__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__f-2-80__lazy.nc`
- Source type: `SpikeData`
- Upstream raw source: `pkl`
- Extracted signal: `population rates`
- PSD method: `Welch`
- Population filter: exclude names containing `frz`
- Populations: `4`
- Rate dims: `('rxe', 'rxi', 'pop', 'time')`
- Rate shape: `(5, 5, 4, 800)`
- PSD dims: `('rxe', 'rxi', 'pop', 'freq')`
- PSD shape: `(5, 5, 4, 157)`

## Batch Coordinates

- `rxe`: [1.0, 3750.75, 7500.5, 11250.25, 15000.0]
- `rxi`: [1.0, 375.75, 750.5, 1125.25, 1500.0]
- time range: `1` .. `4.995` s
- freq range: `2` .. `80` Hz

## Top Populations By Mean Rate

- `IRE`: `22.1275`
- `IREM`: `15.346`
- `TI`: `14.359`
- `TIM`: `11.0635`

## Top Populations By Mean PSD Band Power

- `TIM`: `0.0151381` at peak `3.5` Hz
- `TI`: `0.0112033` at peak `4.5` Hz
- `IREM`: `0.00774384` at peak `2` Hz
- `IRE`: `0.00642331` at peak `2` Hz
