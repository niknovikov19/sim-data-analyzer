# PKL vs SpikeData Equivalence

## Rates

- direct cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates__source-pkl__var-rates__pops-no-frz-fixed-order-windowed__dt-5ms__tau-20ms__lazy.nc`
- spike-data cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__lazy.nc`
- finite-mask identical: `True`
- max abs diff: `0`
- mean abs diff: `0`
- p95 abs diff: `0`
- p99 abs diff: `0`

### Mean-rate difference by population

- `TI`: direct `14.359`, spike `14.359`, |diff| `0`
- `TIM`: direct `11.0635`, spike `11.0635`, |diff| `0`
- `IRE`: direct `22.1275`, spike `22.1275`, |diff| `0`
- `IREM`: direct `15.346`, spike `15.346`, |diff| `0`

## PSD

- direct cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates_psd__source-pkl__var-rates__pops-no-frz-fixed-order-windowed__dt-5ms__tau-20ms__f-2-80__lazy.nc`
- spike-data cache: `/home/nnovikov/repo/sim_data_analyzer/dev_scratch/data_proc/grid_5x5_thal/batch_rates_psd__source-spike-data__var-rates__pops-no-frz-fixed-order__dt-5ms__tau-20ms__f-2-80__lazy.nc`
- finite-mask identical: `True`
- max abs diff: `0`
- mean abs diff: `0`
- p95 abs diff: `0`
- p99 abs diff: `0`

### Mean-PSD difference by population

- `TI`: direct `0.0112033`, spike `0.0112033`, |diff| `0`
- `TIM`: direct `0.0151381`, spike `0.0151381`, |diff| `0`
- `IRE`: direct `0.00642331`, spike `0.00642331`, |diff| `0`
- `IREM`: direct `0.00774384`, spike `0.00774384`, |diff| `0`
