# Batch Collection Matrix

## Comparison Table

| Layer | Source | Job discovery | Param extraction | Lookup model | Input artifacts | Outputs |
| --- | --- | --- | --- | --- | --- | --- |
| `BatchAnalyzer` | `batch_osc_analyzer` | count `grid_?????_params.json` files | direct `simConfig` keys by `par_names` | list position and slice helpers | cfg JSON + raw data `pkl` | derived rate/spect/LFP caches as pickles |
| `BatchAnalyzer` | `batch_osc_analyzer_new` | glob cfg JSON files | pandas pivot to xarray job index | xarray-shaped job grid | cfg JSON + raw data `pkl` | intended xarray rate cache, current prototype mixed with pickles |
| `BatchResultManager` | `A1-OUinp` | recursive `*_cfg.json` scan | fixed field map from cfg JSON | nearest-neighbor in parameter space | cfg JSON + raw data `pkl` paths | job id lookup and data path lookup |
| `batch_utils.extract_batch_params_to_xr()` | `A1-OUinp` | recursive cfg scan | nested cfg field extraction | exact xarray job grid | cfg JSON | xarray job-id grid |
| `batch_utils.collect_batch_xr_data()` | `A1-OUinp` | iterate over xarray job-id grid | none | exact grid placement | precomputed per-job `.nc` | merged `xarray` |
| `batch_utils.collect_batch_json_data()` | `A1-OUinp` | iterate over xarray job-id grid | nested JSON keys | exact grid placement, optional sparse skip | precomputed per-job `.json` | merged `xarray.Dataset` |
| notebook collector layer | `A1-OUinp/exp_configs` | driven from notebook cells over cfg grids | `batch_utils` plus ad hoc JSON/CSV loading | exact grid placement or experiment-specific post-processing | cfg JSON + result JSON/NetCDF + target CSV | in-memory xarray, plots, and sometimes CSV selections |
| `BatchAnalyzer` | `model_tuner` | count `exp_name_?????_cfg.json` files | nested cfg field extraction with dotted names | exact by file index and `par_names` | cfg JSON + raw data `pkl` | cfg/data path accessors and param lists |
| `BatchMetricGetter1D` | `model_tuner` | delegates to typed batch analyzer | same as analyzer | exact | raw data `pkl` through parser+cache | per-pop rate arrays |

## `batch_osc_analyzer`

- The original analyzer assumes one consistent folder layout and derives cache file names from analysis parameters.
- It is tightly coupled to `grid_?????` naming.
- `get_slice_idx()` and `get_data_slice()` offer a lightweight parameter-slice API on top of `par_vals_lst`.
- The new analyzer tries to move to an xarray-shaped batch index but is not yet storage-consistent.

## `A1-OUinp`

- There are two distinct batch-collection styles.
- `BatchResultManager` is a small OU-grid-specific convenience layer.
- `batch_utils.py` is more general:
  it builds an xarray job-id grid from cfg JSON
  then fills that grid from already-derived `.nc` or `.json` results
- A1 also has workflow scripts that bypass a central collector and compute tables directly from raw `pkl` results.
- In addition, `exp_configs` notebooks form a visible post-processing layer on top of `batch_utils`:
  `proc_batch.ipynb` manually fills xarray datasets from per-job JSON
  `collect_batch_res.ipynb` uses `collect_batch_xr_data()` against `rates_xr` and `vstats_xr`
  `select_rx_points.ipynb` combines `batch_utils`, JSON results, smoothing, and CSV export
  `plot_nullclines.ipynb` uses cfg-grid indexing plus result JSON for parameter-slice analysis

## `model_tuner`

- The base `BatchAnalyzer` is a clean exact-index reader for cfg/data filenames.
- `BatchMetricGetter1D` is the real bridge from batch folders to reusable metrics.
- It parses each raw result through:
  `SimResultFile`
  `SimResultParserNetPyNE`
  `DataProcessor`
  `DataKeeper`
- This makes `model_tuner` the only stack here where batch harvesting is already wired into a typed artifact pipeline.

## Main Differences That Matter

- Exact vs nearest lookup:
  `A1 BatchResultManager` is the only layer here that intentionally does nearest-neighbor lookup in parameter space.
- Raw vs derived collection:
  `batch_osc_analyzer` and `model_tuner` start from raw simulation pickles.
  `A1 batch_utils` is optimized to collect already-derived JSON/NetCDF outputs.
- Batch abstraction depth:
  `batch_osc_analyzer` stops at derived caches.
  `A1` spans from raw jobs to CSV/JSON/NetCDF products.
  `model_tuner` goes from raw jobs to typed cached artifacts to metric arrays.
