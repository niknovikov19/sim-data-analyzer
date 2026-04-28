# Links Between Sources

## Direct In-Repo Usage Links

### `A1-OUinp`

- Many plot/table scripts import `BatchResultManager`, `netpyne_res_parse_utils`, or `data_proc_utils` directly.
- `sim_res_proc_utils.py` sits above the low-level parser and processing helpers.
- `batch_utils.py` links cfg parsing to already-derived `.json` or `.nc` result products.
- `exp_configs` notebooks are part of the same practical analysis graph:
  `proc_batch.ipynb`, `collect_batch_res.ipynb`, `select_rx_points.ipynb`, and `plot_nullclines.ipynb` import `batch_utils` and build notebook-local collectors or visualizations on top of it.
- Some `compare_rates.ipynb` notebooks use `sim_res_proc_utils` directly on raw results, which means the A1 helper layer is consumed from both `analysis/` and `exp_configs/`.

### `batch_osc_analyzer`

- Plot scripts import `BatchAnalyzerOsc`.
- `BatchAnalyzerOsc` uses `common.py`, `lfp_analyzer.py`, and the narrow `sim_res_parser.py`.

### `sim_res_analyzer`

- `misc/` and `workflow/` scripts import `DataProcessor`, `PSDParams`, `SimResultParser`, and `DataKeeper`.
- The internal flow is:
  raw `pkl` -> `SimResultParser` -> xarray artifacts in `DataKeeper` -> `DataProcessor` transforms

### `model_tuner`

- `model_tuner.main` imports `NetSpikesParams`, `NetRatesParams`, `BatchMetricGetter1D`, `DataKeeper`, `SimResultParserNetPyNE`, and `DataProcessor`.
- The internal flow is:
  `SimResultFile` -> parser -> typed cached artifact -> processor -> typed cached artifact -> main/opt code

### `xr_utils_neuro`

- No in-repo consumers were found during this audit.
- It behaves as a standalone extraction rather than part of a larger package graph.

## Cross-Source Conceptual Links

### A1 <-> `model_tuner`

- Strongest low-level parser overlap.
- Strongest low-level rate/CV overlap.
- Best explanation is that `model_tuner.data_proc` narrowed and structured a subset of the A1 low-level surface.

### `sim_res_analyzer` <-> `model_tuner`

- Strong cache/artifact lineage.
- Strong high-level parser/processor lineage, but with different target abstractions:
  xarray artifacts in `sim_res_analyzer`
  typed artifacts plus `DataIndex` in `model_tuner`

### `sim_res_analyzer` <-> `xr_utils_neuro`

- Clear xarray-helper lineage.
- `xr_diff.py` is identical.
- `xr_spect.py` keeps the same problem domain but diverges in implementation and scope.

### `batch_osc_analyzer` <-> others

- Overlap is mostly conceptual rather than import-based.
- It shares the same raw input domain and some LFP/rate math concerns, but it does not sit in the same parser/cache lineage as `sim_res_analyzer` or `model_tuner`.

## What The Links Mean For Unification

- The strongest reusable cross-source links are at the low-level parser and low-level rate/CV layers.
- The weakest cross-source links are at the workflow and batch layers, because those encode experiment layout and artifact conventions.
- The safest future extraction path is to treat batch layers as adapters around a shared parser/processing/artifact core.
