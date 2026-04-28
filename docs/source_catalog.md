# Source Catalog

## Snapshot Table

| Source | Primary path | Role | Low-level focus | Main artifact formats | Tests |
| --- | --- | --- | --- | --- | --- |
| `A1-OUinp` | `A1-OUinp/analysis/ou_tuning` | Experiment-specific analysis, table generation, plotting, and batch harvesting | richest NetPyNE parser surface, voltage helpers, workflow helpers | raw `pkl`, cfg `json`, derived `csv`, derived `json`, derived `nc` | no dedicated test folder in `analysis/ou_tuning` |
| `batch_osc_analyzer` | `batch_osc_analyzer/` | Batch-centric rate/spectrum/LFP analysis scripts | old batch analyzer classes plus LFP/math helpers | raw `pkl`, cfg `json`, derived `pkl`, one unfinished `.nc` path | small `test/` folder |
| `sim_res_analyzer` | `sim_res_analyzer/code` | Generic xarray-oriented parsing and signal processing | compact parser, xarray `DataKeeper`, xarray `DataProcessor` | raw `pkl`, metadata `json`, derived `nc` | `code/tests` |
| `xr_utils_neuro` | `xr_utils_neuro/` | Small utility extraction for xarray math/plotting | xarray diff/spectral helpers | no storage layer | no explicit tests |
| `model_tuner` | `model_tuner/model_tuner/data_proc` | Typed artifact pipeline used by optimization-facing code | parser subset + typed params + typed artifacts + generalized cache | raw `pkl`, metadata `json`, derived `pkl`, derived `json`, derived `nc` | `tests/data_proc` and broader package tests |

## `A1-OUinp/analysis/ou_tuning`

### Identity

- This is the widest and most experiment-aware stack.
- It works directly against NetPyNE result folders and then fans out into CSV tables, JSON result collections, NetCDF/xarray exports, and many plotting scripts.
- It contains both low-level helpers and workflow glue in the same directory.

### Key Modules

| Module | Role | Inputs | Outputs |
| --- | --- | --- | --- |
| `netpyne_res_parse_utils.py` | richest low-level NetPyNE parser surface | NetPyNE result dict or live sim object | spikes, metadata, voltages, `get_trace_xr()`, `get_voltages_xr()` |
| `data_proc_utils.py` | low-level rate/CV and rate-dynamics math | spike trains | `calc_net_rates()`, `calc_net_cvs()`, `calc_pop_rate_dynamics()`, `calc_net_rate_dynamics()` |
| `sim_res_proc_utils.py` | higher-level analysis helpers | live sim or result dict | `calc_rates_and_cvs()`, `calc_trace_stats()`, `calc_v_stats()`, `calc_rate_dynamics()` |
| `BatchResultManager` | simple job-id lookup for OU grids | cfg JSON files | job id, params, data path |
| `batch_utils.py` | generalized xarray/JSON batch harvesters | cfg JSON plus per-job derived files | `xarray` job grids and datasets |
| `create_batch_res_table.py` | batch table builder | cfg JSON + raw `*_data.pkl` | `analysis/batch_result.csv` |
| `create_batch_voltage_table.py` | batch voltage table builder | cfg JSON + raw `*_data.pkl` | `batch_voltages.csv` |
| `batch_res_to_xr.py` | batch xarray builder | cfg JSON + raw `*_data.pkl` | `analysis/batch_result.nc` |

### Notebook Workflow Layer In `exp_configs/`

These notebooks are not in `analysis/ou_tuning`, but they are part of the same effective analysis surface because they import and drive those helpers directly.

| Notebook | Role | Main dependencies | Main artifacts |
| --- | --- | --- | --- |
| `exp_configs/batch_rxbkg_unconn_state1_mech1/proc_batch.ipynb` | notebook batch collector from per-job JSON results into xarray | `batch_utils.extract_batch_params_to_xr`, manual JSON loading | cfg JSON + result JSON -> in-memory `xarray.Dataset` |
| `exp_configs/batch_unconn/fi_tuning/collect_batch_res.ipynb` | notebook collector for per-job NetCDF rate and voltage stats | `batch_utils.extract_batch_params_to_xr`, `batch_utils.collect_batch_xr_data` | cfg JSON + `rates_xr/*.nc` + `vstats_xr/*.nc` |
| `exp_configs/batch_rxbkg_unconn_state1_mech1/select_rx_points.ipynb` | notebook region/point selector on top of batch JSON outputs | `batch_utils.extract_batch_params_to_xr`, manual JSON loading, smoothing/interpolation | cfg JSON + result JSON + output CSV/selection artifacts |
| `exp_configs/batch_rxbkg_state1_mech1/net_newsec_ee_fade_var_rpop/plot_nullclines.ipynb` | notebook analysis of parameter-grid results and frozen-rate cfg values | `batch_utils.extract_batch_params_to_xr`, cfg JSON, result JSON | param grid + target CSV + cfg JSON |
| `exp_configs/.../compare_rates.ipynb` variants | notebook comparison/visualization layer over batch or single-run outputs | `batch_utils` or `sim_res_proc_utils`, CSV/JSON/raw pkl readers | target CSV + result JSON or raw `pkl` |

### Notes

- This stack mixes low-level reusable helpers with workflow scripts that assume a specific experiment layout.
- It is the only source inspected here that still exposes voltage extraction and trace-to-xarray helpers at the low-level parser layer.
- It also has the richest high-level convenience layer around those low-level helpers via `sim_res_proc_utils.py`.
- A1’s effective workflow layer is split across two locations:
  reusable helpers under `analysis/ou_tuning`
  notebook-driven post-processing and exploration under `exp_configs`
- `batch_utils.py` is the most reusable A1 batch component because it can collect already-derived `.nc` or `.json` results into xarray structures.

## `batch_osc_analyzer`

### Identity

- This is an older batch-first analyzer built around raw NetPyNE result pickles and derived per-batch caches.
- The public surface is mostly top-level scripts that import `BatchAnalyzerOsc`.
- The low-level parser is much narrower than the A1/model_tuner/sim_res_analyzer family.

### Key Modules

| Module | Role | Inputs | Outputs |
| --- | --- | --- | --- |
| `batch_analyzer.py` | original batch analyzer | `grid_?????_params.json`, `grid_?????_data.pkl` | derived rate/spect/LFP pickles |
| `batch_analyzer_new.py` | in-progress analyzer rewrite | same raw inputs | intended xarray rate cache, currently mixed with pickle behavior |
| `sim_res_parser.py` | narrow LFP/pop-layer parser | NetPyNE result dict | pop names, LFP coords, LFP traces, layer bounds |
| `common.py` | generic math/helper utilities | rate vectors, JSON/PKL files | correlations, FFT, filters, loaders |
| `lfp_analyzer.py` | LFP-specific analysis class | sim result or LFP traces | LFP-derived views |

### Notes

- The original batch analyzer stores derived artifacts as Python pickles and reloads them by filename convention.
- The new analyzer introduces xarray/dask allocation and `.nc` naming, but it still serializes Python lists with pickle in the main rate-data path.
- The parser surface is intentionally smaller than the other stacks and does not expose net-wide spike extraction helpers.

## `sim_res_analyzer/code`

### Identity

- This is the clearest xarray-first intermediate layer.
- It reads raw simulation pickles and writes derived xarray artifacts through a simple `DataKeeper`.
- The processor layer is about signal transforms on xarray data, not typed spike/rate artifacts.

### Key Modules

| Module | Role | Inputs | Outputs |
| --- | --- | --- | --- |
| `sim_res_parse_utils.py` | low-level parser subset | NetPyNE result dict | LFP helpers, pop helpers, spike extraction, parser-side `calc_rate_dynamics()` |
| `sim_res_parser.py` | high-level parser | raw result `pkl` | `xarray.DataArray` artifacts in `DataKeeper` |
| `data_keeper.py` | xarray-only artifact store | params + `xarray.DataArray` | metadata JSON + `.nc` |
| `data_proc.py` | xarray processor | cached xarray inputs | bipolar/CSD/PSD xarray outputs |
| `xr_proc/` | shared xarray diff/spectral helpers | xarray inputs | xarray outputs |
| `workflow/` | runnable workflows | cached artifacts or raw results | plots and derived workflow outputs |

### Notes

- The parser is more compact than A1/model_tuner and stops at xarray-ready outputs rather than preserving typed spike/rate objects.
- It is the main ancestor of the `xr_proc` helpers and the simpler `DataKeeper` interface.

## `xr_utils_neuro`

### Identity

- This repo is a tiny extraction of xarray helpers, not a full analysis stack.
- It does not own parsing, storage, or batch collection.

### Key Modules

| Module | Role |
| --- | --- |
| `xr_diff.py` | xarray finite-difference helper |
| `xr_spect.py` | Welch/CPSD/time-frequency helpers |
| `xr_plot.py` | plotting utilities for xarray data |

### Notes

- `xr_diff.py` is byte-identical to the copies in `A1-OUinp/analysis/xr_proc` and `sim_res_analyzer/code/xr_proc`.
- `xr_spect.py` is not identical to either source stack; it is a small utility-focused branch with CPSD and TF helpers.

## `model_tuner/model_tuner/data_proc`

### Identity

- This is the most structured descendant.
- It wraps parsing and processing in typed params, typed data objects, parameter chains, and a generalized artifact store.
- It is already imported by higher-level `model_tuner.main` code.

### Key Modules

| Module | Role | Inputs | Outputs |
| --- | --- | --- | --- |
| `netpyne_res_parse_utils.py` | low-level parser subset | NetPyNE result dict | spikes, metadata, LFP helpers |
| `data_proc_utils.py` | low-level rate/CV math | spike trains | rates and CVs |
| `data_proc_funcs.py` | high-level pure functions | sim results or typed artifacts | `NetSpikesData`, `NetRatesData` |
| `netpyne_result_parser.py` | parser class | `SimResultFile` | typed `DataIndex` entries in `DataKeeper` |
| `data_processor.py` | processor class | cached `DataIndex` inputs | cached derived `DataIndex` outputs |
| `data_keeper.py` | generalized artifact store | typed params + arbitrary artifacts | metadata JSON + `pkl`/`json`/`nc` |
| `netpyne_batch_analyzer.py` | exact batch index reader | cfg JSON + file naming convention | job params, cfg paths, data paths |
| `batch_metric_getter.py` | typed batch-to-metric adapter | batch folder + param names | arrays of per-pop metrics |

### Notes

- The low-level parser is a strict subset of the richer A1 parser surface.
- The high-level layer is no longer xarray-first; it is typed-artifact-first.
- `SimResultFile` adds an abstraction point for local or filesystem-backed results.
