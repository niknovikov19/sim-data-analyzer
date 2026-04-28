# `sim_data_analyzer` Audit

This folder is a documentation-first audit of the result-processing code that is currently split across:

- `A1-OUinp/analysis`
- `batch_osc_analyzer`
- `sim_res_analyzer/code`
- `xr_utils_neuro`
- `model_tuner/model_tuner/data_proc`

The snapshot below reflects the checked-out workspace on `2026-04-24`.

## Collection Milestone

The first shared parser collection now exists at `sim_data_analyzer/netpyne_res_parse_utils.py`.
The first xarray adapter collection now exists at `sim_data_analyzer/xr_adapters.py`.
The first low-level processing collection now exists at `sim_data_analyzer/data_proc_utils.py`.
The first low-level spectral collection now exists at `sim_data_analyzer/xr_spect.py`.

This milestone is intentionally narrow:

| Status | Scope |
| --- | --- |
| collected now | A1/model_tuner shared parser core: `get_pop_names`, `get_lfp_coords`, `get_record_times`, `get_lfp`, `get_pop_lfps`, `get_pop_ylim`, `get_layer_borders`, `get_net_params`, `get_pop_params`, `get_pop_cell_gids`, `get_sim_data`, `get_pop_size`, `get_net_size`, `get_sim_duration`, `get_pop_spikes`, `get_net_spikes` |
| collected now, A1-sourced extension | `get_timestep`, `get_pop_voltages`, `get_voltages` |
| collected now, xarray adapter layer | A1-sourced `get_trace_xr`, `get_voltages_xr` plus sim_res_analyzer-sourced `get_lfp_xr`, `get_pop_lfps_xr` in `sim_data_analyzer/xr_adapters.py` |
| collected now, low-level processing core | A1/model_tuner shared `calc_pop_rate`, `calc_net_rates`, `calc_pop_cv`, `calc_net_cvs`, plus A1 `calc_pop_rate_dynamics`, `calc_net_rate_dynamics` in `sim_data_analyzer/data_proc_utils.py` |
| collected now, low-level spectral core | `calc_xr_welch`, `calc_xr_tf`, `calc_xr_cpsd` in `sim_data_analyzer/xr_spect.py`, based on `xr_utils_neuro` with explicit `compute` and optional `proc_steps` attrs |
| deferred A1-only | `prepare_sim_result` |
| deferred sim_res_analyzer-only | `get_pop_cell_rates`, parser-side `calc_rate_dynamics()` |
| deferred batch_osc-specific | the narrowed `sim_res_parser.py` fragment and its current semantic drift |

The collected parser module started as a direct copy of the stable A1/model_tuner overlap and now also includes the small A1-only voltage/timestep helpers that do not add extra heavy dependencies. The xarray-facing trace helpers are collected separately in `sim_data_analyzer/xr_adapters.py`. Old repos and old imports are intentionally untouched in this phase.

| Source | Current branch | Identity | Dominant artifacts | Batch style |
| --- | --- | --- | --- | --- |
| `A1-OUinp/analysis` | `netstim-bkg-opt` | Experiment-heavy analysis and workflow layer | raw `pkl`, cfg `json`, derived `csv`/`json`/`nc` | direct folder scans plus xarray/JSON collectors |
| `batch_osc_analyzer` | `main` | Older batch-first analyzer for rates/spectra/LFP | raw `pkl`, cfg `json`, derived `pkl` and one unfinished `.nc` path | filename-templated batch analyzer classes |
| `sim_res_analyzer/code` | `main` | Intermediate generic xarray processing stack | raw `pkl`, metadata `json`, derived `nc` | no dedicated batch layer in-core |
| `xr_utils_neuro` | `main` | Small xarray utility extraction | none | none |
| `model_tuner/model_tuner/data_proc` | `dev` | Most structured typed artifact pipeline | raw `pkl`, metadata `json`, derived `pkl`/`json`/`nc` | typed batch analyzer plus metric getter |

## High-Level Findings

- The cleanest low-level parser lineage is `A1 -> model_tuner`: the shared NetPyNE parser functions are byte-identical, and `model_tuner` is mostly a subset plus a higher-level typed wrapper.
- That shared parser core has now been collected into `sim_data_analyzer/netpyne_res_parse_utils.py`. One intentional safety fix has already been adopted there: `get_pop_spikes()` now uses `if tmax is None` instead of `tmax = tmax or ...`.
- `sim_res_analyzer` shares a large core with the same parser lineage, but it diverges in small behavioral ways that matter analytically, especially around combined spike return shape and explicit `tmax is None` handling.
- `batch_osc_analyzer` is not the direct descendant of the richer NetPyNE parser family. Its `sim_res_parser.py` is a narrower LFP-oriented fragment, and `get_pop_names()` currently returns population dict values rather than names.
- `A1` keeps the widest low-level surface: voltages, trace-to-xarray helpers, `prepare_sim_result()`, CSV builders, xarray builders, and JSON/xarray batch collectors.
- `get_trace_xr()` and `get_voltages_xr()` now sit in a separate collected adapter module rather than the parser-core module, which keeps the parser/xarray boundary explicit.
- The same adapter module now also contains `get_lfp_xr()` and `get_pop_lfps_xr()`, copied from the `sim_res_analyzer` LFP conversion helpers.
- The low-level processing core is now collected too: shared rate/CV math comes from the A1/model_tuner overlap, while rate-dynamics comes from the A1 low-level helpers.
- The low-level spectral core is now collected too: Welch, time-frequency, and cross-PSD helpers come from `xr_utils_neuro`, with explicit immediate/deferred execution control and optional processing metadata stored in xarray attrs.
- `A1` also has an additional notebook-driven analysis layer under `exp_configs/` that was easy to miss on a script-only pass. `proc_batch.ipynb` and similar notebooks consume `analysis.ou_tuning.batch_utils` and derived per-job outputs directly.
- `sim_res_analyzer` and `model_tuner` are the main cache-layer lineage, but `model_tuner` currently appears to have the NetCDF load/save branches reversed inside `data_keeper.py`.
- `batch_osc_analyzer_new.py` looks like an unfinished prototype: it names rate cache files as `.nc`, opens datasets from that path, but later reads/writes pickled Python lists to the same logical artifact.

## Documents

- [Source Catalog](./source_catalog.md)
- [Low-Level NetPyNE Audit](./low_level_netpyne_audit.md)
- [Storage Format Matrix](./storage_format_matrix.md)
- [Batch Collection Matrix](./batch_collection_matrix.md)
- [Interface Matrix](./interface_matrix.md)
- [Duplication And Lineage](./duplication_lineage.md)
- [Abstraction Layers](./abstraction_layers.md)
- [Links Between Sources](./links_between_sources.md)
- [Tooling And Commands](./tooling.md)

## Verification

- The collected parser core is covered by `sim_data_analyzer/tests/test_netpyne_res_parse_utils.py`.
- The collected xarray adapters are covered by `sim_data_analyzer/tests/test_xr_adapters.py`.
- The collected low-level processing core is covered by `sim_data_analyzer/tests/test_data_proc_utils.py`.
- The collected low-level spectral core is covered by `sim_data_analyzer/tests/test_xr_spect.py`.
- The tests compare the collected module directly against:
  `A1-OUinp/analysis/ou_tuning/netpyne_res_parse_utils.py`
  `model_tuner/model_tuner/data_proc/netpyne_res_parse_utils.py`
- The comparison checks function presence, signature equality, representative output equality, and the preserved list-wrapped combined-spike shape.
- The parser tests also pin the newer explicit-`None` `tmax` behavior so `tmax=0` is not silently replaced by the full simulation duration.
- The adapter tests compare `sim_data_analyzer/xr_adapters.py` directly against the A1 xarray trace helpers and the `sim_res_analyzer` LFP conversion helpers, including the `None` return for empty populations.

## Diagrams

### `pyreverse` UML

- [Model Tuner Classes](./diagrams/classes_model_tuner_data_proc.png)
- [Model Tuner Packages](./diagrams/packages_model_tuner_data_proc.png)
- [Sim Res Analyzer Classes](./diagrams/classes_sim_res_analyzer_core.png)
- [Sim Res Analyzer Packages](./diagrams/packages_sim_res_analyzer_core.png)
- [Batch Osc Classes](./diagrams/classes_batch_osc_core.png)
- [Batch Osc Packages](./diagrams/packages_batch_osc_core.png)

### Custom Graphviz Diagrams

- [Source Lineage](./diagrams/source_lineage.png)
- [Abstraction Layers](./diagrams/abstraction_layers.png)
- [A1 OU-Tuning Workflow](./diagrams/a1_ou_tuning_workflow.png)
- [Storage Formats](./diagrams/storage_formats.png)
- [Batch Collection](./diagrams/batch_collection.png)

### DOT Sources

- [Source Lineage DOT](./diagrams/source_lineage.dot)
- [Abstraction Layers DOT](./diagrams/abstraction_layers.dot)
- [A1 OU-Tuning Workflow DOT](./diagrams/a1_ou_tuning_workflow.dot)
- [Storage Formats DOT](./diagrams/storage_formats.dot)
- [Batch Collection DOT](./diagrams/batch_collection.dot)
