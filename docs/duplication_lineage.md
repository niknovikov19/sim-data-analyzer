# Duplication And Lineage

## Status Labels

- `identical`: byte-identical or function-body-identical in the shared scope
- `near-identical`: same purpose and mostly same body, with small API or contract drift
- `descendant`: clear structural evolution into a richer or narrower abstraction
- `source-specific`: not really shared, even if the problem domain overlaps

## Parser Family

| Family | Files | Status | Notes |
| --- | --- | --- | --- |
| collected shared parser core | `sim_data_analyzer/netpyne_res_parse_utils.py` + A1 + model_tuner | `identical` | first consolidation milestone; direct copy of the stable shared subset, no rollout into old repos yet |
| collected A1 parser extensions | `sim_data_analyzer/netpyne_res_parse_utils.py` + A1 | `identical` | `get_timestep`, `get_pop_voltages`, and `get_voltages` are now collected too; they are not shared with model_tuner |
| collected A1 xarray adapters | `sim_data_analyzer/xr_adapters.py` + A1 | `identical` | `get_trace_xr` and `get_voltages_xr` are collected as a separate representation layer instead of being mixed into the parser-core module |
| collected sim_res_analyzer LFP xarray adapters | `sim_data_analyzer/xr_adapters.py` + `sim_res_analyzer/code/sim_res_parser.py` | `identical` | `get_lfp_xr` and `get_pop_lfps_xr` come from the private `_sim_res_to_xr_LFP` and `_sim_res_to_xr_pop_LFPs` helpers |
| LFP/pop helper core | A1 + model_tuner + sim_res_analyzer + batch_osc | `identical` where present | `get_lfp*`, `get_record_times`, `get_pop_ylim`, `get_layer_borders` are shared exactly |
| Metadata/pop helper core | A1 + model_tuner + sim_res_analyzer | `identical` | `get_net_params`, `get_pop_params`, `get_pop_cell_gids`, `get_sim_data`, `get_sim_duration` |
| Net-wide spike helpers | A1 + model_tuner | `identical` | `get_net_spikes`, `get_net_size` survive only in this branch |
| Voltage/xarray helpers | A1 only | `source-specific` | needed by A1 workflow scripts; not present in the other stacks |
| Sim result parser fragment | batch_osc | `descendant` but narrowed and drifted | keeps only the LFP/pop subset and changes `get_pop_names()` semantics |

## Processing Family

| Family | Files | Status | Notes |
| --- | --- | --- | --- |
| `calc_net_rates`, `calc_net_cvs` | A1 `data_proc_utils.py` + model_tuner `data_proc_utils.py` | `identical` | direct low-level carry-over |
| collected processing core | `sim_data_analyzer/data_proc_utils.py` + A1 + model_tuner | `identical` / `near-identical` | shared rate/CV core collected; `calc_pop_rate` keeps the A1 validation guard and the rate-dynamics helpers come from A1 |
| `calc_pop_rate` | A1 + model_tuner | `near-identical` | A1 adds a useful validation guard and clearer contract |
| `calc_pop_cv` | A1 + model_tuner | `near-identical` | same math, doc/comment drift only |
| rate-dynamics helpers | A1 + sim_res_analyzer | `same purpose different interface` | A1 keeps `calc_pop_rate_dynamics()`, `calc_net_rate_dynamics()`, and a higher-level `sim_res_proc_utils.calc_rate_dynamics()` wrapper; sim_res_analyzer keeps a lower-level parser-side `calc_rate_dynamics()` helper |
| high-level extraction layer | A1 `sim_res_proc_utils.py` + model_tuner `data_proc_funcs.py` | `descendant` | A1 is workflow-oriented; model_tuner is typed-artifact-oriented |
| A1-only convenience wrappers | `calc_rates_and_cvs`, `calc_trace_stats`, `calc_v_stats` | `source-specific` | these do not have direct peers in the other stacks |

## Cache Layer

| Family | Files | Status | Notes |
| --- | --- | --- | --- |
| xarray-only cache | `sim_res_analyzer/code/data_keeper.py` | `canonical within xarray-only line` | simple and readable |
| generalized cache | `model_tuner/model_tuner/data_proc/data_keeper.py` | `descendant` | broader format coverage, hashed path layout, likely XR branch bug |
| ad hoc workflow outputs | A1 scripts | `source-specific` | CSV/JSON/NC outputs depend on script purpose |
| filename-based derived caches | batch_osc analyzers | `source-specific` | derived caches are coupled to analyzer methods |

## Batch Layer

| Family | Files | Status | Notes |
| --- | --- | --- | --- |
| original batch analyzer | `batch_osc_analyzer/batch_analyzer.py` | `source-specific` | slice-based batch analysis over raw results |
| new batch analyzer prototype | `batch_osc_analyzer/batch_analyzer_new.py` | `descendant` but incomplete | tries to move toward xarray job grids |
| simple OU job lookup | `A1 BatchResultManager` | `source-specific` | nearest-neighbor lookup for OU parameter pairs |
| generalized A1 harvesters | `A1 batch_utils.py` | `source-specific` | derived-artifact collectors over xarray job grids |
| typed batch analyzer | `model_tuner netpyne_batch_analyzer.py` | `descendant` | clean exact-index batch reader |
| typed metric getter | `model_tuner batch_metric_getter.py` | `descendant` | batch-to-metric adapter on top of parser/cache pipeline |

## Xarray Helper Family

| Family | Files | Status | Notes |
| --- | --- | --- | --- |
| `xr_diff.py` | A1 `analysis/xr_proc`, sim_res_analyzer `code/xr_proc`, `xr_utils_neuro` | `identical` | same SHA1 across all three locations |
| collected y-diff core | `sim_data_analyzer/xr_diff.py` + old `xr_diff.py` copies | `descendant` | collected from the identical low-level xarray diff helper set; adds explicit `compute` control, attr preservation, and optional `proc_steps` attrs |
| collected spectral core | `sim_data_analyzer/xr_spect.py` + `xr_utils_neuro/xr_spect.py` | `identical` / `descendant` | collected from the richest low-level spectral helper set; adds explicit `compute` control and optional `proc_steps` attrs |
| `xr_spect.py` | A1, sim_res_analyzer, xr_utils_neuro | `same purpose different behavior` | all three diverged |
| `xr_proc.py` | A1 + sim_res_analyzer | `near-identical` | import surface differs because A1 still exposes `calc_xr_tf` |

## Short Lineage Read

- A1 and model_tuner share the strongest low-level parser/processing lineage.
- sim_res_analyzer shares a strong subset with that lineage, but diverges at the artifact and xarray-processing layers.
- xr_utils_neuro is an extraction of the xarray helper line, not of the full analysis stacks.
- batch_osc_analyzer is best treated as an older sibling branch with selective overlap, not a straight ancestor of the typed pipeline.
