# Abstraction Layers

## Shared Layer Stack

| Layer | What lives here | Repos that occupy it |
| --- | --- | --- |
| Raw simulation source | NetPyNE result pickles and cfg JSON files | A1, batch_osc_analyzer, sim_res_analyzer, model_tuner |
| Low-level parser utils | spike/LFP/pop/metadata extraction from raw result dicts | A1, sim_res_analyzer, model_tuner, batch_osc fragment |
| Low-level processing utils | rates, CVs, spectra, xarray transforms | A1, batch_osc_analyzer, sim_res_analyzer, model_tuner, xr_utils_neuro |
| Artifact/cache layer | stable re-loadable derived artifacts | sim_res_analyzer, model_tuner, batch_osc derived caches, A1 workflow outputs |
| Batch layer | job discovery, cfg parameter extraction, grid/index logic | A1, batch_osc_analyzer, model_tuner |
| Workflow layer | table generation, plots, convenience scripts | A1, batch_osc_analyzer, sim_res_analyzer workflows |
| Consumer layer | tuning, optimization, or external analysis | A1 experiment scripts, model_tuner main/opt code |

## Source Placement

### `A1-OUinp`

- Touches every layer from raw result parsing up to workflow scripts.
- Lowest-level scope is widest here.
- Highest-level scope is also busiest here.
- The workflow layer is split between Python scripts in `analysis/ou_tuning` and notebooks in `exp_configs`, especially for batch post-processing and parameter-space exploration.

### `batch_osc_analyzer`

- Strong in the batch layer and workflow layer.
- Narrow low-level parser.
- Derived caches are analyzer-owned rather than centralized.

### `sim_res_analyzer`

- Strong in parser, xarray transform, and xarray cache layers.
- Weak in reusable batch abstraction.

### `xr_utils_neuro`

- Only occupies the low-level xarray-processing layer.

### `model_tuner.data_proc`

- Strong in parser, artifact/cache, and consumer-facing typed pipeline layers.
- More structured than A1 or sim_res_analyzer, but intentionally narrower at the raw parser surface.

## Interpretation

- The future shared core should probably be assembled from two overlapping centers of gravity:
  the A1/model_tuner low-level NetPyNE parser family
  the sim_res_analyzer/model_tuner artifact and processing family
- The batch layer is less unified than the parser/processing layers and should probably stay adapter-like.
