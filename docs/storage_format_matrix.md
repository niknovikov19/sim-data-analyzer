# Storage Format Matrix

## Summary Table

| Source | Raw inputs | Derived artifacts | What gets reloaded later | Authoritative store pattern |
| --- | --- | --- | --- | --- |
| `A1-OUinp/analysis/ou_tuning` | `*_data.pkl`, `*_cfg.json` | `batch_result.csv`, `batch_voltages.csv`, `batch_result.nc`, assorted result JSONs | CSV, JSON, and NetCDF collectors are all used depending on workflow | mixed workflow outputs, not one central cache |
| `batch_osc_analyzer` | `grid_?????_data.pkl`, `grid_?????_params.json` | derived `rate_data*.pkl`, `spect_data*.pkl`, `lfp_data*.pkl` | pickled derived caches | filename-templated derived cache files |
| `sim_res_analyzer` | raw result `pkl` | metadata `json`, xarray `.nc` | `DataKeeper` metadata plus `.nc` | xarray-only cache |
| `xr_utils_neuro` | none | none | none | no storage layer |
| `model_tuner.data_proc` | raw result `pkl` | metadata `json`, derived `pkl`, `json`, `nc` | generalized `DataKeeper` entries | typed artifact store keyed by hashed params |

## `A1-OUinp`

- Raw simulations are still read directly from pickled NetPyNE result dicts.
- Configs are read from per-job JSON files.
- Derived artifacts are workflow-specific rather than centrally normalized:
  `create_batch_res_table.py` writes `analysis/batch_result.csv`
  `create_batch_voltage_table.py` writes `batch_voltages.csv`
  `batch_res_to_xr.py` writes `analysis/batch_result.nc`
  `batch_utils.py` can harvest already-derived `.nc` or `.json` files into xarray structures
- Practical implication:
  A1 has the richest downstream format diversity, but not a single authoritative cache abstraction.

## `batch_osc_analyzer`

- Raw job data comes from `grid_?????_data.pkl`.
- Batch parameters come from `grid_?????_params.json`.
- Derived caches are written as pickled Python structures by the original analyzer.
- `batch_analyzer_new.py` introduces `.nc` file naming and `xr.open_dataset()` but still writes pickles in the main rate-data flow.
- Practical implication:
  storage semantics are stable in the original analyzer and unstable in the new analyzer.

## `sim_res_analyzer`

- Raw results are loaded from pickle in `sim_res_parser.py`.
- The `DataKeeper` stores metadata in JSON and xarray payloads as NetCDF.
- The cache contract is simple:
  each named artifact plus params hash maps to one `.nc` file.
- Practical implication:
  this is the cleanest xarray-only cache layer in the set.

## `model_tuner.data_proc`

- Raw simulation results are accessed through `SimResultFile`.
- `DataKeeper` generalizes artifact storage across `PKL`, `JSON`, and `XR`.
- Metadata is JSON; file path layout is derived from hashed params and `data_name`.
- The intent is clearly broader than `sim_res_analyzer`.
- Practical implication:
  the abstraction is stronger, but the XR read/write branches should be audited before trusting it as the canonical storage layer.

## `xr_utils_neuro`

- No storage layer.
- This repo is purely computational/plotting utility code.

## Recommended Interpretation

- `pkl` is still the authoritative raw simulation source in all stacks that parse NetPyNE output directly.
- `json` means two different things depending on the source:
  config metadata in all repos
  derived result tables or harvested per-job outputs in A1 and model_tuner
- `nc` is the main xarray artifact format in `sim_res_analyzer`, a generalized artifact option in `model_tuner`, and a workflow export target in A1.
