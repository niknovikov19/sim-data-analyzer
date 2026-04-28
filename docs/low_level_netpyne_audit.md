# Low-Level NetPyNE Audit

This document focuses on the lowest layers where small behavioral drift can change results without changing the apparent API shape.

## Shared-Core Collection Milestone

The first shared parser collection is now implemented at
`sim_data_analyzer/netpyne_res_parse_utils.py`.
The first xarray adapter collection is now implemented at
`sim_data_analyzer/xr_adapters.py`.
The first low-level processing collection is now implemented at
`sim_data_analyzer/data_proc_utils.py`.
The first low-level spectral collection is now implemented at
`sim_data_analyzer/xr_spect.py`.

This first pass is intentionally a direct collection of the stable A1/model_tuner
overlap, not a semantic cleanup or rollout.

| Status | Scope |
| --- | --- |
| collected now | `get_pop_names`, `get_lfp_coords`, `get_record_times`, `get_lfp`, `get_pop_lfps`, `get_pop_ylim`, `get_layer_borders`, `get_net_params`, `get_pop_params`, `get_pop_cell_gids`, `get_sim_data`, `get_pop_size`, `get_net_size`, `get_sim_duration`, `get_pop_spikes`, `get_net_spikes` |
| collected now, A1-sourced extension | `get_timestep`, `get_pop_voltages`, `get_voltages` |
| collected now, xarray adapter layer | A1-sourced `get_trace_xr`, `get_voltages_xr` plus sim_res_analyzer-sourced `get_lfp_xr`, `get_pop_lfps_xr` in `sim_data_analyzer/xr_adapters.py` |
| collected now, low-level processing core | A1/model_tuner shared `calc_pop_rate`, `calc_net_rates`, `calc_pop_cv`, `calc_net_cvs`, plus A1 `calc_pop_rate_dynamics`, `calc_net_rate_dynamics` in `sim_data_analyzer/data_proc_utils.py` |
| collected now, low-level spectral core | `calc_xr_welch`, `calc_xr_tf`, `calc_xr_cpsd` in `sim_data_analyzer/xr_spect.py`, based on `xr_utils_neuro` with explicit `compute` control and optional `proc_steps` attrs |
| deferred A1-only | `prepare_sim_result` |
| deferred sim_res_analyzer-only | `get_pop_cell_rates`, parser-side `calc_rate_dynamics()` |
| deferred batch_osc-specific | the narrowed `sim_res_parser.py` fragment, including its current `get_pop_names()` drift |

Verification for the collected module now lives in
`sim_data_analyzer/tests/test_netpyne_res_parse_utils.py`.
Verification for the collected xarray adapter module now lives in
`sim_data_analyzer/tests/test_xr_adapters.py`.
Verification for the collected low-level processing module now lives in
`sim_data_analyzer/tests/test_data_proc_utils.py`.
Verification for the collected low-level spectral module now lives in
`sim_data_analyzer/tests/test_xr_spect.py`.

## Files Audited

### Parser family

- `A1-OUinp/analysis/ou_tuning/netpyne_res_parse_utils.py`
- `model_tuner/model_tuner/data_proc/netpyne_res_parse_utils.py`
- `sim_res_analyzer/code/sim_res_parse_utils.py`
- `batch_osc_analyzer/sim_res_parser.py`

### Processing family

- `A1-OUinp/analysis/ou_tuning/data_proc_utils.py`
- `A1-OUinp/analysis/ou_tuning/sim_res_proc_utils.py`
- `model_tuner/model_tuner/data_proc/data_proc_utils.py`
- `model_tuner/model_tuner/data_proc/data_proc_funcs.py`
- `sim_res_analyzer/code/data_proc.py`
- `batch_osc_analyzer/common.py`

## Parser Family Summary

| Function family | A1 | model_tuner | sim_res_analyzer | batch_osc | Notes |
| --- | --- | --- | --- | --- | --- |
| LFP coordinate/time helpers | full | identical subset | identical subset | identical subset | `get_lfp_coords`, `get_record_times`, `get_lfp`, `get_pop_lfps`, `get_pop_ylim`, `get_layer_borders` are identical where present |
| Metadata/pop helpers | full | identical subset | identical subset | absent | `get_net_params`, `get_pop_params`, `get_pop_cell_gids`, `get_sim_data`, `get_sim_duration` are shared exactly across A1/model_tuner/sim_res_analyzer |
| `get_pop_names()` | canonical | identical | identical | divergent | `batch_osc` returns `list(sim_result['net']['pops'].values())` instead of keys |
| `get_pop_size()` | canonical | identical | same logic | absent | `sim_res_analyzer` uses the same body, without the type annotation |
| `get_pop_spikes()` | canonical | identical | divergent | absent | `sim_res_analyzer` returns a bare `ndarray` for combined spikes, not `[ndarray]` |
| `get_net_spikes()` | canonical | identical | absent | absent | only A1/model_tuner expose the net-wide convenience wrapper |
| `get_net_size()` | canonical | identical | absent | absent | only A1/model_tuner expose the net-size helper |
| basic voltage helpers | present | absent | absent | absent | `get_pop_voltages` and `get_voltages` exist only in A1 and are now collected into `sim_data_analyzer` as A1-sourced extensions |
| xarray trace adapters | present | absent | absent | absent | `get_trace_xr` and `get_voltages_xr` exist only in A1 and are now collected into `sim_data_analyzer/xr_adapters.py` |
| xarray LFP adapters | absent | absent | present in parser layer | absent | `get_lfp_xr` and `get_pop_lfps_xr` are now collected into `sim_data_analyzer/xr_adapters.py` from `sim_res_analyzer/code/sim_res_parser.py` private helpers |
| `get_timestep()` | present | absent | absent | absent | A1-only helper, now also collected into `sim_data_analyzer` |
| `prepare_sim_result()` | present | absent | absent | absent | A1-only live-sim to dict conversion helper |
| `get_pop_cell_rates()` | absent | absent | present | absent | sim_res_analyzer-only low-level rate helper |
| parser-side `calc_rate_dynamics()` | absent at parser layer | absent | present | absent | only `sim_res_analyzer/sim_res_parse_utils.py` keeps this as a parser-side helper; A1 has similarly named helpers in the processing layer instead |

## Concrete Drift Notes

### A1 vs `model_tuner` parser lineage

- The shared core functions are byte-identical.
- `model_tuner` is best described as a strict subset of the A1 parser surface rather than a semantic rewrite of that shared subset.
- That byte-identical shared core is now copied directly into
  `sim_data_analyzer/netpyne_res_parse_utils.py`.
- The collected version now intentionally adopts the safer
  `if tmax is None` check from `sim_res_analyzer` inside `get_pop_spikes()`,
  so `tmax=0` is preserved instead of being replaced by the full simulation
  duration.
- `sim_data_analyzer/netpyne_res_parse_utils.py` also now includes
  `get_timestep()`, `get_pop_voltages()`, and `get_voltages()` as small
  A1-sourced extensions that do not pull in `xarray`.
- `get_trace_xr()` and `get_voltages_xr()` are now collected separately in
  `sim_data_analyzer/xr_adapters.py`, because they define an xarray-facing
  representation layer rather than a plain dict/NumPy parser layer.
- `get_lfp_xr()` and `get_pop_lfps_xr()` are now collected in the same adapter
  module from the `sim_res_analyzer` LFP conversion helpers, since they are the
  clearest existing xarray representation functions for raw LFP outputs.
- The missing pieces in `model_tuner` are the voltage/xarray helpers and live-sim conversion helpers that A1 still needs for workflow scripts.

## Processing Collection Notes

- `sim_data_analyzer/data_proc_utils.py` now collects the clearest low-level
  processing overlap first:
  `calc_pop_rate`, `calc_net_rates`, `calc_pop_cv`, and `calc_net_cvs`.
- The collected `calc_pop_rate()` keeps the A1 validation guard that rejects
  multi-train input together with `ncells != 1`.
- The collected rate-dynamics helpers come from A1:
  `calc_pop_rate_dynamics()` and `calc_net_rate_dynamics()`.
- For the combined-single-train, unsmoothed case, the collected
  `calc_pop_rate_dynamics()` also matches
  `sim_res_analyzer/sim_res_parse_utils.py::calc_rate_dynamics()`.

## Spectral Collection Notes

- `sim_data_analyzer/xr_spect.py` now collects the low-level xarray spectral
  helpers from the `xr_utils_neuro` lineage:
  `calc_xr_welch()`, `calc_xr_tf()`, and `calc_xr_cpsd()`.
- The collected spectral helpers make execution behavior explicit:
  `compute=False` keeps deferred/lazy behavior for dask-backed inputs,
  while `compute=True` realizes the result before return.
- The collected spectral helpers also support optional processing metadata via
  a simple JSON-serializable `proc_steps` xarray attr.
- Compatibility checks cover the overlapping older lines:
  `calc_xr_welch()` against `sim_res_analyzer` and `calc_xr_tf()` against A1.

### `sim_res_analyzer` parser drift

- `sim_res_analyzer.get_pop_spikes()` uses `if tmax is None`, which is slightly safer than A1/model_tuner `tmax = tmax or get_sim_duration(...)`.
- In the combined-spike branch, `sim_res_analyzer` returns `np.round(...)`, while A1/model_tuner return `[np.round(...)]`.
- That single-container difference changes downstream expectations:
  A1/model_tuner low-level rate helpers expect a list of spike trains even when the spikes were combined upstream.
- `sim_res_analyzer` does not expose `get_net_spikes()` or `get_net_size()`, because its higher-level parser goes straight to xarray-oriented outputs.

### `batch_osc_analyzer` parser drift

- `batch_osc_analyzer/sim_res_parser.py` keeps only the LFP/pop-layer subset.
- `get_pop_names()` currently returns population dict values instead of names.
- That makes it the least compatible member of the parser family and a likely bug or incomplete refactor.

## Processing Family Summary

| Function family | A1 | model_tuner | sim_res_analyzer | batch_osc | Notes |
| --- | --- | --- | --- | --- | --- |
| `calc_net_rates()` | canonical | identical | indirect via parser/xarray flow | similar math lives elsewhere | A1 and model_tuner bodies are identical |
| `calc_net_cvs()` | canonical | identical | absent | absent | same low-level CV aggregation in A1/model_tuner |
| `calc_pop_rate()` | canonical | near-identical | absent | rate-dynamics handled in analyzer/common | A1 adds a validation guard for per-cell input semantics |
| `calc_pop_cv()` | canonical | same math | absent | absent | A1 adds documentation, not different math |
| `calc_pop_rate_dynamics()` | present | removed/commented | absent | absent | A1 low-level per-pop binned-rate helper with optional smoothing and epoch folding |
| `calc_net_rate_dynamics()` | present | removed/commented | absent | absent | A1 low-level net wrapper over `calc_pop_rate_dynamics()` |
| `calc_rate_dynamics()` | present in `sim_res_proc_utils.py` | absent | present in `sim_res_parse_utils.py` | absent | same name, but not the same layer: A1 uses a higher-level wrapper over parsed spikes; sim_res_analyzer uses a lower-level parser utility |
| A1-only convenience wrappers | `calc_rates_and_cvs()`, `calc_trace_stats()`, `calc_v_stats()` | absent | absent | absent | workflow-facing helpers that sit above the low-level parser and processing utilities |
| high-level extraction | direct utility functions | typed pure functions | xarray parser methods | batch analyzer methods | three distinct architectural directions |

## A1 vs `model_tuner` Low-Level Processing

- `calc_net_rates()` is identical.
- `calc_net_cvs()` is identical.
- `calc_pop_rate()` is the same computation, but A1 adds a guard:
  if `pop_spikes` has more than one entry, `ncells` must be `1`.
- `calc_pop_cv()` differs only in comments/docstring.
- A1 still owns `calc_pop_rate_dynamics()` and `calc_net_rate_dynamics()`, including optional Gaussian smoothing and epoch folding.
- A1 also keeps a workflow-oriented helper layer in `sim_res_proc_utils.py`:
  `calc_rates_and_cvs()`, `calc_trace_stats()`, `calc_v_stats()`, and `calc_rate_dynamics()`.
- `model_tuner` has these helpers commented out, which signals a deliberate narrowing of the low-level scope.

## High-Level Layer Split

### A1

- `sim_res_proc_utils.py` operates directly on a live NetPyNE sim object or a stored result dict.
- It bundles exact helper functions:
  `calc_rates_and_cvs()`
  `calc_trace_stats()`
  `calc_v_stats()`
  `calc_rate_dynamics()`
- Its `calc_rate_dynamics()` is not the same function as `sim_res_analyzer/sim_res_parse_utils.py::calc_rate_dynamics()`:
  A1 wraps low-level spike extraction plus `data_proc_utils.calc_net_rate_dynamics()`,
  while sim_res_analyzer exposes a lower-level time-binned helper directly from the parser utility layer.

### `sim_res_analyzer`

- `SimResultParser` is still high-level, but it maps raw pickles directly to xarray artifacts such as `LFP`, `LFPpop`, and `rpop_dyn`.
- `DataProcessor` then operates on those xarray artifacts using bipolar/CSD/PSD transforms.

### `model_tuner`

- `data_proc_funcs.extract_net_spikes()` and `calc_net_rates()` operate on typed params and typed data objects.
- `SimResultParserNetPyNE` and `DataProcessor` work by producing `DataIndex` handles into a generalized `DataKeeper`.
- The main architectural change is typed artifact pipelines rather than raw-xarray pipelines.

## Suspicious Or Likely Regression Candidates

- `batch_osc_analyzer/sim_res_parser.py:6-8`
  `get_pop_names()` returns dict values instead of keys.
- `model_tuner/model_tuner/data_proc/data_keeper.py:79-102`
  the XR branches in `_load_formatted_data()` and `_save_formatted_data()` appear reversed.
- `batch_osc_analyzer/batch_analyzer_new.py:133-180`
  the rate cache is named `.nc`, opened with `xr.open_dataset()`, but later loaded/saved via pickle.
- `batch_osc_analyzer/batch_analyzer_new.py:150-154`
  the warm-up call to `_calc_job_rate_data()` passes `pops_incl=pop_names[0]`, which means a single string rather than a full population list.

## Practical Takeaway

- If the future shared core begins at the lowest parser level, use the A1/model_tuner common subset as the reference implementation.
- If it begins at the artifact/cache layer, treat `sim_res_analyzer` and `model_tuner` as the main lineage, but audit the XR branches in `model_tuner` first.
- `batch_osc_analyzer` should be treated as an older sibling with reusable ideas, not as the canonical low-level implementation.
