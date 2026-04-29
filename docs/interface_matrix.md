# Interface Matrix

| Layer | A1 | `batch_osc_analyzer` | `sim_res_analyzer` | `model_tuner.data_proc` | `xr_utils_neuro` |
| --- | --- | --- | --- | --- | --- |
| Low-level parser | rich free-function API over result dicts, including voltages and xarray traces | narrow LFP/pop helper fragment | compact free-function API over result dicts | subset of A1 parser family | none |
| Low-level processing | free functions for rates, CVs, rate dynamics | generic math in `common.py`; analyzer-specific batch computations | xarray processor methods plus low-level rate-dynamics helper in parser utils | free functions returning typed data objects | xarray math helpers only |
| High-level parser | workflow helpers in `sim_res_proc_utils.py` | none; batch analyzer owns extraction | `SimResultParser` produces xarray artifacts | `SimResultParserNetPyNE` produces typed cached artifacts | none |
| Cache/artifact store | no single central store | derived-cache files by naming convention | `DataKeeper(storage_dir, metadata_file)` for xarray `.nc` only | generalized `DataKeeper` for `pkl`/`json`/`nc` | none |
| Batch layer | `BatchResultManager`, `batch_utils`, direct workflow scripts | `BatchAnalyzer`, `BatchAnalyzerOsc`, `BatchAnalyzerNew` | no in-core batch collector | `BatchAnalyzer`, `BatchMetricGetter*` | none |
| Workflow/consumer layer | table builders, plot scripts, xarray collectors | plot scripts and small tests | workflows and misc scripts | optimization/main scripts | plotting helpers only |

## Interface Shape Examples

- collected shared parser core:
  `sim_data_analyzer.netpyne_res_parse_utils`
  current surface centers on the stable A1/model_tuner overlap and also includes the small A1-sourced helpers `get_timestep`, `get_pop_voltages`, and `get_voltages`; xarray/live-sim helpers and repo-specific drift remain deferred
- collected xarray adapter layer:
  `sim_data_analyzer.xr_adapters`
  currently contains A1 trace adapters `get_trace_xr`, `get_voltages_xr` plus sim_res_analyzer LFP adapters `get_lfp_xr`, `get_pop_lfps_xr`
- collected low-level processing core:
  `sim_data_analyzer.data_proc_utils`
  currently contains shared rate/CV math plus A1 low-level rate-dynamics helpers
- collected low-level spectral core:
  `sim_data_analyzer.xr_spect`
  currently contains Welch, time-frequency, and cross-PSD xarray helpers with explicit `compute` control and optional `proc_steps` attrs
- collected low-level y-diff core:
  `sim_data_analyzer.xr_diff`
  currently contains shared y-diff, bipolar, and CSD xarray helpers with explicit `compute` control and optional `proc_steps` attrs
- A1 low-level parser:
  `get_net_spikes(sim_result, pop_names=None, combine_cells=True, t0=0, tmax=None, subtract_t0=True, ms=False, ndigits=6)`
- A1 low-level processing:
  `calc_pop_rate_dynamics(pop_spikes, time_range, dt_bin=5e-3, tau_smooth=None, ncells=1, epoch_len=None)`
  `calc_net_rate_dynamics(net_spikes, time_range, dt_bin=5e-3, tau_smooth=None, ncells=None, pop_names=None, epoch_len=None)`
- A1 high-level convenience layer:
  `calc_rates_and_cvs(sim, t_limits=None, nspikes_min=3, per_cell_rates=False)`
  `calc_trace_stats(sim, trace_name, t_limits=None)`
  `calc_v_stats(sim, t_limits=None, med_win=0.1, thresh=-40)`
  `calc_rate_dynamics(sim, t_limits=(None, None), dt_bin=5e-3, tau_smooth=None, pops_used=None)`
- sim_res_analyzer high-level parser:
  `extract_lfp(output_name='LFP')`
  `extract_pop_rates_dyn(rate_par, output_name='rpop_dyn')`
- sim_res_analyzer processor:
  `calc_bipolar(inp_name, inp_params=None, out_name=None, recalc=False)`
  `calc_psd(inp_name, inp_params=None, psd_params=None, out_name=None, recalc=False)`
- model_tuner high-level parser:
  `extract_net_spikes(params, data_name_out='net_spikes', recalc=False) -> DataIndex`
- model_tuner processor:
  `calc_net_rates(data_id_in, step_params, data_name_out='net_rates', recalc=False) -> DataIndex`

## Practical Read

- A1 exposes the broadest free-function surface and the thinnest abstraction boundaries.
- sim_res_analyzer exposes a smaller but clean xarray-first contract.
- model_tuner exposes the narrowest public surface per step, but the strongest artifact identity model.
- batch_osc_analyzer exposes mostly batch-analysis APIs rather than reusable parsing or storage APIs.
