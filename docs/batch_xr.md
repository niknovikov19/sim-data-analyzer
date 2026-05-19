# Batch XR Utilities

This document describes the public batch utilities in [batch_xr.py](/home/nnovikov/repo/sim_data_analyzer/batch_xr.py).

The batch layer is built around a simple idea:

1. Build a job-index xarray from cfg files.
2. Choose a source-specific `collect_batch_*` function.
3. Optionally give it `cache_path` to reuse or create one batch NetCDF file.
4. Use ordinary xarray processing helpers on the collected batch object.

## Public Surface

The main public functions are:

- `extract_batch_params_to_xr(...)`
- `iter_batch_jobs(...)`
- `collect_batch_xr(...)`
- `collect_batch_xr_set(...)`
- `collect_batch_json(...)`
- `collect_batch_rates_from_pkl(...)`
- `collect_batch_lfp_from_pkl(...)`

The public split is semantic:

- raw `pkl` rates and raw `pkl` LFP stay separate functions
- already-formed job outputs get one function per source type

## Batch Index

Use `extract_batch_params_to_xr(...)` to build the parameter grid:

```python
job_idx_xr = extract_batch_params_to_xr(
    dirpath_cfg,
    cfg_param_fields={"rx": "rx", "wx": "wx"},
    fname_cfg_templ="grid_*_cfg.json",
    job_pos_in_fname=-2,
)
```

The result is an xarray whose values are job ids and whose dims are the chosen batch parameters.

## Common Collection Controls

All public `collect_batch_*` functions share the same batch/cache controls:

- `cache_path=None`
- `lazy=False`
- `chunks=None`
- `load=False`
- `open_kwargs=None`
- `skip_missing=True`
- `overwrite=True`

Behavior:

- `cache_path=None`
  build the batch and return it directly
- `cache_path=...`
  reuse the existing batch file if its stored metadata matches the current call
- cache mismatch
  raise `ValueError`
- `lazy=False`
  build in memory first, then optionally save
- `lazy=True`
  stream directly into one chunked NetCDF file
- `lazy=True` without `cache_path`
  raise `ValueError`

`lazy` is an execution/storage detail, not a separate public API family.

## Source-Specific Functions

### Per-job xarray files

Use `collect_batch_xr(...)` when each job already produced one `nc` file.

Useful args:

- `fname_templ`
- `data_type`
- `variable`

Example:

```python
X = collect_batch_xr(
    job_idx_xr,
    dirpath_xr,
    fname_templ="summary_{job:05d}.nc",
)
```

### Per-job xarray file sets

Use `collect_batch_xr_set(...)` when each job produced several related `nc` files.

Useful args:

- `fname_templ`
- `labels`
- `combine`
- `concat_dim`
- `concat_labels`
- `select_label`

Example:

```python
X = collect_batch_xr_set(
    job_idx_xr,
    dirpath_xr,
    fname_templ="job_{job:05d}_{label}.nc",
    labels=["IT2", "PV2"],
    combine="concat",
    concat_dim="pop",
)
```

### Per-job JSON summaries

Use `collect_batch_json(...)` when jobs wrote summary JSON files.

Useful args:

- `var_mappings`
- `dict_dims`
- `extra_coords`

Example:

```python
X = collect_batch_json(
    job_idx_xr,
    dirpath_results,
    var_mappings={"rate": "rates", "cv": "cvs"},
    dict_dims={"pop": ["IT2", "PV2"]},
)
```

### Raw sim-result pickles: rates

Use `collect_batch_rates_from_pkl(...)` when you want to compute rate dynamics from raw NetPyNE result pickles.

Useful args:

- `fname_templ`
- `t_limits`
- `dt_bin`
- `tau_smooth`
- `avg_cells`

Example:

```python
rates_xr = collect_batch_rates_from_pkl(
    job_idx_xr,
    dirpath_data,
    dt_bin=5e-3,
    tau_smooth=20e-3,
)
```

### Raw sim-result pickles: LFP

Use `collect_batch_lfp_from_pkl(...)` when you want to extract LFP traces from raw NetPyNE result pickles.

Useful args:

- `fname_templ`

Example:

```python
lfp_xr = collect_batch_lfp_from_pkl(
    job_idx_xr,
    dirpath_data,
)
```

## One-File Batch Cache

The batch cache is handled inside `batch_xr.py`, not in user scripts.

The intended use is:

```python
rates_xr = collect_batch_rates_from_pkl(
    job_idx_xr,
    dirpath_data,
    cache_path=dirpath_cache / "batch_rates.nc",
    lazy=True,
    dt_bin=5e-3,
    tau_smooth=20e-3,
    chunks={"rx": 1, "time": 2000},
    open_kwargs={"chunks": {"rx": 1, "time": 2000}},
)
```

On the first run:

- the batch is written into `batch_rates.nc`

On later runs:

- the file is reopened if its stored `cache_info` matches the current request

If it does not match:

- `collect_batch_*` raises `ValueError`

The batch cache uses the same `cache_info` style as `xr_cache.py`, but the actual load/reuse/build decision belongs to `batch_xr.py`.

## Eager vs Lazy

`lazy=False`

- simpler path
- batch is assembled in RAM
- useful for smaller batches

`lazy=True`

- batch is streamed directly into one NetCDF file
- avoids assembling the whole batch array in RAM
- useful for larger batches

Only the `lazy=True` path protects RAM during batch assembly.

## Downstream Processing Cache

Once the batch is collected, downstream xarray processing should still use [xr_cache.py](/home/nnovikov/repo/sim_data_analyzer/xr_cache.py):

```python
psd_xr, cache_hit = load_or_run_xr(
    dirpath_cache / "batch_rates_psd.nc",
    calc_xr_welch,
    rates_xr,
    win_len=4.0,
    win_overlap=0.75,
    fmin=2.0,
    fmax=100.0,
    compute=False,
)
```

That keeps the responsibilities split cleanly:

- `batch_xr.py`
  source-specific batch collection and batch-file reuse
- `xr_cache.py`
  generic xarray processing cache

## Demo

For a simple end-to-end example, see:

- [batch_rate_psd_chain.py](/home/nnovikov/repo/sim_data_analyzer/dev_scratch/demo/batch_rate_psd_chain.py)

That demo shows:

- cfg discovery
- one-file batch-rate collection with `cache_path` and `lazy=True`
- cached PSD on top via `load_or_run_xr(...)`
