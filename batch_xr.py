"""Batch-to-xarray helpers for collecting heterogeneous per-job outputs.

The public surface is organized around two ideas:
- Build a batch job-index xarray from cfg files.
- Convert one job at a time into xarray, then stack those results over batch dims.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from sim_data_analyzer import netpyne_res_parse_utils as parse_utils
from sim_data_analyzer.data_proc_utils import calc_pop_rate_dynamics
from sim_data_analyzer.spike_data import SpikeData
from sim_data_analyzer.xr_adapters import get_lfp_xr, get_net_rate_dynamics_xr
from sim_data_analyzer.xr_cache import (
    cache_info_matches,
    decode_xr_attrs_json,
    encode_xr_attrs_json,
    make_cache_info,
    normalize_cache_params,
    stamp_xr_cache_info,
)
from sim_data_analyzer.xr_io import load_xr, save_xr


__all__ = [
    "extract_batch_params_to_xr",
    "iter_batch_jobs",
    "extract_batch_spike_data_from_pkl",
    "collect_batch_xr",
    "collect_batch_xr_set",
    "collect_batch_json",
    "collect_batch_rates_from_pkl",
    "collect_batch_lfp_from_pkl",
    "collect_batch_rates_from_spike_data",
]


def _extract_nested(x: dict, key_seq: str):
    """Extract x[key1][key2]... using dotted keys."""
    value = x
    for key in key_seq.split("."):
        value = value[key]
    return value


def extract_batch_params_to_xr(
        dirpath_exp: str | Path,
        cfg_param_fields: dict[str, str],
        fname_cfg_templ: str = "*_cfg.json",
        job_pos_in_fname: int = -2,
        ) -> xr.DataArray:
    """Build a parameter-grid xarray that stores job ids at each batch point.

    Parameters
    ----------
    dirpath_exp
        Root folder that contains cfg JSON files for one batch.
    cfg_param_fields
        Mapping from output batch dimension names to dotted cfg field paths.
    fname_cfg_templ
        Glob pattern used to find cfg files under ``dirpath_exp``.
    job_pos_in_fname
        Position of the integer job id in the cfg filename stem split by ``_``.
    """
    dirpath_exp = Path(dirpath_exp)
    cfg_files = list(dirpath_exp.rglob(fname_cfg_templ))

    # Read each cfg file into one row of parameter values plus job id.
    job_idx_by_params = []
    for fpath_cfg in cfg_files:
        with fpath_cfg.open("r", encoding="utf-8") as fobj:
            cfg = json.load(fobj)
        params = {
            par_name: _extract_nested(cfg["simConfig"], field_seq)
            for par_name, field_seq in cfg_param_fields.items()
        }
        par_vals = [params[par_name] for par_name in cfg_param_fields]
        job_id = int(fpath_cfg.stem.split("_")[job_pos_in_fname])
        job_idx_by_params.append(par_vals + [job_id])

    # Convert the parameter table into a sorted xarray grid of job ids.
    dims = list(cfg_param_fields.keys())
    frame = pd.DataFrame(job_idx_by_params, columns=dims + ["job_id"])
    for dim in dims:
        frame[dim] = pd.Categorical(
            frame[dim],
            categories=sorted(frame[dim].unique()),
            ordered=True,
        )
    job_idx_xr = frame.set_index(dims).sort_index().to_xarray()["job_id"]
    return job_idx_xr.assign_coords({
        dim: np.asarray(job_idx_xr.coords[dim].values)
        for dim in job_idx_xr.dims
    })


def _iter_job_cells(job_idx_xr: xr.DataArray, skip_nan: bool = True):
    """Yield parameter-grid cells with both index and coordinate selections."""
    job_dims = list(job_idx_xr.dims)
    coord_values = {
        dim: job_idx_xr.coords[dim].values
        for dim in job_dims
    }
    for idx in np.ndindex(job_idx_xr.shape):
        raw_job_id = job_idx_xr.values[idx]
        if np.issubdtype(np.asarray(raw_job_id).dtype, np.floating) and np.isnan(raw_job_id):
            if skip_nan:
                continue
            job_id = None
        else:
            job_id = int(np.asarray(raw_job_id).item())

        sel = {}
        isel = {}
        for axis, dim in enumerate(job_dims):
            isel[dim] = idx[axis]
            coord_value = coord_values[dim][idx[axis]]
            if isinstance(coord_value, np.generic):
                coord_value = coord_value.item()
            sel[dim] = coord_value

        yield {
            "job_id": job_id,
            "idx": idx,
            "isel": isel,
            "sel": sel,
        }


def iter_batch_jobs(job_idx_xr: xr.DataArray):
    """Iterate over valid batch jobs with decoded selection metadata.

    Each yielded item contains the numeric ``job_id`` and both index-based and
    coordinate-based selectors for the corresponding batch point.
    """
    yield from _iter_job_cells(job_idx_xr, skip_nan=True)


def _get_fpath_by_templ(dirpath: Path, fname_templ: str) -> Path:
    """Resolve one file path using a glob template that should match once."""
    files = list(dirpath.glob(fname_templ))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one match in {dirpath} for pattern {fname_templ!r}"
        )
    return files[0]


def load_job_pkl(
        job: dict[str, Any],
        dirpath_data: str | Path,
        fname_templ: str = "grid_{job:05d}_data.pkl",
        ):
    """Load one pickled per-job result from a batch data folder."""
    dirpath_data = Path(dirpath_data)
    fpath = _get_fpath_by_templ(dirpath_data, fname_templ.format(job=job["job_id"]))
    with fpath.open("rb") as fobj:
        return pickle.load(fobj)


def load_job_json(
        job: dict[str, Any],
        dirpath_data: str | Path,
        fname_templ: str = "result_{job:05d}_*.json",
        ):
    """Load one per-job JSON payload from a batch results folder."""
    dirpath_data = Path(dirpath_data)
    fpath = _get_fpath_by_templ(dirpath_data, fname_templ.format(job=job["job_id"]))
    with fpath.open("r", encoding="utf-8") as fobj:
        return json.load(fobj)


def _open_job_xr(
        fpath_data: Path,
        data_type: str = "auto",
        variable: str | None = None,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Open one xarray file, optionally selecting a variable from a Dataset."""
    open_kwargs = {} if open_kwargs is None else dict(open_kwargs)
    if data_type == "auto":
        try:
            X = load_xr(fpath_data, data_type="dataarray", load=load, **open_kwargs)
        except (ValueError, OSError):
            X = load_xr(fpath_data, data_type="dataset", load=load, **open_kwargs)
    else:
        X = load_xr(fpath_data, data_type=data_type, load=load, **open_kwargs)

    if variable is not None:
        if not isinstance(X, xr.Dataset):
            raise ValueError("variable selection requires a Dataset input file")
        X = X[variable]
    return X


def load_job_xr(
        job: dict[str, Any],
        dirpath_data: str | Path,
        fname_templ: str = "{job:05d}_*.nc",
        data_type: str = "auto",
        variable: str | None = None,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Load one per-job xarray NetCDF file.

    Use ``variable`` to select one variable from a Dataset-backed file while
    keeping the rest of the batch-collection logic unchanged.
    """
    dirpath_data = Path(dirpath_data)
    fpath = _get_fpath_by_templ(dirpath_data, fname_templ.format(job=job["job_id"]))
    return _open_job_xr(
        fpath,
        data_type=data_type,
        variable=variable,
        load=load,
        open_kwargs=open_kwargs,
    )


def load_job_xr_set(
        job: dict[str, Any],
        dirpath_data: str | Path,
        fname_templ: str = "{job:05d}_{label}.nc",
        labels: list[str] | tuple[str, ...] | None = None,
        data_type: str = "auto",
        variable: str | None = None,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ) -> dict[str, xr.DataArray | xr.Dataset]:
    """Load a labeled set of per-job xarray files, typically one file per pop.

    This is the entry point for batch layouts where one job produces several
    compatible NetCDF files that should later be concatenated or selected.
    """
    dirpath_data = Path(dirpath_data)

    items = {}
    if labels is None:
        if "{label}" in fname_templ:
            raise ValueError("labels should be provided when fname_templ uses {label}")
        pattern = fname_templ.format(job=job["job_id"])
        for fpath in sorted(dirpath_data.glob(pattern)):
            items[fpath.stem] = _open_job_xr(
                fpath,
                data_type=data_type,
                variable=variable,
                load=load,
                open_kwargs=open_kwargs,
            )
        return items

    for label in labels:
        fpath = _get_fpath_by_templ(
            dirpath_data,
            fname_templ.format(job=job["job_id"], label=label),
        )
        items[str(label)] = _open_job_xr(
            fpath,
            data_type=data_type,
            variable=variable,
            load=load,
            open_kwargs=open_kwargs,
        )
    return items


def _fill_nested_array(
        out: np.ndarray,
        value: Any,
        dim_names: list[str],
        dict_dims: dict[str, list[Any]],
        prefix_idx: list[int] | None = None,
        ) -> None:
    """Fill one array from nested dicts whose keys map to dim coordinate values."""
    prefix_idx = [] if prefix_idx is None else prefix_idx
    if len(prefix_idx) == len(dim_names):
        out[tuple(prefix_idx)] = value
        return

    dim_name = dim_names[len(prefix_idx)]
    dim_values = list(dict_dims[dim_name])
    if not isinstance(value, dict):
        if len(dim_names) == len(prefix_idx) + 1:
            arr = np.asarray(value)
            if arr.shape == (len(dim_values),):
                out[tuple(prefix_idx + [slice(None)])] = arr
        return

    for idx, dim_value in enumerate(dim_values):
        if dim_value in value:
            _fill_nested_array(
                out,
                value[dim_value],
                dim_names,
                dict_dims,
                prefix_idx + [idx],
            )


def json_to_xr(
        payload: dict[str, Any],
        var_mappings: dict[str, str],
        dict_dims: dict[str, list[Any]] | None = None,
        extra_coords: dict[str, tuple[str, list[Any]]] | None = None,
        ) -> xr.Dataset:
    """Convert one JSON payload into an xarray Dataset with optional extra dims.

    ``dict_dims`` declares how nested dict keys in the JSON payload should
    become xarray dimensions, for example turning population keys into ``pop``.
    """
    dict_dims = {} if dict_dims is None else dict(dict_dims)
    dim_names = list(dict_dims.keys())
    coords = {dim_name: list(dim_values) for dim_name, dim_values in dict_dims.items()}

    # Materialize each requested JSON variable into one xarray variable.
    data_vars = {}
    shape = tuple(len(coords[dim_name]) for dim_name in dim_names)
    for var_name, json_key in var_mappings.items():
        value = _extract_nested(payload, json_key)
        if not dim_names:
            data_vars[var_name] = xr.DataArray(np.asarray(value))
            continue

        arr = np.full(shape, np.nan, dtype=np.float64)
        _fill_nested_array(arr, value, dim_names, dict_dims)
        data_vars[var_name] = xr.DataArray(arr, dims=dim_names, coords=coords)

    X = xr.Dataset(data_vars)
    if extra_coords:
        X = X.assign_coords(extra_coords)
    return X


def xr_set_to_xr(
        items: dict[str, xr.DataArray | xr.Dataset],
        combine: str = "concat",
        concat_dim: str = "item",
        concat_labels: list[str] | tuple[str, ...] | None = None,
        select_label: str | None = None,
        ):
    """Combine a set of per-job xr files by concatenating or selecting one item.

    ``combine='concat'`` is the typical mode for one-file-per-pop job outputs.
    ``combine='select'`` is useful when every job writes several files but only
    one of them should participate in the batch collection.
    """
    if combine == "select":
        if select_label is None:
            if len(items) != 1:
                raise ValueError("select mode needs select_label when many items exist")
            return next(iter(items.values()))
        return items[select_label]

    if combine != "concat":
        raise ValueError(f"Unsupported combine mode: {combine!r}")

    labels = list(items) if concat_labels is None else [str(label) for label in concat_labels]
    arrays = [items[label] for label in labels]
    return xr.concat(arrays, dim=xr.IndexVariable(concat_dim, labels))


def sim_result_to_rates_xr(
        sim_result: dict[str, Any],
        t_limits: tuple[float, float | None] = (0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        avg_cells: bool = True,
        pop_names: list[str] | tuple[str, ...] | None = None,
        ) -> xr.DataArray:
    """Extract one per-job population-rate xarray from a raw sim-result dict."""
    return get_net_rate_dynamics_xr(
        sim_result,
        t_limits=t_limits,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
        avg_cells=avg_cells,
        pop_names=pop_names,
    )


def sim_result_to_lfp_xr(sim_result: dict[str, Any]) -> xr.DataArray:
    """Extract one per-job LFP xarray from a raw sim-result dict."""
    return get_lfp_xr(sim_result)


def load_job_spike_data(
        job: dict[str, Any],
        dirpath_data: str | Path,
        fname_templ: str = "spikes_{job:05d}.npz",
        ) -> SpikeData:
    """Load one per-job SpikeData artifact."""
    dirpath_data = Path(dirpath_data)
    fpath = _get_fpath_by_templ(dirpath_data, fname_templ.format(job=job["job_id"]))
    return SpikeData.load(fpath)


def spike_data_to_rates_xr(
        spike_data: SpikeData,
        t_limits: tuple[float, float] | None = None,
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        ) -> xr.DataArray:
    """Convert combined SpikeData into population-rate dynamics xarray.

    This provides the fast-path batch input for workflows that cache extracted
    spikes separately from the heavier raw simulation results.
    """
    if not spike_data.combine_mode:
        spike_data = spike_data.combine()

    # Reuse stored spike metadata to recover the default analysis window.
    meta = spike_data.metadata
    if t_limits is None:
        t0 = 0.0 if meta["subtract_t0"] else float(meta["t0"])
        t1 = float(meta["tmax"]) - float(meta["t0"]) if meta["subtract_t0"] else float(meta["tmax"])
        t_limits = (t0, t1)

    pop_names = spike_data.get_pop_names()
    values = []
    tvec = None
    for pop_name in pop_names:
        pop_spikes = spike_data.get_pop_spikes(pop_name)
        pop_size = spike_data.get_pop_size(pop_name)
        if pop_size == 0:
            n_bins = int((t_limits[1] - t_limits[0]) / dt_bin)
            pop_tvec = np.arange(n_bins, dtype=np.float64) * dt_bin + t_limits[0]
            pop_values = np.full(n_bins, np.nan, dtype=np.float64)
        else:
            pop_tvec, pop_values = calc_pop_rate_dynamics(
                pop_spikes,
                t_limits,
                dt_bin=dt_bin,
                tau_smooth=tau_smooth,
                ncells=pop_size,
            )
        if tvec is None:
            tvec = pop_tvec
        values.append(np.asarray(pop_values, dtype=np.float64))

    X = xr.DataArray(
        np.asarray(values, dtype=np.float64),
        dims=["pop", "time"],
        coords={"pop": pop_names, "time": tvec},
    )
    X.attrs["time_units"] = "ms" if meta["ms"] else "s"
    return X


def _resolve_spike_extraction_request(
        sim_result: dict[str, Any],
        pop_names: list[str] | tuple[str, ...] | None,
        t_limits: tuple[float, float | None],
        combine: bool,
        subtract_t0: bool,
        ms: bool,
        ndigits: int,
        ) -> dict[str, Any]:
    """Resolve one SpikeData extraction request into concrete per-job values."""
    t0 = float(t_limits[0])
    tmax = t_limits[1]
    if tmax is None:
        tmax = parse_utils.get_sim_duration(sim_result)
    pop_names_resolved = parse_utils.get_pop_names(sim_result) if pop_names is None else list(pop_names)
    return {
        "pop_names": [str(pop_name) for pop_name in pop_names_resolved],
        "combine": bool(combine),
        "t0": t0,
        "tmax": float(tmax),
        "subtract_t0": bool(subtract_t0),
        "ms": bool(ms),
        "ndigits": int(ndigits),
    }


def _make_spike_cache_path(
        dirpath_spikes: str | Path,
        job_id: int,
        fname_spikes_templ: str,
        ) -> Path:
    """Build one per-job SpikeData cache path from its template."""
    return Path(dirpath_spikes) / fname_spikes_templ.format(job=job_id)


def _ensure_job_xr_compatible(
        template: xr.DataArray | xr.Dataset,
        X_job: xr.DataArray | xr.Dataset,
        ) -> None:
    """Check that one per-job object matches the probed batch schema."""
    if isinstance(template, xr.DataArray) != isinstance(X_job, xr.DataArray):
        raise ValueError("Per-job batch objects should all be DataArray or all be Dataset")

    if isinstance(template, xr.DataArray):
        if tuple(template.dims) != tuple(X_job.dims):
            raise ValueError(
                f"Incompatible per-job dims: expected {template.dims}, got {X_job.dims}"
            )
        for dim_name in template.dims:
            if template.sizes[dim_name] != X_job.sizes[dim_name]:
                msg = (
                    f"Incompatible per-job size for dim {dim_name!r}: "
                    f"expected {template.sizes[dim_name]}, got {X_job.sizes[dim_name]}"
                )
                if dim_name == "time":
                    msg += (
                        ". If this came from SpikeData-derived rates, pass an "
                        "explicit t_limits so every job uses the same time window."
                    )
                raise ValueError(msg)
            if dim_name in template.coords and dim_name in X_job.coords:
                if not np.array_equal(
                        np.asarray(template.coords[dim_name].values),
                        np.asarray(X_job.coords[dim_name].values)):
                    raise ValueError(
                        f"Incompatible per-job coordinate values for dim {dim_name!r}. "
                        "If this came from SpikeData-derived rates, pass an explicit "
                        "t_limits so every job uses the same time window."
                    )
        return

    assert isinstance(template, xr.Dataset) and isinstance(X_job, xr.Dataset)
    if list(template.data_vars) != list(X_job.data_vars):
        raise ValueError(
            f"Incompatible per-job Dataset variables: expected {list(template.data_vars)}, "
            f"got {list(X_job.data_vars)}"
        )
    for var_name in template.data_vars:
        _ensure_job_xr_compatible(template[var_name], X_job[var_name])


def _make_combined_coords(job_idx_xr: xr.DataArray, template: xr.DataArray | xr.Dataset):
    """Merge batch coords with one template object's native coords."""
    coords = xr.merge([
        job_idx_xr.coords.to_dataset(),
        template.coords.to_dataset(),
    ]).coords
    coords["job_id"] = job_idx_xr
    return coords


def _allocate_dataarray(
        job_idx_xr: xr.DataArray,
        template: xr.DataArray,
        ) -> xr.DataArray:
    """Allocate one eager combined DataArray shaped like batch dims + template dims."""
    job_dims = list(job_idx_xr.dims)
    job_shape = tuple(job_idx_xr.sizes[dim] for dim in job_dims)
    full_shape = job_shape + tuple(template.sizes[dim] for dim in template.dims)
    dtype = np.result_type(template.dtype, np.float64)
    return xr.DataArray(
        np.full(full_shape, np.nan, dtype=dtype),
        dims=job_dims + list(template.dims),
        coords=_make_combined_coords(job_idx_xr, template),
        attrs=copy.deepcopy(dict(template.attrs)),
        name=template.name,
    )


def _allocate_dataset(
        job_idx_xr: xr.DataArray,
        template: xr.Dataset,
        ) -> xr.Dataset:
    """Allocate one eager combined Dataset shaped like batch dims + template dims."""
    job_dims = list(job_idx_xr.dims)
    job_shape = tuple(job_idx_xr.sizes[dim] for dim in job_dims)
    data_vars = {}
    for var_name, X in template.data_vars.items():
        full_shape = job_shape + tuple(X.sizes[dim] for dim in X.dims)
        dtype = np.result_type(X.dtype, np.float64)
        data_vars[var_name] = (
            job_dims + list(X.dims),
            np.full(full_shape, np.nan, dtype=dtype),
        )
    X_out = xr.Dataset(
        data_vars=data_vars,
        coords=_make_combined_coords(job_idx_xr, template),
        attrs=copy.deepcopy(dict(template.attrs)),
    )
    for var_name in template.data_vars:
        X_out[var_name].attrs = copy.deepcopy(dict(template[var_name].attrs))
    return X_out


def _make_output_data_var_name(template: xr.DataArray) -> str:
    """Return the NetCDF variable name used for one output DataArray."""
    if template.name is not None:
        return str(template.name)
    return "__xarray_dataarray_variable__"


def _normalize_chunk_tuple(
        dim_names: list[str] | tuple[str, ...],
        sizes: dict[str, int],
        chunks: dict[str, int] | None,
        ) -> tuple[int, ...] | None:
    """Convert a dim-to-size chunk mapping into one per-variable chunk tuple."""
    if not chunks:
        return None

    chunk_tuple = []
    for dim_name in dim_names:
        dim_size = int(sizes[dim_name])
        chunk_size = int(chunks.get(dim_name, dim_size))
        chunk_tuple.append(max(1, min(dim_size, chunk_size)))
    return tuple(chunk_tuple)


def _get_numeric_storage_dtype(dtype) -> np.dtype:
    """Promote numeric dtypes so missing jobs can safely stay as NaN."""
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.number) or np.issubdtype(np_dtype, np.bool_):
        return np.dtype(np.result_type(np_dtype, np.float64))
    return np_dtype


def _get_h5_string_dtype():
    """Return the UTF-8 string dtype used for NetCDF string coordinates."""
    import h5py
    return h5py.string_dtype(encoding="utf-8")


def _get_coord_storage_dtype(values: np.ndarray):
    """Choose one NetCDF-compatible dtype for a coordinate variable."""
    if values.dtype.kind in {"U", "S"}:
        return _get_h5_string_dtype()
    if values.dtype.kind == "O":
        return _get_h5_string_dtype()
    return values.dtype


def _encode_attr_value_for_netcdf(value: Any) -> Any:
    """Encode nested attrs into JSON strings before writing NetCDF metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return "__json__:" + json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _stable_json_hash(value: Any) -> str:
    """Build a compact stable hash from a JSON-serializable object."""
    return hashlib.md5(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def _infer_batch_source(
        job_idx_xr: xr.DataArray,
        read_job_xr: Callable[[dict[str, Any]], xr.DataArray | xr.Dataset | None],
        ) -> list[Any]:
    """Build a lightweight source fingerprint for one batch collector call."""
    source_parts = [{"kind": "job_grid", "job_grid_hash": _stable_json_hash(job_idx_xr.values.tolist())}]
    reader_name = getattr(read_job_xr, "__name__", None)
    if reader_name is not None:
        source_parts.append({"kind": "reader", "name": str(reader_name)})
    return source_parts


def _require_incremental_netcdf_backend() -> None:
    """Ensure the optional backend needed for incremental NetCDF writes exists."""
    if importlib.util.find_spec("h5netcdf") is None:
        raise ImportError(
            "Incremental batch NetCDF writing requires the optional 'h5netcdf' package"
        )


def _find_first_job_object(
        job_idx_xr: xr.DataArray,
        read_job_xr: Callable[[dict[str, Any]], xr.DataArray | xr.Dataset | None],
        skip_missing: bool = True,
        ) -> tuple[dict[str, Any], xr.DataArray | xr.Dataset]:
    """Probe the first readable job to infer one batch output schema."""
    for entry in iter_batch_jobs(job_idx_xr):
        try:
            first_obj = read_job_xr(entry)
        except (FileNotFoundError, OSError, RuntimeError):
            if skip_missing:
                continue
            raise
        if first_obj is not None:
            return entry, first_obj
    raise ValueError("No readable per-job xarray was found for the batch")


def _non_dim_coord_names(
        coords: xr.core.coordinates.DataArrayCoordinates | xr.core.coordinates.DatasetCoordinates,
        var_dims: list[str] | tuple[str, ...],
        ) -> list[str]:
    """List non-dimension coords that should be attached to one data variable."""
    names = []
    var_dim_set = set(var_dims)
    for coord_name, coord in coords.items():
        if coord_name in coord.dims and coord.dims == (coord_name,):
            continue
        if set(coord.dims).issubset(var_dim_set):
            names.append(str(coord_name))
    return names


def _create_coord_variables(
        nc_file,
        coords: xr.core.coordinates.DataArrayCoordinates | xr.core.coordinates.DatasetCoordinates,
        ) -> None:
    """Create NetCDF coordinate variables for both dimension and auxiliary coords."""
    for coord_name, coord in coords.items():
        values = np.asarray(coord.values)
        dtype = _get_coord_storage_dtype(values)
        var = nc_file.create_variable(str(coord_name), tuple(coord.dims), dtype=dtype)
        var[:] = values
        for attr_name, attr_value in dict(coord.attrs).items():
            var.attrs[str(attr_name)] = _encode_attr_value_for_netcdf(attr_value)


def _create_data_variable(
        nc_file,
        var_name: str,
        X_var: xr.DataArray,
        coords,
        chunks: dict[str, int] | None = None,
        ) -> None:
    """Create one writable NetCDF data variable for a batch output."""
    storage_dtype = _get_numeric_storage_dtype(X_var.dtype)
    chunk_tuple = _normalize_chunk_tuple(list(X_var.dims), dict(X_var.sizes), chunks)
    create_kwargs = {"fillvalue": np.nan}
    if chunk_tuple is not None:
        create_kwargs["chunks"] = chunk_tuple
    var = nc_file.create_variable(
        str(var_name),
        tuple(X_var.dims),
        dtype=storage_dtype,
        **create_kwargs,
    )
    for attr_name, attr_value in dict(X_var.attrs).items():
        var.attrs[str(attr_name)] = _encode_attr_value_for_netcdf(attr_value)
    coord_names = _non_dim_coord_names(coords, X_var.dims)
    if coord_names:
        var.attrs["coordinates"] = " ".join(coord_names)


def _create_incremental_batch_file(
        fpath_out: Path,
        job_idx_xr: xr.DataArray,
        template: xr.DataArray | xr.Dataset,
        attrs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        ) -> str:
    """Create an empty chunked NetCDF batch file and return its data type."""
    import h5netcdf

    attrs = {} if attrs is None else dict(attrs)
    combined_coords = _make_combined_coords(job_idx_xr, template)
    combined_sizes = {str(dim_name): int(dim_size) for dim_name, dim_size in job_idx_xr.sizes.items()}
    for dim_name, dim_size in template.sizes.items():
        combined_sizes[str(dim_name)] = int(dim_size)
    data_type = "dataarray" if isinstance(template, xr.DataArray) else "dataset"
    data_var_name = _make_output_data_var_name(template) if data_type == "dataarray" else None

    fpath_out.parent.mkdir(parents=True, exist_ok=True)

    # Create the batch file structure before any per-job slices are written.
    with h5netcdf.File(fpath_out, "w") as nc_file:
        for dim_name, dim_size in combined_sizes.items():
            nc_file.dimensions[str(dim_name)] = int(dim_size)

        # Write all dimension and auxiliary coordinates up front.
        _create_coord_variables(nc_file, combined_coords)

        if isinstance(template, xr.DataArray):
            data_attrs = copy.deepcopy(dict(template.attrs))
            for attr_name, attr_value in attrs.items():
                if attr_name != "batch_data_var_name":
                    data_attrs[str(attr_name)] = attr_value
            X_var = xr.DataArray(
                dims=list(job_idx_xr.dims) + list(template.dims),
                coords=combined_coords,
                attrs=data_attrs,
                name=data_var_name,
            )
            _create_data_variable(
                nc_file,
                data_var_name,
                X_var,
                combined_coords,
                chunks=chunks,
            )
        else:
            for var_name, X in template.data_vars.items():
                X_var = xr.DataArray(
                    dims=list(job_idx_xr.dims) + list(X.dims),
                    coords=combined_coords,
                    attrs=copy.deepcopy(dict(X.attrs)),
                    name=str(var_name),
                )
                _create_data_variable(
                    nc_file,
                    str(var_name),
                    X_var,
                    combined_coords,
                    chunks=chunks,
                )

        # Store combined attrs and the DataArray marker used during reopen.
        for attr_name, attr_value in attrs.items():
            nc_file.attrs[str(attr_name)] = _encode_attr_value_for_netcdf(attr_value)
        if data_var_name is not None:
            nc_file.attrs["batch_data_var_name"] = str(data_var_name)

    return data_type


def _write_batch_entry(
        nc_file,
        entry: dict[str, Any],
        X_job: xr.DataArray | xr.Dataset,
        data_type: str,
        data_var_name: str | None = None,
        ) -> None:
    """Write one per-job object into its batch slice in the open NetCDF file."""
    if data_type == "dataarray":
        assert data_var_name is not None
        nc_file.variables[data_var_name][entry["idx"]] = np.asarray(X_job.values)
        return

    assert isinstance(X_job, xr.Dataset)
    for var_name in X_job.data_vars:
        nc_file.variables[str(var_name)][entry["idx"]] = np.asarray(X_job[var_name].values)


def _open_incremental_batch_result(
        fpath_out: Path,
        data_type: str,
        data_var_name: str | None = None,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ):
    """Reopen one completed incremental batch file as xarray."""
    open_kwargs = {} if open_kwargs is None else dict(open_kwargs)
    try:
        X = decode_xr_attrs_json(load_xr(fpath_out, data_type="dataset", load=load, **open_kwargs))
    except ImportError:
        open_kwargs_fallback = dict(open_kwargs)
        open_kwargs_fallback.pop("chunks", None)
        X = decode_xr_attrs_json(
            load_xr(fpath_out, data_type="dataset", load=load, **open_kwargs_fallback)
        )
    if data_type == "dataset":
        return X

    if data_var_name is None:
        data_var_name = str(X.attrs.get("batch_data_var_name"))
    X_data = decode_xr_attrs_json(X[data_var_name])
    for attr_name in ["cache_info", "proc_steps"]:
        if attr_name not in X_data.attrs and attr_name in X.attrs:
            X_data.attrs[attr_name] = copy.deepcopy(X.attrs[attr_name])
    return X_data


def _collect_batch_eager(
        job_idx_xr: xr.DataArray,
        read_job_xr: Callable[[dict[str, Any]], xr.DataArray | xr.Dataset | None],
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        ) -> xr.DataArray | xr.Dataset:
    """Collect one batch of per-job xr outputs into a single combined xarray.

    Parameters
    ----------
    job_idx_xr
        Parameter-grid xarray that maps each batch point to one job id.
    read_job_xr
        Callable that converts one yielded ``iter_batch_jobs()`` record into an
        xarray object for that job.
    chunks
        Optional output chunk layout applied after the eager batch assembly.
    skip_missing
        If True, ignore missing job artifacts. Otherwise raise immediately.
    """
    # Probe the first readable job to infer the per-job output schema.
    first_entry, first_obj = _find_first_job_object(
        job_idx_xr,
        read_job_xr,
        skip_missing=skip_missing,
    )

    if isinstance(first_obj, xr.DataArray):
        X_out = _allocate_dataarray(job_idx_xr, first_obj)
        X_out.data[first_entry["idx"]] = np.asarray(first_obj.values)
    elif isinstance(first_obj, xr.Dataset):
        X_out = _allocate_dataset(job_idx_xr, first_obj)
        for var_name in first_obj.data_vars:
            X_out[var_name].data[first_entry["idx"]] = np.asarray(first_obj[var_name].values)
    else:
        raise TypeError("read_job_xr should return an xarray DataArray, Dataset, or None")

    # Fill the remaining batch cells with per-job xarray outputs.
    for entry in _iter_job_cells(job_idx_xr, skip_nan=True):
        if entry["idx"] == first_entry["idx"]:
            continue
        try:
            X_job = read_job_xr(entry)
        except (FileNotFoundError, OSError, RuntimeError):
            if skip_missing:
                continue
            raise
        if X_job is None:
            continue

        if isinstance(X_out, xr.DataArray):
            _ensure_job_xr_compatible(first_obj, X_job)
            X_out.data[entry["idx"]] = np.asarray(X_job.values)
        else:
            _ensure_job_xr_compatible(first_obj, X_job)
            for var_name in X_out.data_vars:
                X_out[var_name].data[entry["idx"]] = np.asarray(X_job[var_name].values)

    if chunks:
        try:
            return X_out.chunk(chunks)
        except ImportError:
            return X_out
    return X_out


def _write_batch_netcdf(
        fpath_out: str | Path,
        job_idx_xr: xr.DataArray,
        read_job_xr: Callable[[dict[str, Any]], xr.DataArray | xr.Dataset | None],
        chunks: dict[str, int] | None = None,
        attrs: dict[str, Any] | None = None,
        skip_missing: bool = True,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        overwrite: bool = True,
        ) -> xr.DataArray | xr.Dataset:
    """Write one batch collection directly into a single chunked NetCDF file.

    Unlike ``collect_batch()``, this path does not allocate the full combined
    batch array in RAM. It creates the output file first and then writes one
    job slice at a time into that file.
    """
    import h5netcdf

    _require_incremental_netcdf_backend()
    fpath_out = Path(fpath_out)
    if fpath_out.exists():
        if not overwrite:
            raise FileExistsError(f"Batch output already exists: {fpath_out}")
        fpath_out.unlink()

    # Probe the first readable job to infer the output schema and file layout.
    first_entry, first_obj = _find_first_job_object(
        job_idx_xr,
        read_job_xr,
        skip_missing=skip_missing,
    )
    data_type = _create_incremental_batch_file(
        fpath_out,
        job_idx_xr,
        first_obj,
        attrs=attrs,
        chunks=chunks,
    )
    data_var_name = _make_output_data_var_name(first_obj) if data_type == "dataarray" else None

    # Write each batch point into its slice without materializing the full batch.
    with h5netcdf.File(fpath_out, "a") as nc_file:
        _write_batch_entry(
            nc_file,
            first_entry,
            first_obj,
            data_type=data_type,
            data_var_name=data_var_name,
        )
        for entry in _iter_job_cells(job_idx_xr, skip_nan=True):
            if entry["idx"] == first_entry["idx"]:
                continue
            try:
                X_job = read_job_xr(entry)
            except (FileNotFoundError, OSError, RuntimeError):
                if skip_missing:
                    continue
                raise
            if X_job is None:
                continue
            _ensure_job_xr_compatible(first_obj, X_job)
            _write_batch_entry(
                nc_file,
                entry,
                X_job,
                data_type=data_type,
                data_var_name=data_var_name,
            )

    # Reopen the finished batch file through xarray for downstream processing.
    return _open_incremental_batch_result(
        fpath_out,
        data_type=data_type,
        data_var_name=data_var_name,
        load=load,
        open_kwargs=open_kwargs,
    )


def _open_batch_cache_xr(
        fpath_cache: str | Path,
        data_type: str = "auto",
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Open one cached batch artifact with JSON attrs decoded."""
    open_kwargs = {} if open_kwargs is None else dict(open_kwargs)
    try:
        if data_type == "auto":
            try:
                X = load_xr(fpath_cache, data_type="dataarray", load=load, **open_kwargs)
            except (ValueError, OSError):
                X = load_xr(fpath_cache, data_type="dataset", load=load, **open_kwargs)
        else:
            X = load_xr(fpath_cache, data_type=data_type, load=load, **open_kwargs)
    except ImportError:
        open_kwargs_fallback = dict(open_kwargs)
        open_kwargs_fallback.pop("chunks", None)
        if data_type == "auto":
            try:
                X = load_xr(fpath_cache, data_type="dataarray", load=load, **open_kwargs_fallback)
            except (ValueError, OSError):
                X = load_xr(fpath_cache, data_type="dataset", load=load, **open_kwargs_fallback)
        else:
            X = load_xr(fpath_cache, data_type=data_type, load=load, **open_kwargs_fallback)
    return decode_xr_attrs_json(X)


def _save_batch_cache_xr(
        fpath_cache: str | Path,
        X: xr.DataArray | xr.Dataset,
        overwrite: bool = True,
        ) -> None:
    """Save one eager batch artifact to NetCDF with JSON-safe attrs."""
    fpath_cache = Path(fpath_cache)
    if fpath_cache.exists() and not overwrite:
        raise FileExistsError(f"Batch cache already exists: {fpath_cache}")

    # Save through a temp file so partial writes do not replace a good cache.
    fpath_cache.parent.mkdir(parents=True, exist_ok=True)
    fpath_tmp = fpath_cache.with_suffix(fpath_cache.suffix + ".tmp")
    save_xr(encode_xr_attrs_json(X), fpath_tmp)
    fpath_tmp.replace(fpath_cache)


def _make_batch_source_fingerprint(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        ) -> dict[str, Any]:
    """Build one lightweight source fingerprint for batch cache validation."""
    dirpath_data = Path(dirpath_data)
    if dirpath_data.exists():
        dirpath_str = str(dirpath_data.resolve())
    else:
        dirpath_str = str(dirpath_data)

    # Capture the selected job grid without storing the full grid verbatim in attrs.
    job_grid_payload = {
        "dims": list(job_idx_xr.dims),
        "coords": {
            dim_name: np.asarray(job_idx_xr.coords[dim_name].values).tolist()
            for dim_name in job_idx_xr.dims
        },
        "values": np.asarray(job_idx_xr.values).tolist(),
    }
    return {
        "dirpath_data": dirpath_str,
        "job_grid_hash": _stable_json_hash(job_grid_payload),
    }


def _build_batch_cache_payload(
        step: str,
        params: dict[str, Any],
        source: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the expected cache record and attrs for one batch artifact."""
    params_norm = normalize_cache_params(params)
    cache_info = make_cache_info(step, params_norm, source=source)
    attrs = {
        "cache_info": cache_info,
        "proc_steps": [{
            "name": str(step),
            "params": params_norm,
        }],
    }
    return cache_info, attrs


def _load_or_build_batch_xr(
        *,
        cache_step: str,
        cache_params: dict[str, Any],
        cache_source: dict[str, Any],
        build_eager: Callable[[], xr.DataArray | xr.Dataset],
        build_lazy: Callable[[str | Path, dict[str, Any]], xr.DataArray | xr.Dataset],
        cache_path: str | Path | None = None,
        cache_data_type: str = "auto",
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        overwrite: bool = True,
        ) -> xr.DataArray | xr.Dataset:
    """Reuse or build one batch artifact while keeping cache logic internal."""
    cache_info, cache_attrs = _build_batch_cache_payload(
        cache_step,
        cache_params,
        cache_source,
    )

    if cache_path is None:
        if lazy:
            raise ValueError("lazy=True requires cache_path for batch collection")
        return build_eager()

    fpath_cache = Path(cache_path)
    if fpath_cache.exists():
        cached = _open_batch_cache_xr(
            fpath_cache,
            data_type=cache_data_type,
            load=load,
            open_kwargs=open_kwargs,
        )
        found = cached.attrs.get("cache_info")
        if isinstance(found, dict) and cache_info_matches(found, cache_info):
            return cached
        if hasattr(cached, "close"):
            cached.close()
        raise ValueError(f"Cached batch artifact does not match requested step: {fpath_cache}")

    # Choose eager vs streaming build only when the cache is actually missing.
    if lazy:
        return build_lazy(fpath_cache, cache_attrs)

    X = build_eager()
    if not isinstance(X, (xr.DataArray, xr.Dataset)):
        raise TypeError("Batch builder should return an xarray DataArray or Dataset")
    X = stamp_xr_cache_info(
        X,
        step=cache_step,
        params=cache_params,
        source=cache_source,
        append_proc_step=True,
    )
    _save_batch_cache_xr(fpath_cache, X, overwrite=overwrite)
    return X


def _make_xr_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        data_type: str,
        variable: str | None,
        open_kwargs: dict[str, Any] | None = None,
        ) -> Callable[[dict[str, Any]], xr.DataArray | xr.Dataset]:
    """Build one per-job reader for already-formed xarray files."""
    return lambda job: load_job_xr(
        job,
        dirpath_data,
        fname_templ=fname_templ,
        data_type=data_type,
        variable=variable,
        load=False,
        open_kwargs=open_kwargs,
    )


def _make_xr_set_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        labels: list[str] | tuple[str, ...] | None,
        combine: str,
        concat_dim: str,
        concat_labels: list[str] | tuple[str, ...] | None,
        select_label: str | None,
        data_type: str,
        variable: str | None,
        open_kwargs: dict[str, Any] | None = None,
        ) -> Callable[[dict[str, Any]], xr.DataArray | xr.Dataset]:
    """Build one per-job reader for labeled xr-file sets."""
    return lambda job: xr_set_to_xr(
        load_job_xr_set(
            job,
            dirpath_data,
            fname_templ=fname_templ,
            labels=labels,
            data_type=data_type,
            variable=variable,
            load=False,
            open_kwargs=open_kwargs,
        ),
        combine=combine,
        concat_dim=concat_dim,
        concat_labels=concat_labels,
        select_label=select_label,
    )


def _make_json_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        var_mappings: dict[str, str],
        dict_dims: dict[str, list[Any]] | None = None,
        extra_coords: dict[str, tuple[str, list[Any]]] | None = None,
        ) -> Callable[[dict[str, Any]], xr.Dataset]:
    """Build one per-job reader for JSON summaries."""
    return lambda job: json_to_xr(
        load_job_json(job, dirpath_data, fname_templ=fname_templ),
        var_mappings=var_mappings,
        dict_dims=dict_dims,
        extra_coords=extra_coords,
    )


def _make_rates_from_pkl_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        t_limits: tuple[float, float | None],
        dt_bin: float,
        tau_smooth: float | None,
        avg_cells: bool,
        pop_names: list[str] | tuple[str, ...] | None = None,
        ) -> Callable[[dict[str, Any]], xr.DataArray]:
    """Build one per-job reader for rates computed from raw pickles."""
    pop_names_state = None if pop_names is None else [str(pop_name) for pop_name in pop_names]

    def read_job(job: dict[str, Any]) -> xr.DataArray:
        nonlocal pop_names_state
        sim_result = load_job_pkl(job, dirpath_data, fname_templ=fname_templ)
        if pop_names_state is None:
            pop_names_state = [str(pop_name) for pop_name in parse_utils.get_pop_names(sim_result)]
        return sim_result_to_rates_xr(
            sim_result,
            t_limits=t_limits,
            dt_bin=dt_bin,
            tau_smooth=tau_smooth,
            avg_cells=avg_cells,
            pop_names=pop_names_state,
        )

    return read_job


def _make_lfp_from_pkl_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        ) -> Callable[[dict[str, Any]], xr.DataArray]:
    """Build one per-job reader for LFP extracted from raw pickles."""
    return lambda job: sim_result_to_lfp_xr(
        load_job_pkl(job, dirpath_data, fname_templ=fname_templ),
    )


def _make_rates_from_spike_data_reader(
        dirpath_data: str | Path,
        fname_templ: str,
        t_limits: tuple[float, float] | None,
        dt_bin: float,
        tau_smooth: float | None,
        ) -> Callable[[dict[str, Any]], xr.DataArray]:
    """Build one per-job reader for rates computed from cached SpikeData."""
    return lambda job: spike_data_to_rates_xr(
        load_job_spike_data(job, dirpath_data, fname_templ=fname_templ),
        t_limits=t_limits,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
    )


def extract_batch_spike_data_from_pkl(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        dirpath_spikes: str | Path,
        fname_data_templ: str = "grid_{job:05d}_data.pkl",
        fname_spikes_templ: str = "spikes_{job:05d}.npz",
        pop_names: list[str] | tuple[str, ...] | None = None,
        t_limits: tuple[float, float | None] = (0, None),
        combine: bool = True,
        subtract_t0: bool = False,
        ms: bool = False,
        ndigits: int = 6,
        skip_missing: bool = True,
        ) -> Path:
    """Extract or reuse one per-job SpikeData cache for every batch point."""
    dirpath_data = Path(dirpath_data)
    dirpath_spikes = Path(dirpath_spikes)
    dirpath_spikes.mkdir(parents=True, exist_ok=True)

    # Walk the batch once and ensure one matching NPZ exists per readable job.
    for job in iter_batch_jobs(job_idx_xr):
        fpath_spikes = _make_spike_cache_path(
            dirpath_spikes,
            job["job_id"],
            fname_spikes_templ,
        )
        sim_result = None
        if fpath_spikes.exists() and pop_names is not None and t_limits[1] is not None:
            request = {
                "pop_names": [str(pop_name) for pop_name in pop_names],
                "combine": bool(combine),
                "t0": float(t_limits[0]),
                "tmax": float(t_limits[1]),
                "subtract_t0": bool(subtract_t0),
                "ms": bool(ms),
                "ndigits": int(ndigits),
            }
        else:
            try:
                sim_result = load_job_pkl(job, dirpath_data, fname_templ=fname_data_templ)
            except (FileNotFoundError, OSError, RuntimeError):
                if skip_missing:
                    continue
                raise
            request = _resolve_spike_extraction_request(
                sim_result,
                pop_names=pop_names,
                t_limits=t_limits,
                combine=combine,
                subtract_t0=subtract_t0,
                ms=ms,
                ndigits=ndigits,
            )

        # Reuse a matching per-job spike cache and fail fast on mismatches.
        if fpath_spikes.exists():
            spike_data = SpikeData.load(fpath_spikes)
            if spike_data.matches_request(**request):
                continue
            raise ValueError(
                f"SpikeData cache does not match requested extraction settings: {fpath_spikes}"
            )

        if sim_result is None:
            try:
                sim_result = load_job_pkl(job, dirpath_data, fname_templ=fname_data_templ)
            except (FileNotFoundError, OSError, RuntimeError):
                if skip_missing:
                    continue
                raise
        spike_data = SpikeData.from_sim_result(
            sim_result,
            pop_names=request["pop_names"],
            combine=request["combine"],
            t0=request["t0"],
            tmax=request["tmax"],
            subtract_t0=request["subtract_t0"],
            ms=request["ms"],
            ndigits=request["ndigits"],
        )
        spike_data.save(fpath_spikes)

    return dirpath_spikes


def collect_batch_xr(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        fname_templ: str = "{job:05d}_*.nc",
        data_type: str = "auto",
        variable: str | None = None,
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ):
    """Collect per-job xarray files into a single batch xarray.

    This is the direct replacement for the older A1-style collector used on
    already-derived NetCDF outputs such as ``rates_xr`` or ``vstats_xr``.
    """
    reader = _make_xr_reader(
        dirpath_data,
        fname_templ=fname_templ,
        data_type=data_type,
        variable=variable,
        open_kwargs=open_kwargs,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_xr",
        cache_params={
            "fname_templ": fname_templ,
            "data_type": data_type,
            "variable": variable,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="auto",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )


def collect_batch_xr_set(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        fname_templ: str = "{job:05d}_{label}.nc",
        labels: list[str] | tuple[str, ...] | None = None,
        combine: str = "concat",
        concat_dim: str = "pop",
        concat_labels: list[str] | tuple[str, ...] | None = None,
        select_label: str | None = None,
        data_type: str = "auto",
        variable: str | None = None,
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ):
    """Collect a labeled per-job set of xr files after concat/select combination.

    Use this when each job produces several related NetCDF files, for example
    one file per population or one file per variant.
    """
    reader = _make_xr_set_reader(
        dirpath_data,
        fname_templ=fname_templ,
        labels=labels,
        combine=combine,
        concat_dim=concat_dim,
        concat_labels=concat_labels,
        select_label=select_label,
        data_type=data_type,
        variable=variable,
        open_kwargs=open_kwargs,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_xr_set",
        cache_params={
            "fname_templ": fname_templ,
            "labels": labels,
            "combine": combine,
            "concat_dim": concat_dim,
            "concat_labels": concat_labels,
            "select_label": select_label,
            "data_type": data_type,
            "variable": variable,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="auto",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )


def collect_batch_json(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        var_mappings: dict[str, str],
        fname_templ: str = "result_{job:05d}_*.json",
        dict_dims: dict[str, list[Any]] | None = None,
        extra_coords: dict[str, tuple[str, list[Any]]] | None = None,
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ) -> xr.Dataset:
    """Collect per-job JSON outputs into a single batch Dataset.

    This is the generic path for job-written summary JSON files whose nested
    dict keys should become extra xarray dimensions such as ``pop``.
    """
    reader = _make_json_reader(
        dirpath_data,
        fname_templ=fname_templ,
        var_mappings=var_mappings,
        dict_dims=dict_dims,
        extra_coords=extra_coords,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_json",
        cache_params={
            "fname_templ": fname_templ,
            "var_mappings": var_mappings,
            "dict_dims": dict_dims,
            "extra_coords": extra_coords,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="dataset",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )


def collect_batch_rates_from_pkl(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        fname_templ: str = "grid_{job:05d}_data.pkl",
        t_limits: tuple[float, float | None] = (0, None),
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        avg_cells: bool = True,
        pop_names: list[str] | tuple[str, ...] | None = None,
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ) -> xr.DataArray:
    """Collect population-rate xarrays computed from raw per-job sim-result pickles.

    This is the main low-level batch builder for rate-dynamics analyses on raw
    NetPyNE result pickles.
    """
    reader = _make_rates_from_pkl_reader(
        dirpath_data,
        fname_templ=fname_templ,
        t_limits=t_limits,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
        avg_cells=avg_cells,
        pop_names=pop_names,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_rates_from_pkl",
        cache_params={
            "fname_templ": fname_templ,
            "t_limits": t_limits,
            "dt_bin": dt_bin,
            "tau_smooth": tau_smooth,
            "avg_cells": avg_cells,
            "pop_names": pop_names,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="dataarray",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )


def collect_batch_lfp_from_pkl(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        fname_templ: str = "grid_{job:05d}_data.pkl",
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ) -> xr.DataArray:
    """Collect LFP xarrays computed from raw per-job sim-result pickles."""
    reader = _make_lfp_from_pkl_reader(
        dirpath_data,
        fname_templ=fname_templ,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_lfp_from_pkl",
        cache_params={
            "fname_templ": fname_templ,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="dataarray",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )


def collect_batch_rates_from_spike_data(
        job_idx_xr: xr.DataArray,
        dirpath_data: str | Path,
        fname_templ: str = "spikes_{job:05d}.npz",
        t_limits: tuple[float, float] | None = None,
        dt_bin: float = 5e-3,
        tau_smooth: float | None = None,
        cache_path: str | Path | None = None,
        lazy: bool = False,
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
        skip_missing: bool = True,
        overwrite: bool = True,
        ) -> xr.DataArray:
    """Collect population-rate xarrays computed from cached per-job SpikeData."""
    reader = _make_rates_from_spike_data_reader(
        dirpath_data,
        fname_templ=fname_templ,
        t_limits=t_limits,
        dt_bin=dt_bin,
        tau_smooth=tau_smooth,
    )
    return _load_or_build_batch_xr(
        cache_step="collect_batch_rates_from_spike_data",
        cache_params={
            "fname_templ": fname_templ,
            "t_limits": t_limits,
            "dt_bin": dt_bin,
            "tau_smooth": tau_smooth,
            "skip_missing": skip_missing,
        },
        cache_source=_make_batch_source_fingerprint(job_idx_xr, dirpath_data),
        build_eager=lambda: _collect_batch_eager(
            job_idx_xr,
            reader,
            chunks=chunks,
            skip_missing=skip_missing,
        ),
        build_lazy=lambda fpath_cache, attrs: _write_batch_netcdf(
            fpath_cache,
            job_idx_xr,
            reader,
            chunks=chunks,
            attrs=attrs,
            skip_missing=skip_missing,
            load=load,
            open_kwargs=open_kwargs,
            overwrite=overwrite,
        ),
        cache_path=cache_path,
        cache_data_type="dataarray",
        lazy=lazy,
        load=load,
        open_kwargs=open_kwargs,
        overwrite=overwrite,
    )
