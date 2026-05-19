"""Lightweight cached xarray helpers for exploratory processing chains.

The public helpers keep caching explicit while storing the minimal metadata
needed to validate cached xarray artifacts after NetCDF round-trips.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sim_data_analyzer.xr_io import load_xr, save_xr


JSON_ATTR_PREFIX = "__json__:"
DEFAULT_IGNORED_CACHE_PARAM_KEYS = frozenset({
    "compute",
    "store_proc_info",
    "load",
    "engine",
    "chunks",
    "open_kwargs",
    "save_kwargs",
    "cache_open_kwargs",
    "cache_save_kwargs",
})


def _normalize_jsonable(value: Any) -> Any:
    """Convert one object into a stable JSON-serializable form."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_jsonable(val)
            for key, val in value.items()
        }
    return value


def _stable_json_dumps(value: Any) -> str:
    """Serialize a normalized object with a stable key order."""
    return json.dumps(_normalize_jsonable(value), sort_keys=True, separators=(",", ":"))


def _make_short_hash(value: Any) -> str:
    """Build a compact stable hash from one normalized object."""
    return hashlib.md5(_stable_json_dumps(value).encode("utf-8")).hexdigest()[:12]


def _is_scalar_attr_value(value: Any) -> bool:
    """Return whether one attr value is NetCDF-friendly without JSON encoding."""
    return isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            np.integer,
            np.floating,
            np.bool_,
        ),
    ) or value is None


def _encode_attr_value(value: Any) -> Any:
    """Encode a possibly nested attr value into a NetCDF-safe scalar."""
    if _is_scalar_attr_value(value):
        return value
    return JSON_ATTR_PREFIX + _stable_json_dumps(value)


def _decode_attr_value(value: Any) -> Any:
    """Decode one attr value produced by _encode_attr_value()."""
    if isinstance(value, str) and value.startswith(JSON_ATTR_PREFIX):
        return json.loads(value[len(JSON_ATTR_PREFIX):])
    return value


def encode_xr_attrs_json(
        X: xr.DataArray | xr.Dataset,
        keys: tuple[str, ...] | list[str] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Return a shallow copy with selected attrs JSON-encoded when needed.

    Use this before saving xarray objects whose attrs contain nested dict/list
    metadata that should survive a NetCDF round-trip.
    """
    X_out = X.copy(deep=False)
    attrs = copy.deepcopy(dict(X.attrs))
    keys_used = set(attrs) if keys is None else set(keys)
    for key in list(attrs):
        if key in keys_used:
            attrs[key] = _encode_attr_value(attrs[key])
    X_out.attrs = attrs
    return X_out


def decode_xr_attrs_json(
        X: xr.DataArray | xr.Dataset,
        keys: tuple[str, ...] | list[str] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Return a shallow copy with selected attrs JSON-decoded when possible.

    This reverses ``encode_xr_attrs_json()`` after loading cached artifacts.
    """
    X_out = X.copy(deep=False)
    attrs = copy.deepcopy(dict(X.attrs))
    keys_used = set(attrs) if keys is None else set(keys)
    for key in list(attrs):
        if key in keys_used:
            attrs[key] = _decode_attr_value(attrs[key])
    X_out.attrs = attrs
    return X_out


def _fingerprint_small_xarray(X: xr.DataArray | xr.Dataset) -> dict[str, Any] | None:
    """Fingerprint small xarray inputs without materializing large arrays."""
    size_limit = 10_000
    total_size = 1
    for size in X.sizes.values():
        total_size *= int(size)
    if total_size > size_limit:
        return None

    if isinstance(X, xr.DataArray):
        return {
            "kind": "dataarray",
            "dims": list(X.dims),
            "coords": {
                str(name): np.asarray(coord.values).tolist()
                for name, coord in X.coords.items()
            },
            "values": np.asarray(X.values).tolist(),
        }

    return {
        "kind": "dataset",
        "dims": dict(X.sizes),
        "coords": {
            str(name): np.asarray(coord.values).tolist()
            for name, coord in X.coords.items()
        },
        "data_vars": {
            str(name): np.asarray(X[name].values).tolist()
            for name in X.data_vars
        },
    }


def _fingerprint_source_arg(value: Any) -> Any:
    """Return one lightweight fingerprint for a likely source object."""
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        cache_info = value.attrs.get("cache_info")
        if isinstance(cache_info, str) and cache_info.startswith(JSON_ATTR_PREFIX):
            cache_info = _decode_attr_value(cache_info)
        if isinstance(cache_info, dict) and "cache_id" in cache_info:
            return {"kind": "xr_cache", "cache_id": str(cache_info["cache_id"])}
        return _fingerprint_small_xarray(value)

    if isinstance(value, (str, Path)):
        path = Path(value)
        resolved = str(path.resolve()) if path.exists() else str(path)
        if path.exists() and path.is_file():
            stat = path.stat()
            return {
                "kind": "file",
                "path": resolved,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        if path.exists() and path.is_dir():
            return {"kind": "dir", "path": resolved}
        return {"kind": "path", "path": resolved}

    return None


def infer_source_fingerprint(
        args: tuple[Any, ...] | list[Any],
        source_data: Any = None,
        ) -> Any:
    """Infer a lightweight provenance fingerprint from cached-step inputs.

    The returned object is intentionally compact: it is meant for cache
    validation, not for full provenance capture.
    """
    if source_data is not None:
        args = tuple(args) + (source_data,)

    # Collect simple fingerprints from the provided sources and upstream xr data.
    parts = []
    for arg in args:
        part = _fingerprint_source_arg(arg)
        if part is not None:
            parts.append(part)

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def normalize_cache_params(
        params: dict[str, Any] | None,
        ignored_keys: set[str] | frozenset[str] | None = None,
        ) -> dict[str, Any]:
    """Normalize semantic cache parameters and drop execution-only keys.

    This keeps cache validation focused on parameters that affect the resulting
    values, while ignoring runtime concerns such as chunking or eager loading.
    """
    params = {} if params is None else dict(params)
    ignored_keys = DEFAULT_IGNORED_CACHE_PARAM_KEYS if ignored_keys is None else ignored_keys
    return {
        str(key): _normalize_jsonable(value)
        for key, value in params.items()
        if key not in ignored_keys
    }


def make_cache_info(
        step: str,
        params: dict[str, Any] | None,
        source: Any = None,
        cache_version: int = 1,
        ) -> dict[str, Any]:
    """Build one compact machine-readable cache record.

    The returned dict is designed to live in ``xarray.attrs['cache_info']`` and
    to support equality-based cache validation.
    """
    info = {
        "step": str(step),
        "params": normalize_cache_params(params),
        "source": _normalize_jsonable(source),
        "cache_version": int(cache_version),
    }
    info["cache_id"] = _make_short_hash(info)
    return info


def stamp_xr_cache_info(
        X: xr.DataArray | xr.Dataset,
        step: str,
        params: dict[str, Any] | None,
        source: Any = None,
        append_proc_step: bool = True,
        cache_version: int = 1,
        ) -> xr.DataArray | xr.Dataset:
    """Attach cache_info and optionally append a readable processing step.

    ``cache_info`` is the machine-facing validation record. ``proc_steps`` is a
    lightweight readable history that can accumulate across processing steps.
    """
    X_out = X.copy(deep=False)
    attrs = copy.deepcopy(dict(X.attrs))
    cache_info = make_cache_info(step, params, source=source, cache_version=cache_version)
    attrs["cache_info"] = cache_info

    # Extend the readable per-step history when requested.
    if append_proc_step:
        proc_steps = copy.deepcopy(attrs.get("proc_steps", []))
        if isinstance(proc_steps, str) and proc_steps.startswith(JSON_ATTR_PREFIX):
            proc_steps = _decode_attr_value(proc_steps)
        if not isinstance(proc_steps, list):
            proc_steps = []
        proc_steps.append({
            "name": str(step),
            "params": normalize_cache_params(params),
        })
        attrs["proc_steps"] = proc_steps

    X_out.attrs = attrs
    return X_out


def _load_cached_xr(
        fpath_cache,
        data_type: str = "auto",
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        ) -> xr.DataArray | xr.Dataset:
    """Open one cached xarray artifact, auto-detecting its concrete type."""
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


def cache_info_matches(found: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return whether the stored cache_info matches the requested semantics."""
    return (
        found.get("step") == expected.get("step")
        and found.get("params") == expected.get("params")
        and found.get("source") == expected.get("source")
        and found.get("cache_version") == expected.get("cache_version")
    )


def load_or_run_xr(
        fpath_cache,
        fn,
        *args,
        cache_step: str | None = None,
        cache_params: dict[str, Any] | None = None,
        source_data: Any = None,
        ignored_param_keys: set[str] | frozenset[str] | None = None,
        data_type: str = "auto",
        load: bool = False,
        open_kwargs: dict[str, Any] | None = None,
        save_kwargs: dict[str, Any] | None = None,
        on_mismatch: str = "recompute",
        append_proc_step: bool = True,
        cache_version: int = 1,
        **fn_kwargs,
        ) -> tuple[xr.DataArray | xr.Dataset, bool]:
    """Load one cached XR artifact or compute, stamp, save, and return it.

    Parameters
    ----------
    fpath_cache
        Path to the NetCDF cache artifact.
    fn
        Callable that returns an xarray DataArray or Dataset when recomputation
        is needed.
    cache_step, cache_params
        Optional explicit cache metadata. By default they are inferred from
        ``fn`` and ``fn_kwargs``.
    source_data
        Optional extra object used only for source fingerprinting.
    on_mismatch
        Either recompute and overwrite the cache, or raise immediately.
    """
    fpath_cache = Path(fpath_cache)
    open_kwargs = {} if open_kwargs is None else dict(open_kwargs)
    save_kwargs = {} if save_kwargs is None else dict(save_kwargs)

    # Build the expected cache metadata from the requested processing step.
    step = str(cache_step or getattr(fn, "__name__", "cached_step"))
    params = normalize_cache_params(
        fn_kwargs if cache_params is None else cache_params,
        ignored_keys=ignored_param_keys,
    )
    source = infer_source_fingerprint(args, source_data=source_data)
    expected = make_cache_info(step, params, source=source, cache_version=cache_version)

    # Reuse an existing artifact only if its cache metadata matches exactly.
    if fpath_cache.exists():
        cached = _load_cached_xr(
            fpath_cache,
            data_type=data_type,
            load=load,
            open_kwargs=open_kwargs,
        )
        found = cached.attrs.get("cache_info")
        if isinstance(found, dict) and cache_info_matches(found, expected):
            return cached, True
        if on_mismatch == "raise":
            raise ValueError(
                f'Cached XR artifact does not match requested step: {fpath_cache}'
            )
        if hasattr(cached, "close"):
            cached.close()

    # Compute the artifact, stamp cache metadata, and replace the cache atomically.
    X = fn(*args, **fn_kwargs)
    if not isinstance(X, (xr.DataArray, xr.Dataset)):
        raise TypeError("fn should return an xarray DataArray or Dataset")

    X = stamp_xr_cache_info(
        X,
        step=step,
        params=params,
        source=source,
        append_proc_step=append_proc_step,
        cache_version=cache_version,
    )
    fpath_tmp = fpath_cache.with_suffix(fpath_cache.suffix + ".tmp")
    save_xr(encode_xr_attrs_json(X), fpath_tmp, **save_kwargs)
    fpath_tmp.replace(fpath_cache)
    return X, False
