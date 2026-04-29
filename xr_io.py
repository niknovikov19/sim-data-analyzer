import importlib.util
from pathlib import Path

import xarray as xr


def _resolve_engine(engine: str | None) -> str | None:
    """Resolve a NetCDF engine, falling back when optional deps are missing. """
    if engine is None:
        return None
    if engine != 'h5netcdf':
        return engine
    if importlib.util.find_spec('h5netcdf') is not None:
        return engine
    return 'scipy'


def save_xr_dataarray(
        X: xr.DataArray,
        fpath,
        engine: str = 'h5netcdf',
        **kwargs
        ) -> None:
    """Save an xarray DataArray to a NetCDF file. """
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    engine = _resolve_engine(engine)
    X.to_netcdf(fpath, engine=engine, **kwargs)


def load_xr_dataarray(
        fpath,
        engine: str = 'h5netcdf',
        load: bool = False,
        **kwargs
        ) -> xr.DataArray:
    """Open an xarray DataArray from a NetCDF file. """
    engine = _resolve_engine(engine)
    X = xr.open_dataarray(fpath, engine=engine, **kwargs)
    if load:
        X.load()
    return X


def save_xr_dataset(
        X: xr.Dataset,
        fpath,
        engine: str = 'h5netcdf',
        **kwargs
        ) -> None:
    """Save an xarray Dataset to a NetCDF file. """
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    engine = _resolve_engine(engine)
    X.to_netcdf(fpath, engine=engine, **kwargs)


def load_xr_dataset(
        fpath,
        engine: str = 'h5netcdf',
        load: bool = False,
        **kwargs
        ) -> xr.Dataset:
    """Open an xarray Dataset from a NetCDF file. """
    engine = _resolve_engine(engine)
    X = xr.open_dataset(fpath, engine=engine, **kwargs)
    if load:
        X.load()
    return X


def save_xr(X: xr.DataArray | xr.Dataset, fpath, engine: str = 'h5netcdf', **kwargs) -> None:
    """Save an xarray DataArray or Dataset to a NetCDF file. """
    if isinstance(X, xr.DataArray):
        return save_xr_dataarray(X, fpath, engine=engine, **kwargs)
    if isinstance(X, xr.Dataset):
        return save_xr_dataset(X, fpath, engine=engine, **kwargs)
    raise TypeError('X should be an xarray DataArray or Dataset')


def load_xr(
        fpath,
        data_type: str = 'dataarray',
        engine: str = 'h5netcdf',
        load: bool = False,
        **kwargs
        ) -> xr.DataArray | xr.Dataset:
    """Open an xarray DataArray or Dataset from a NetCDF file. """
    if data_type == 'dataarray':
        return load_xr_dataarray(fpath, engine=engine, load=load, **kwargs)
    if data_type == 'dataset':
        return load_xr_dataset(fpath, engine=engine, load=load, **kwargs)
    raise ValueError(f'Unsupported data_type {data_type!r}')
