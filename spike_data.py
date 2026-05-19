"""Spike-data container with lightweight NPZ persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from sim_data_analyzer import netpyne_res_parse_utils as parse_utils


def _copy_spike_list(spike_list: List[np.ndarray]) -> List[np.ndarray]:
    """Copy a list of spike arrays."""
    return [np.array(spikes, copy=True) for spikes in spike_list]


def _encode_ragged(spike_list: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Encode ragged spike arrays as concatenated values plus offsets."""
    offsets = np.zeros(len(spike_list) + 1, dtype=np.int64)
    parts = []
    n_total = 0
    for idx, spikes in enumerate(spike_list):
        spikes = np.asarray(spikes)
        parts.append(spikes)
        n_total += len(spikes)
        offsets[idx + 1] = n_total
    if parts:
        values = np.concatenate(parts)
    else:
        values = np.array([], dtype=np.float64)
    return values, offsets


def _decode_ragged(values: np.ndarray, offsets: np.ndarray) -> List[np.ndarray]:
    """Decode concatenated spike values plus offsets to ragged arrays."""
    spike_list = []
    for idx in range(len(offsets) - 1):
        spike_list.append(np.array(values[offsets[idx]:offsets[idx + 1]], copy=True))
    return spike_list


def _normalize_pop_names(pop_names: List[str] | tuple[str, ...] | None) -> List[str] | None:
    """Convert optional population names into one stable string list."""
    if pop_names is None:
        return None
    return [str(pop_name) for pop_name in pop_names]


@dataclass(frozen=True)
class _SpikeMeta:
    """Metadata that defines one extracted spike representation."""

    combine: bool
    t0: float
    tmax: float
    subtract_t0: bool
    ms: bool
    ndigits: int


class SpikeData:
    """Stored spike data for one extracted simulation-result view."""

    def __init__(
            self,
            spikes_by_pop: Dict[str, List[np.ndarray]],
            *,
            meta: _SpikeMeta,
            cell_gids_by_pop: Dict[str, np.ndarray] | None = None,
            pop_sizes: Dict[str, int] | None = None,
            ):
        self._meta = meta
        self._pop_names = list(spikes_by_pop)
        self._spikes_by_pop = {
            pop_name: _copy_spike_list(spikes_by_pop[pop_name])
            for pop_name in self._pop_names
        }
        self._cell_gids_by_pop = None
        if cell_gids_by_pop is not None:
            self._cell_gids_by_pop = {
                pop_name: np.array(cell_gids_by_pop[pop_name], copy=True)
                for pop_name in self._pop_names
            }
        if pop_sizes is None:
            if meta.combine:
                pop_sizes = {pop_name: 1 for pop_name in self._pop_names}
            else:
                pop_sizes = {
                    pop_name: len(self._spikes_by_pop[pop_name])
                    for pop_name in self._pop_names
                }
        self._pop_sizes = dict(pop_sizes)

    @classmethod
    def from_sim_result(
            cls,
            sim_result,
            pop_names=None,
            combine: bool = True,
            t0: float = 0,
            tmax: float | None = None,
            subtract_t0: bool = True,
            ms: bool = False,
            ndigits: int = 6,
            ) -> "SpikeData":
        """Extract spikes from a NetPyNE sim result into a fixed representation."""
        pop_names = pop_names or parse_utils.get_pop_names(sim_result)
        if tmax is None:
            tmax = parse_utils.get_sim_duration(sim_result)

        spikes_by_pop = {}
        cell_gids_by_pop = {} if not combine else None
        pop_sizes = {}
        for pop_name in pop_names:
            spikes = parse_utils.get_pop_spikes(
                sim_result,
                pop_name,
                combine_cells=combine,
                t0=t0,
                tmax=tmax,
                subtract_t0=subtract_t0,
                ms=ms,
                ndigits=ndigits,
            )
            spikes_by_pop[pop_name] = _copy_spike_list(spikes)
            pop_sizes[pop_name] = parse_utils.get_pop_size(sim_result, pop_name)
            if not combine:
                cell_gids_by_pop[pop_name] = parse_utils.get_pop_cell_gids(sim_result, pop_name)

        meta = _SpikeMeta(
            combine=combine,
            t0=float(t0),
            tmax=float(tmax),
            subtract_t0=bool(subtract_t0),
            ms=bool(ms),
            ndigits=int(ndigits),
        )
        return cls(
            spikes_by_pop,
            meta=meta,
            cell_gids_by_pop=cell_gids_by_pop,
            pop_sizes=pop_sizes,
        )

    @classmethod
    def load(cls, fpath) -> "SpikeData":
        """Load spike data from an NPZ file."""
        fpath = Path(fpath)
        with np.load(fpath, allow_pickle=False) as data:
            pop_names = data["pop_names"].astype(str).tolist()
            meta = _SpikeMeta(
                combine=bool(data["meta__combine"].item()),
                t0=float(data["meta__t0"].item()),
                tmax=float(data["meta__tmax"].item()),
                subtract_t0=bool(data["meta__subtract_t0"].item()),
                ms=bool(data["meta__ms"].item()),
                ndigits=int(data["meta__ndigits"].item()),
            )
            spikes_by_pop = {}
            cell_gids_by_pop = {} if not meta.combine else None
            pop_sizes = {}

            for idx, pop_name in enumerate(pop_names):
                prefix = f"pop_{idx}__"
                pop_sizes[pop_name] = int(data[f"{prefix}pop_size"].item())
                if meta.combine:
                    spikes_by_pop[pop_name] = [
                        np.array(data[f"{prefix}combined_times"], copy=True)
                    ]
                else:
                    spikes_by_pop[pop_name] = _decode_ragged(
                        data[f"{prefix}cell_times"],
                        data[f"{prefix}cell_offsets"],
                    )
                    cell_gids_by_pop[pop_name] = np.array(
                        data[f"{prefix}cell_gids"], copy=True
                    )

        return cls(
            spikes_by_pop,
            meta=meta,
            cell_gids_by_pop=cell_gids_by_pop,
            pop_sizes=pop_sizes,
        )

    @property
    def combine_mode(self) -> bool:
        """Whether the stored representation is combined across cells."""
        return self._meta.combine

    @property
    def metadata(self) -> dict:
        """Return extraction metadata as a plain dict."""
        return {
            "combine": self._meta.combine,
            "t0": self._meta.t0,
            "tmax": self._meta.tmax,
            "subtract_t0": self._meta.subtract_t0,
            "ms": self._meta.ms,
            "ndigits": self._meta.ndigits,
        }

    def matches_request(
            self,
            *,
            pop_names: List[str] | tuple[str, ...] | None,
            combine: bool,
            t0: float,
            tmax: float,
            subtract_t0: bool,
            ms: bool,
            ndigits: int,
            ) -> bool:
        """Return whether this SpikeData matches one resolved extraction request.

        ``t0`` and ``tmax`` should be concrete resolved values rather than a
        partially specified window such as ``(t0, None)``.
        """
        expected_pop_names = _normalize_pop_names(pop_names)
        if expected_pop_names is not None and self.get_pop_names() != expected_pop_names:
            return False

        meta = self.metadata
        return (
            bool(meta["combine"]) == bool(combine) and
            float(meta["t0"]) == float(t0) and
            float(meta["tmax"]) == float(tmax) and
            bool(meta["subtract_t0"]) == bool(subtract_t0) and
            bool(meta["ms"]) == bool(ms) and
            int(meta["ndigits"]) == int(ndigits)
        )

    def get_pop_names(self) -> List[str]:
        """Return population names in stored order."""
        return list(self._pop_names)

    def get_pop_size(self, pop_name: str) -> int:
        """Return the number of cells in a population."""
        return self._pop_sizes[pop_name]

    def get_pop_cell_gids(self, pop_name: str) -> np.ndarray:
        """Return stored population cell gids for per-cell spike data."""
        if self._cell_gids_by_pop is None:
            raise ValueError("Cell gids are only available for per-cell SpikeData")
        return np.array(self._cell_gids_by_pop[pop_name], copy=True)

    def get_pop_spikes(self, pop_name: str) -> List[np.ndarray]:
        """Return stored spikes for one population."""
        return _copy_spike_list(self._spikes_by_pop[pop_name])

    def get_net_spikes(self, pop_names=None) -> Dict[str, List[np.ndarray]]:
        """Return stored spikes for selected populations."""
        pop_names = pop_names or self._pop_names
        return {
            pop_name: self.get_pop_spikes(pop_name)
            for pop_name in pop_names
        }

    def combine(self) -> "SpikeData":
        """Combine per-cell spikes into one spike train per population."""
        if self._meta.combine:
            raise ValueError("SpikeData is already stored in combined mode")

        spikes_by_pop = {}
        for pop_name in self._pop_names:
            spike_list = self._spikes_by_pop[pop_name]
            if spike_list:
                combined = np.sort(np.concatenate(spike_list))
            else:
                combined = np.array([], dtype=np.float64)
            spikes_by_pop[pop_name] = [combined]

        meta = _SpikeMeta(
            combine=True,
            t0=self._meta.t0,
            tmax=self._meta.tmax,
            subtract_t0=self._meta.subtract_t0,
            ms=self._meta.ms,
            ndigits=self._meta.ndigits,
        )
        return SpikeData(spikes_by_pop, meta=meta, pop_sizes=self._pop_sizes)

    def save(self, fpath) -> None:
        """Save spike data to an NPZ file."""
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "format_version": np.array(1, dtype=np.int64),
            "pop_names": np.array(self._pop_names, dtype=str),
            "meta__combine": np.array(self._meta.combine, dtype=bool),
            "meta__t0": np.array(self._meta.t0, dtype=np.float64),
            "meta__tmax": np.array(self._meta.tmax, dtype=np.float64),
            "meta__subtract_t0": np.array(self._meta.subtract_t0, dtype=bool),
            "meta__ms": np.array(self._meta.ms, dtype=bool),
            "meta__ndigits": np.array(self._meta.ndigits, dtype=np.int64),
        }

        for idx, pop_name in enumerate(self._pop_names):
            prefix = f"pop_{idx}__"
            arrays[f"{prefix}pop_size"] = np.array(
                self._pop_sizes[pop_name], dtype=np.int64
            )
            if self._meta.combine:
                arrays[f"{prefix}combined_times"] = np.array(
                    self._spikes_by_pop[pop_name][0], copy=True
                )
            else:
                arrays[f"{prefix}cell_gids"] = np.array(
                    self._cell_gids_by_pop[pop_name], copy=True
                )
                values, offsets = _encode_ragged(self._spikes_by_pop[pop_name])
                arrays[f"{prefix}cell_times"] = values
                arrays[f"{prefix}cell_offsets"] = offsets

        np.savez(fpath, **arrays)
