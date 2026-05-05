"""Generic helpers for extracting and sampling intervals from binary masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr


def extract_mask_intervals(mask: xr.DataArray) -> dict:
    """Extract contiguous mask==1 and mask==0 intervals from one mask."""
    values, time_values, y_values = _normalize_mask(mask)
    if y_values is None:
        return {
            'mask_1': _extract_value_intervals(values, time_values, target=1),
            'mask_0': _extract_value_intervals(values, time_values, target=0),
        }

    mask_1 = {}
    mask_0 = {}
    for channel_index, y_value in enumerate(y_values):
        key = _format_y_key(y_value)
        channel_values = values[channel_index]
        mask_1[key] = _extract_value_intervals(channel_values, time_values, target=1)
        mask_0[key] = _extract_value_intervals(channel_values, time_values, target=0)
    return {'mask_1': mask_1, 'mask_0': mask_0}


def sample_control_intervals(
        mask1_intervals: Sequence[tuple[float, float]],
        mask0_intervals: Sequence[tuple[float, float]],
        seed: int = 0,
        max_seed_tries: int = 1000,
        ) -> tuple[list[tuple[float, float]], int]:
    """Sample non-overlapping control intervals matching target durations."""
    targets = _normalize_intervals(mask1_intervals, require_non_overlap=False)
    candidates = _normalize_intervals(mask0_intervals, require_non_overlap=True)
    if max_seed_tries <= 0:
        raise ValueError('max_seed_tries should be positive')
    for seed_offset in range(int(max_seed_tries)):
        current_seed = int(seed) + seed_offset
        rng = np.random.default_rng(current_seed)
        sampled = _sample_control_intervals_with_rng(targets, candidates, rng)
        if sampled is not None:
            return sampled, current_seed
    raise RuntimeError(
        f'Could not sample non-overlapping control intervals after {max_seed_tries} seed tries'
    )


def sample_control_intervals_by_channel(
        mask1_by_channel: Mapping[str, Sequence[tuple[float, float]]],
        mask0_by_channel: Mapping[str, Sequence[tuple[float, float]]],
        seed: int = 0,
        max_seed_tries: int = 1000,
        ) -> tuple[dict[str, list[tuple[float, float]]], int]:
    """Sample matched control intervals for each channel with one shared seed."""
    keys = list(mask1_by_channel.keys())
    if list(mask0_by_channel.keys()) != keys:
        raise ValueError('mask1_by_channel and mask0_by_channel should have matching channel keys')
    if max_seed_tries <= 0:
        raise ValueError('max_seed_tries should be positive')

    normalized_targets = {
        key: _normalize_intervals(mask1_by_channel[key], require_non_overlap=False)
        for key in keys
    }
    normalized_candidates = {
        key: _normalize_intervals(mask0_by_channel[key], require_non_overlap=True)
        for key in keys
    }

    for seed_offset in range(int(max_seed_tries)):
        current_seed = int(seed) + seed_offset
        rng = np.random.default_rng(current_seed)
        sampled_by_channel = {}
        success = True
        for key in keys:
            sampled = _sample_control_intervals_with_rng(
                normalized_targets[key],
                normalized_candidates[key],
                rng,
            )
            if sampled is None:
                success = False
                break
            sampled_by_channel[key] = sampled
        if success:
            return sampled_by_channel, current_seed

    raise RuntimeError(
        f'Could not sample non-overlapping control intervals across channels after {max_seed_tries} seed tries'
    )


def _normalize_mask(mask: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate one binary mask and return values plus normalized coordinates."""
    if not isinstance(mask, xr.DataArray):
        raise TypeError('mask should be an xarray.DataArray')
    if 'time' not in mask.dims:
        raise ValueError(f'mask should have a time dimension, got dims={mask.dims}')

    if mask.ndim == 1:
        if tuple(mask.dims) != ('time',):
            mask = mask.transpose('time')
        values = np.asarray(mask.values)
        time_values = _validate_time_values(mask.coords['time'].values)
        _validate_mask_values(values)
        return values.astype(np.uint8, copy=False), time_values, None

    if mask.ndim == 2:
        if 'y' not in mask.dims:
            raise ValueError(f'2-D masks should have y and time dims, got dims={mask.dims}')
        mask = mask.transpose('y', 'time')
        values = np.asarray(mask.values)
        time_values = _validate_time_values(mask.coords['time'].values)
        y_values = np.asarray(mask.coords['y'].values, dtype=float)
        _validate_mask_values(values)
        return values.astype(np.uint8, copy=False), time_values, y_values

    raise ValueError(f'mask should be 1-D or 2-D, got ndim={mask.ndim}')


def _validate_time_values(time_values) -> np.ndarray:
    """Validate one monotonic time coordinate."""
    time_values = np.asarray(time_values, dtype=float)
    if time_values.ndim != 1:
        raise ValueError('time coordinate should be one-dimensional')
    if time_values.size == 0:
        raise ValueError('time coordinate should not be empty')
    if time_values.size > 1 and not np.all(np.diff(time_values) > 0):
        raise ValueError('time coordinate should be strictly increasing')
    return time_values


def _validate_mask_values(values: np.ndarray) -> None:
    """Validate that one mask contains only 0/1 values."""
    unique_values = np.unique(np.asarray(values))
    if not np.all(np.isin(unique_values, [0, 1, False, True])):
        raise ValueError(f'mask should contain only 0/1 values, got {unique_values.tolist()}')


def _extract_value_intervals(
        values: np.ndarray,
        time_values: np.ndarray,
        target: int,
        ) -> list[tuple[float, float]]:
    """Extract contiguous intervals for one mask value."""
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 1:
        raise ValueError('values should be one-dimensional for interval extraction')
    match = values == int(target)
    if not np.any(match):
        return []

    padded = np.pad(match.astype(np.int8), (1, 1), constant_values=0)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [
        (float(time_values[i0]), float(time_values[i1]))
        for i0, i1 in zip(starts, ends)
    ]


def _format_y_key(y_value: float) -> str:
    """Format one y coordinate into the public channel-key form."""
    return f'y_{float(y_value):g}'


def _normalize_intervals(
        intervals: Sequence[tuple[float, float]],
        require_non_overlap: bool,
        ) -> list[tuple[float, float]]:
    """Validate and sort one interval list."""
    normalized = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(f'Expected 2-item intervals, got {interval!r}')
        start, end = [float(value) for value in interval]
        if end < start:
            raise ValueError(f'Interval end should be >= start, got {interval!r}')
        normalized.append((start, end))
    if require_non_overlap:
        normalized.sort(key=lambda item: (item[0], item[1]))
        for left, right in zip(normalized[:-1], normalized[1:]):
            if _intervals_intersect(left, right):
                raise ValueError(f'Intervals should not overlap, got {left!r} and {right!r}')
    return normalized


def _sample_control_intervals_with_rng(
        target_intervals: list[tuple[float, float]],
        candidate_intervals: list[tuple[float, float]],
        rng: np.random.Generator,
        ) -> list[tuple[float, float]] | None:
    """Try to sample one full control-interval set with one RNG state."""
    if not target_intervals:
        return []

    requests = [
        (index, interval[1] - interval[0])
        for index, interval in enumerate(target_intervals)
    ]
    requests.sort(key=lambda item: (-item[1], item[0]))

    available = list(candidate_intervals)
    sampled = [None] * len(target_intervals)
    for original_index, duration in requests:
        eligible = [
            interval for interval in available
            if _interval_duration(interval) >= duration
        ]
        if not eligible:
            return None
        chosen_interval = eligible[int(rng.integers(len(eligible)))]
        start_min = chosen_interval[0]
        start_max = chosen_interval[1] - duration
        if start_max < start_min:
            return None
        if np.isclose(start_max, start_min):
            start = start_min
        else:
            start = float(rng.uniform(start_min, start_max))
        end = start + duration
        placed = (float(start), float(end))
        sampled[original_index] = placed
        available = _subtract_interval_list(available, placed)
    return sampled


def _interval_duration(interval: tuple[float, float]) -> float:
    """Return one interval duration under inclusive endpoint semantics."""
    return float(interval[1] - interval[0])


def _subtract_interval_list(
        intervals: Sequence[tuple[float, float]],
        taken: tuple[float, float],
        ) -> list[tuple[float, float]]:
    """Subtract one closed interval from a sorted non-overlapping interval list."""
    result = []
    for interval in intervals:
        if not _intervals_intersect(interval, taken):
            result.append(interval)
            continue
        left_start, left_end = interval[0], np.nextafter(taken[0], -np.inf)
        right_start, right_end = np.nextafter(taken[1], np.inf), interval[1]
        if left_end >= left_start:
            result.append((left_start, float(left_end)))
        if right_end >= right_start:
            result.append((float(right_start), right_end))
    return result


def _intervals_intersect(
        left: tuple[float, float],
        right: tuple[float, float],
        ) -> bool:
    """Return whether two closed intervals intersect."""
    return max(left[0], right[0]) <= min(left[1], right[1])
