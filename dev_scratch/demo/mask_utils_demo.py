"""Demo for extracting and sampling intervals from one saved binary mask."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DIR_PACKAGE = Path(__file__).resolve().parents[2]
DIR_REPO = DIR_PACKAGE.parent
if str(DIR_REPO) not in sys.path:
    sys.path.insert(0, str(DIR_REPO))

from sim_data_analyzer.mask_utils import (
    extract_mask_intervals,
    sample_control_intervals,
    sample_control_intervals_by_channel,
)
from sim_data_analyzer.xr_io import load_xr


FPATH_MASK = (
    DIR_PACKAGE
    / 'dev_scratch'
    / 'data_proc'
    / 'a1_lfp_30s_0'
    / 'oevent_mask'
    / 'exp1__csd__alpha__t_5_30__y_0_3000__oevcfg_default.nc'
)
DIRPATH_OUT = DIR_PACKAGE / 'dev_scratch' / 'results' / 'a1_lfp_30s_0' / 'demo' / 'mask_utils_demo'
SINGLE_CHANNEL_KEY = 'y_600'
SAMPLE_SEED = 0
MAX_SEED_TRIES = 1000
PNG_NAME = 'mask_utils_demo.png'


def _format_interval(interval: tuple[float, float]) -> list[float]:
    """Convert one interval tuple into a JSON-friendly list."""
    return [float(interval[0]), float(interval[1])]


def _format_interval_list(intervals) -> list[list[float]]:
    """Convert one interval list into JSON-friendly nested lists."""
    return [_format_interval(interval) for interval in intervals]


def _format_interval_dict(intervals_by_channel) -> dict[str, list[list[float]]]:
    """Convert one channel-indexed interval dict into JSON-friendly lists."""
    return {
        str(channel_key): _format_interval_list(intervals)
        for channel_key, intervals in intervals_by_channel.items()
    }


def _count_intervals(intervals_by_channel) -> dict[str, int]:
    """Count intervals for each channel in one interval dict."""
    return {
        str(channel_key): len(intervals)
        for channel_key, intervals in intervals_by_channel.items()
    }


def _parse_y_key(channel_key: str) -> float:
    """Convert one public channel key like y_600 back to a numeric depth."""
    if not str(channel_key).startswith('y_'):
        raise ValueError(f'Expected a y_* channel key, got {channel_key!r}')
    return float(str(channel_key)[2:])


def _choose_single_channel(intervals_by_channel: dict[str, list[tuple[float, float]]]) -> str:
    """Pick one valid channel key for the single-channel demo path."""
    if SINGLE_CHANNEL_KEY in intervals_by_channel:
        return SINGLE_CHANNEL_KEY
    for channel_key, intervals in intervals_by_channel.items():
        if intervals:
            return channel_key
    raise ValueError('Could not find any channel with mask==1 intervals')


def _make_demo_png(
        fpath_out: Path,
        mask,
        channel_key: str,
        burst_intervals: list[tuple[float, float]],
        control_intervals: list[tuple[float, float]],
        multi_controls: dict[str, list[tuple[float, float]]],
        single_seed: int,
        ) -> None:
    """Render one two-panel summary PNG for the mask-utils demo."""
    # Build the figure around one representative channel plus the full mask image.
    y_value = _parse_y_key(channel_key)
    time_values = mask.time.values.astype(float)
    y_values = mask.y.values.astype(float)
    mask_values = mask.values.astype(float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [1.0, 2.6]},
        constrained_layout=True,
    )
    ax_top, ax_bottom = axes

    # Show burst and matched-control intervals as simple horizontal segments.
    lane_specs = [
        ('burst', burst_intervals, '#d62728', 1.0),
        ('matched control', control_intervals, '#1f77b4', 0.0),
    ]
    for label, intervals, color, y_lane in lane_specs:
        for start_t, end_t in intervals:
            ax_top.plot([start_t, end_t], [y_lane, y_lane], color=color, lw=6, solid_capstyle='butt')
        ax_top.plot([], [], color=color, lw=6, label=label)
    ax_top.set_yticks([0.0, 1.0], labels=['matched control', 'burst'])
    ax_top.set_ylim(-0.6, 1.6)
    ax_top.grid(True, axis='x', alpha=0.25)
    ax_top.legend(loc='upper right', frameon=False)
    ax_top.set_title(f'Representative channel {channel_key}, matched controls with seed={single_seed}')

    # Show the full saved mask with y increasing downward and the chosen channel highlighted.
    dt = float(time_values[1] - time_values[0]) if time_values.size > 1 else 1.0
    dy = float(y_values[1] - y_values[0]) if y_values.size > 1 else 1.0
    extent = [
        float(time_values[0] - 0.5 * dt),
        float(time_values[-1] + 0.5 * dt),
        float(y_values[-1] + 0.5 * dy),
        float(y_values[0] - 0.5 * dy),
    ]
    cmap = ListedColormap(['#f5f7fa', '#d62728'])
    ax_bottom.imshow(
        mask_values,
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    for control_channel_key, control_channel_intervals in multi_controls.items():
        control_y = _parse_y_key(control_channel_key)
        for start_t, end_t in control_channel_intervals:
            ax_bottom.plot(
                [start_t, end_t],
                [control_y, control_y],
                color='#1f77b4',
                lw=1.2,
                alpha=0.75,
                solid_capstyle='butt',
            )
    ax_bottom.axhline(y_value, color='#1f77b4', lw=1.5, ls='--', alpha=0.95)
    ax_bottom.set_ylabel('y (um)')
    ax_bottom.set_xlabel('time (s)')
    ax_bottom.set_title('Full binary mask with matched controls overlaid (y=0 at top)')

    fig.savefig(fpath_out, dpi=150)
    plt.close(fig)


def _write_readme(
        fpath_out: Path,
        fpath_json: Path,
        fpath_png: Path,
        channel_key: str,
        single_seed: int,
        multi_seed: int,
        n_channels: int,
        ) -> None:
    """Write one short README next to the exported demo JSON."""
    lines = [
        '# Mask utils demo',
        '',
        f'- Source mask: `{FPATH_MASK}`',
        f'- Exported JSON: `{fpath_json.name}`',
        f'- Exported PNG: `{fpath_png.name}`',
        f'- Single-channel demo key: `{channel_key}`',
        f'- Single-channel successful seed: `{single_seed}`',
        f'- Multi-channel successful seed: `{multi_seed}`',
        f'- Channels in multi-channel demo: `{n_channels}`',
    ]
    fpath_out.write_text('\n'.join(lines) + '\n')


def main() -> None:
    """Run the interval-extraction and control-sampling demo on one saved mask."""
    # Load the saved mask data and validate its shape through the shared xarray helper.
    mask = load_xr(FPATH_MASK, data_type='dataarray', load=True)
    if 'time' not in mask.dims:
        raise ValueError(f'Expected a time dimension in the mask, got dims={mask.dims}')

    # Extract mask==1 and mask==0 intervals for every available channel.
    intervals = extract_mask_intervals(mask)
    mask1_by_channel = intervals['mask_1']
    mask0_by_channel = intervals['mask_0']
    if not isinstance(mask1_by_channel, dict) or not isinstance(mask0_by_channel, dict):
        raise ValueError('Expected a multi-channel y x time mask for this demo')

    # Run the single-channel sampling path on one representative depth.
    channel_key = _choose_single_channel(mask1_by_channel)
    single_controls, single_seed = sample_control_intervals(
        mask1_by_channel[channel_key],
        mask0_by_channel[channel_key],
        seed=SAMPLE_SEED,
        max_seed_tries=MAX_SEED_TRIES,
    )

    # Run the shared-seed multi-channel sampling path across the full mask.
    multi_controls, multi_seed = sample_control_intervals_by_channel(
        mask1_by_channel,
        mask0_by_channel,
        seed=SAMPLE_SEED,
        max_seed_tries=MAX_SEED_TRIES,
    )

    # Save a compact JSON artifact that shows both extraction and matched controls.
    DIRPATH_OUT.mkdir(parents=True, exist_ok=True)
    fpath_json = DIRPATH_OUT / 'mask_utils_demo.json'
    payload = {
        'source_mask': str(FPATH_MASK),
        'mask_shape': list(mask.shape),
        'mask_dims': list(mask.dims),
        'time_range_s': [float(mask.time.values[0]), float(mask.time.values[-1])],
        'y_values': [float(y_value) for y_value in mask.y.values.tolist()],
        'single_channel_demo': {
            'channel_key': channel_key,
            'successful_seed': int(single_seed),
            'mask_1_intervals': _format_interval_list(mask1_by_channel[channel_key]),
            'mask_0_intervals': _format_interval_list(mask0_by_channel[channel_key]),
            'sampled_control_intervals': _format_interval_list(single_controls),
        },
        'multi_channel_demo': {
            'successful_seed': int(multi_seed),
            'mask_1_counts': _count_intervals(mask1_by_channel),
            'mask_0_counts': _count_intervals(mask0_by_channel),
            'sampled_control_counts': _count_intervals(multi_controls),
            'mask_1_intervals': _format_interval_dict(mask1_by_channel),
            'mask_0_intervals': _format_interval_dict(mask0_by_channel),
            'sampled_control_intervals': _format_interval_dict(multi_controls),
        },
    }
    fpath_json.write_text(json.dumps(payload, indent=2) + '\n')

    # Save one compact PNG that shows both the single-channel and full-mask views.
    fpath_png = DIRPATH_OUT / PNG_NAME
    _make_demo_png(
        fpath_png,
        mask=mask,
        channel_key=channel_key,
        burst_intervals=mask1_by_channel[channel_key],
        control_intervals=single_controls,
        multi_controls=multi_controls,
        single_seed=single_seed,
    )

    # Save a short README that points at the generated files and key settings.
    fpath_readme = DIRPATH_OUT / 'README.md'
    _write_readme(
        fpath_readme,
        fpath_json=fpath_json,
        fpath_png=fpath_png,
        channel_key=channel_key,
        single_seed=single_seed,
        multi_seed=multi_seed,
        n_channels=len(mask1_by_channel),
    )

    # Print a compact terminal summary for the demo run.
    print(f'Loaded mask: {FPATH_MASK}')
    print(f'Channels: {len(mask1_by_channel)}, time samples: {mask.sizes["time"]}')
    print(
        f'Single-channel demo: {channel_key}, '
        f'{len(mask1_by_channel[channel_key])} burst intervals, seed={single_seed}'
    )
    print(f'Multi-channel demo: {len(multi_controls)} channels, seed={multi_seed}')
    print(f'Saved JSON: {fpath_json}')
    print(f'Saved PNG: {fpath_png}')
    print(f'Saved README: {fpath_readme}')


if __name__ == '__main__':
    main()
