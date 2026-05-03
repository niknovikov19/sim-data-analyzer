"""Compatibility wrapper for the band-event OEvent analysis script."""

from __future__ import annotations

import sys
from pathlib import Path


DIR_PACKAGE = Path(__file__).resolve().parents[2]
if str(DIR_PACKAGE) not in sys.path:
    sys.path.insert(0, str(DIR_PACKAGE))

from dev_scratch.analysis.oevent_band_events import main


if __name__ == '__main__':
    main()
