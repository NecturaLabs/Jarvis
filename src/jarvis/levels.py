"""The sound levels of one captured request, logged beside what was heard.

A misheard request is easier to explain with its levels: a quiet or clipped voice, a loud room, or
a first sound cut off before the capture began. Only numbers are measured and logged, never the
sound itself, so the log says how a request sounded without holding it.

Levels are taken over 20 ms windows: the room's floor is the quietest tenth of them and the
speech level the loudest tenth, so a pause inside the request counts toward neither.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

FULL_SCALE = 32768.0
WINDOW = 320  # 20 ms at 16 kHz
# A window at this level holds less than one step of a 16-bit sample: digital silence.
_SILENT_DBFS = 20 * math.log10(0.5 / FULL_SCALE)


@dataclass(frozen=True, slots=True)
class Levels:
    """How loud one capture was, in dB relative to full scale (dBFS)."""

    peak_dbfs: float
    speech_dbfs: float  # the loudest tenth of its 20 ms windows
    floor_dbfs: float  # the quietest tenth: the room between and around the words
    start_db: float  # its first window over the floor; high when speech was already under way
    clipped: float  # the share of samples at full scale

    @property
    def snr_db(self) -> float:
        """How far the speech stands above the room."""
        return self.speech_dbfs - self.floor_dbfs

    def summary(self) -> str:
        """Say the levels in one log line."""
        return (
            f"peak {self.peak_dbfs:.1f} dBFS, speech {self.speech_dbfs:.1f} dBFS, "
            f"floor {self.floor_dbfs:.1f} dBFS (SNR {self.snr_db:.1f} dB), "
            f"start {self.start_db:+.1f} dB over the floor, clipped {self.clipped:.2%}"
        )


def _dbfs(value: float) -> float:
    return max(20 * math.log10(value / FULL_SCALE), _SILENT_DBFS) if value > 0 else _SILENT_DBFS


def levels(samples: NDArray[np.int16]) -> Levels:
    """Measure one capture of 16 kHz, 16-bit samples; an empty one is digital silence."""
    if not len(samples):
        return Levels(_SILENT_DBFS, _SILENT_DBFS, _SILENT_DBFS, 0.0, 0.0)
    wide = samples.astype(np.float64)
    whole = len(wide) // WINDOW * WINDOW or len(wide)
    windows = wide[:whole].reshape(-1, min(WINDOW, whole))
    rms = np.sqrt(np.mean(windows**2, axis=1))
    floor = _dbfs(float(np.percentile(rms, 10)))
    return Levels(
        peak_dbfs=_dbfs(float(np.max(np.abs(wide)))),
        speech_dbfs=_dbfs(float(np.percentile(rms, 90))),
        floor_dbfs=floor,
        start_db=_dbfs(float(rms[0])) - floor,
        clipped=float(np.mean(np.abs(wide) >= FULL_SCALE - 1)),
    )
