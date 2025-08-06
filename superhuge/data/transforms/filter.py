from typing import Any, Sequence

import numpy as np
from scipy.signal import butter, filtfilt

from .abc import Transform


class Filter(Transform):
    """Applies a filter to EEG data."""

    def __init__(
        self,
        Wn: float | Sequence[float],
        fs: float,
        btype: str,
        order: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(Wn, float):
            assert 0 < Wn < 1, f"Wn must be between 0 and 1, got {Wn}."
            assert btype in [
                "lowpass",
                "highpass",
            ], f"btype must be 'lowpass' or 'highpass', got {btype}."
            self.Wn = Wn / (fs / 2)
        elif isinstance(Wn, Sequence):
            assert all(
                0 < wn < 1 for wn in Wn
            ), f"All elements in Wn must be between 0 and 1, got {Wn}."
            assert btype in [
                "bandpass",
                "bandstop",
            ], f"btype must be 'bandpass' or 'bandstop', got {btype}."
            self.Wn = [wn / (fs / 2) for wn in Wn]
        else:
            raise TypeError(
                f"Wn must be a float or a sequence of floats, got {type(Wn)}."
            )
        self.fs = fs
        self.order = order
        self.btype = btype

        result = butter(self.order, self.Wn, btype=self.btype)
        if result is None or len(result) != 2:
            raise ValueError("Butter function did not return expected coefficients.")
        self.b, self.a = result

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x)
        if self.roll():
            x = filtfilt(self.b, self.a, x, axis=0).copy()
        return x, *args
