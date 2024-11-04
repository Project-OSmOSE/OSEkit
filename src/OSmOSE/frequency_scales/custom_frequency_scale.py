from dataclasses import dataclass
from typing import Tuple

from OSmOSE.frequency_scales.abstract_frequency_scale import AbstractFrequencyScale


@dataclass
class CustomFrequencyScale(AbstractFrequencyScale):
    """Defines three frequency bands [0,f1],[f1,f2],[f2,sr/2].
    Each band is rescaled to take a given % of the display.
    Respectively stretched to coef1 % coef2 % and coef3 % of the y-axis
    for example:
        0-100 Hz : 10%
        100-8000 Hz : 80%
        8000-22050 Hz : 10%  (e.g sr = 44100 Hz)
    """

    frequencies: Tuple[int, int] = (22000, 100000)
    coefficients: Tuple[float, float, float] = (0.5, 0.2, 0.3)

    def __post_init__(self):
        assert (
            all(f >= 0 for f in self.frequencies) and self.sr > 0
        ), "Frequencies should be non-negative"
        assert all(
            c >= 0 for c in self.coefficients
        ), "Coefficients should be non-negative"
        assert (
            self.frequencies[0] <= self.frequencies[1] <= 0.5 * self.sr
        ), "Frequencies for custom scale error - should be sorted f1<f2<0.5*sr"
        assert (
            sum(self.coefficients) == 1
        ), "Coefficients for custom scale error, - should sum to 1"
        assert self.coefficients[0] != 0, "Coef1 cannot be zero"

    def map_freq2scale(self, freq):
        f1, f2 = self.frequencies
        c1, c2, c3 = self.coefficients
        if freq <= f1:
            return freq / f1 * c1
        if freq <= f2:
            return c1 + (freq - f1) / (f2 - f1) * c2 if f1 != f2 else c1
        return (
            c1 + c2 + (freq - f2) / (0.5 * self.sr - f2) * c3
            if (f2 != 0.5 * self.sr or c3 == 0)
            else c1 + c2
        )

    def map_scale2freq(self, custom_freq):
        f1, f2 = self.frequencies
        c1, c2, c3 = self.coefficients
        if custom_freq <= c1:
            return custom_freq / c1 * f1
        if custom_freq <= c1 + c2:
            return f1 + (custom_freq - c1) / c2 * (f2 - f1) if c2 != 0 else f1
        return (
            f2 + (custom_freq - c1 - c2) / c3 * (0.5 * self.sr - f2)
            if c3 != 0
            else 0.5 * self.sr
        )
