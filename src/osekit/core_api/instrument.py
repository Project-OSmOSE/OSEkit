"""The instrument class represent the audio acquisition chain.

It embeds the technical properties of the hydrophone,
the gain applied to the measured signal etc.
"""

from __future__ import annotations

import numpy as np


class Instrument:
    """Represent the audio acquisition chain.

    It embeds the technical properties of the hydrophone,
    the gain applied to the measured signal etc.
    """

    P_REF = 1e-6

    def __init__(
        self,
        sensitivity: float = 1.0,
        peak_voltage: float = 1.0,
        gain_db: float = 0.0,
        end_to_end_db: float | None = None,
    ) -> None:
        """Initialize an ``Instrument`` object.

        Parameters
        ----------
        sensitivity: float
            Sensitivity (in Volt per Pascal) of the sensor.
            Defaulted to ``1.``.
        peak_voltage: float
            Voltage that leads to a ``0 dB FS`` signal.
            Defaulted to ``1 V``.
        gain_db: float
            Total gain (in dB) of the chain of amplifiers.
            Defaulted to ``0 dB``.
        end_to_end_db: float | None
            End-to-end calibration value, as given by e.g. SoundTrap datasheets.
            If provided, it will overwrite the former parameters.

        """
        self.sensitivity = sensitivity
        self.peak_voltage = peak_voltage
        self.gain_db = gain_db
        self.end_to_end_db = end_to_end_db

    @property
    def end_to_end(self) -> float:
        """Total ratio between digital signal value and aoustic pressure."""
        if self._end_to_end is not None:
            return self._end_to_end
        return self.peak_voltage / (self.sensitivity * self.gain)

    @end_to_end.setter
    def end_to_end(self, value: float) -> None:
        self._end_to_end = value

    @property
    def end_to_end_db(self) -> float:
        """Total ratio between digital signal value and acoustic pressure level (re ``1uPa``)."""
        return 20 * np.log10(self.end_to_end / self.P_REF)

    @end_to_end_db.setter
    def end_to_end_db(self, value: float) -> None:
        if value is None:
            self._end_to_end = None
            return

        self._end_to_end = self.P_REF * 10 ** (value / 20)

    @property
    def sensitivity(self) -> float:
        """Sensitivity of the sensor, in ``V/Pa``."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = value

    @property
    def peak_voltage(self) -> float:
        """Voltage that leads to a ``0 dB FS`` signal."""
        return self._peak_voltage

    @peak_voltage.setter
    def peak_voltage(self, value: float) -> None:
        self._peak_voltage = value

    @property
    def gain(self) -> float:
        """Total voltage ratio of the chain of amplifiers."""
        return self._g

    @gain.setter
    def gain(self, value: float) -> None:
        self._g = value

    @property
    def gain_db(self) -> float:
        """Total gain (in ``dB``) of the chain of amplifiers."""
        return 20 * np.log10(self.gain)

    @gain_db.setter
    def gain_db(self, value: float) -> None:
        self._g = 10 ** (value / 20)

    def n_to_p(self, digit_value: float) -> float:
        """Convert raw digital data to acoustic pressure (in ``Pa``).

        Parameters
        ----------
        digit_value: float
            Raw digital value.

        Returns
        -------
        float:
            Acoustic pressure (in ``Pa``).

        """
        return digit_value * self.end_to_end

    def to_dict(self) -> dict:
        """Return a dictionary that is used for serialization."""
        return {
            "sensitivity": self.sensitivity,
            "peak_voltage": self.peak_voltage,
            "gain_db": self.gain_db,
            "end_to_end_db": self.end_to_end_db,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> Instrument | None:
        """Deserialize an ``Instrument`` from a dictionary."""
        if data is None:
            return None
        return cls(**data)
