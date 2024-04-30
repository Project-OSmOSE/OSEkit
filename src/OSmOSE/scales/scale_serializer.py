from OSmOSE.scales.custom_scale import CustomScale


class ScaleSerializer:
    def __init__(self):
        self.configurations = {
            "porp_delph": (
                CustomScale,
                {"frequencies": (22000, 100000), "coefficients": (0.5, 0.2, 0.3)},
            ),
            "Audible": (
                CustomScale,
                {"frequencies": (22000, 100000), "coefficients": (1, 0, 0)},
            ),
            "Dual_LF_HF": (
                CustomScale,
                {"frequencies": (22000, 100000), "coefficients": (0.5, 0, 0.5)},
            ),
            # Add more configurations here
        }

    def get_scale(self, config_name, sr):
        if config_name in self.configurations:
            scale_class, kwargs = self.configurations[config_name]
            print(
                f"The y-scale used for spectrogram generation is {config_name} with sampling rate {sr}"
            )
            return scale_class(sr=sr, **kwargs)
        else:
            raise ValueError(
                f"No configuration found for {config_name}, accepted values are {self.configurations.keys()}"
            )
