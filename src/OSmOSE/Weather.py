"""
Class for wind speed estimation using UPA
Works for regression models (user defined or from OSmOSE) and one layer LSTM
author : Anatole Gros-Martial
sponsored by MINKE
"""

import torch
from torch import nn, tensor, utils, device, cuda, optim, long, save
from OSmOSE.config_weather import empirical
from OSmOSE.config import *
from OSmOSE.Auxiliary import Auxiliary
from OSmOSE.utils import deep_learning_utils as dl_utils
from OSmOSE.utils.timestamp_utils import *
import numpy as np
from typing import Union, Tuple, List
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sklearn.metrics as metrics
import pandas as pd
from glob import glob
from tqdm import tqdm


def get_beaufort(x):
    return next(
        (
            i
            for i, limit in enumerate(
                [0.3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
            )
            if x < limit
        ),
        12,
    )


class Weather(Auxiliary):

    def __init__(
        self,
        dataset_path: str,
        method: str = None,
        ground_truth: str = "era",
        weather_params: dict = None,
        *,
        gps_coordinates: Union[str, List, Tuple, bool] = True,
        depth: Union[str, int, bool] = True,
        owner_group: str = None,
        batch_number: int = 5,
        local: bool = True,
        era: Union[str, bool] = False,
        annotation: Union[dict, bool] = False,
        other: dict = None,
    ):
        """
        Parameters:
                dataset_path (str): The path to the dataset.
                method (str) : Method or more generally processing pipeline and model used to estimate wind speed. Found in config_weather.py.
                ground_truth (str) : Column name from auxiliary data that stores wind speed data that will be used as ground truth.
                dataset_sr (int, optional): The dataset sampling rate. Default is None.
                weather_params (dict) : Enter your own parameters for wind estimation. Will be taken into account if method = None.
                        - frequency : 'int'
                        - samplerate : 'int'
                        - preprocessing
                                - nfft : 'int'
                                - window_size : 'int'
                                - spectro_duration : 'int'
                                - window : 'str'
                                - overlap : 'float'
                        - function : func
                        - averaging_duration : 'int'
                        - parameters
                                - a : 'float'
                                - b : 'float'
                                - ...
                analysis_params (dict, optional): Additional analysis parameters. Default is None.
                gps_coordinates (str, list, tuple, bool, optional): Whether GPS data is included. Default is True. If string, enter the filename (csv) where gps data is stored.
                depth (str, int, bool, optional): Whether depth data is included. Default is True. If string, enter the filename (csv) where depth data is stored.
                era (bool, optional): Whether era data is included. Default is False. If string, enter the filename (Network Common Data Form) where era data is stored.
                annotation (bool, optional): Annotation data is included. Dictionary containing key (column name of annotation data) and absolute path of csv file where annotation data is stored. Default is False.
                other (dict, optional): Additional data (csv format) to join to acoustic data. Key is name of data (column name) to join to acoustic dataset, value is the absolute path where to find the csv. Default is None.
        """

        if method:
            self.method = empirical[method]
        else:
            self.method = weather_params

        super().__init__(
            dataset_path,
            gps_coordinates=gps_coordinates,
            depth=depth,
            dataset_sr=self.method["samplerate"],
            owner_group=owner_group,
            analysis_params=self.method["preprocessing"],
            batch_number=batch_number,
            local=local,
            era=era,
            annotation=annotation,
            other=other,
        )

        self.ground_truth = ground_truth
        if self.ground_truth not in self.df:
            print(
                f"Ground truth data '{self.ground_truth}' was not found in joined dataframe.\nPlease call the correct joining method or automatic_join()"
            )
        self.popt, self.wind_model_stats = {}, {}
        self.df.columns = [
            int(col) if col.isdigit() else col for col in self.df.columns
        ]

    def beaufort(self):
        self.df["classes"] = self.df[self.ground_truth].apply(get_beaufort)

    def __str__(self):
        if "wind_model_stats" in dir(self):
            print("Model has been trained with following parameters : \n")
            for key, value in self.method.items():
                print(f"{key} : {value}")
            print("ground truth : ", self.ground_truth)
            print("-----------\nThe model has the following performance :")
            for key, value in self.wind_model_stats.items():
                print(f"{key} : {value}")
            return "You can plot your estimation using the plot_estimation() method"
        else:
            print(
                "Model has not been fitted to any data yet.\nWill be fitted with following parameters : \n"
            )
            for key, value in self.method.items():
                print(f"{key:<{6}} : {value}")
            return "To fit your model, please call skf_fit() for example"

    def median_filtering(self, kernel_size=5):
        """
        Whether or not to apply scipy's median filtering to data
        """
        self.df["filtered"] = medfilt(
            self.df[self.method["frequency"]], kernel_size=kernel_size
        )
        self.method["frequency"] = "filtered"

    def temporal_fit(self, **kwargs):
        default = {"split": 0.8, "scaling_factor": 0.2, "maxfev": 25000}
        params = {**default, **kwargs}
        if "bounds" not in params.keys():
            params["bounds"] = np.hstack(
                (
                    np.array(
                        [
                            [
                                value - params["scaling_factor"] * abs(value),
                                value + params["scaling_factor"] * abs(value),
                            ]
                            for value in self.method["parameters"].values()
                        ]
                    ).T,
                    [[-np.inf], [np.inf]],
                )
            )
        self.df["temporal_estimation"] = np.nan
        trainset = self.df.iloc[: int(params["split"] * len(self.df))].dropna(
            subset=[self.method["frequency"], self.ground_truth]
        )
        testset = self.df.iloc[int(params["split"] * len(self.df)) :].dropna(
            subset=[self.method["frequency"], self.ground_truth]
        )
        popt, popv = curve_fit(
            self.method["function"],
            trainset[self.method["frequency"]].to_numpy(),
            trainset[self.ground_truth].to_numpy(),
            bounds=params["bounds"],
            maxfev=params["maxfev"],
        )
        estimation = self.method["function"](
            testset[self.method["frequency"]].to_numpy(), *popt
        )
        mae = metrics.mean_absolute_error(testset[self.ground_truth], estimation)
        rmse = metrics.root_mean_squared_error(testset[self.ground_truth], estimation)
        r2 = metrics.r2_score(testset[self.ground_truth], estimation)
        var = np.var(abs(testset[self.ground_truth]) - abs(estimation))
        std = np.std(abs(testset[self.ground_truth]) - abs(estimation))
        self.df.loc[testset.index, "temporal_estimation"] = estimation
        self.popt["temporal_fit"] = popt
        self.wind_model_stats.update(
            {
                "temporal_mae": mae,
                "temporal_rmse": rmse,
                "temporal_r2": r2,
                "temporal_var": var,
                "temporal_std": std,
            }
        )

    def skf_fit(self, **kwargs):
        """
        Parameters :
                n_splits: Number of stratified K folds used for training, defaults to 5
                scaling_factor: Percentage of variability around initial parameters, defaults to 0.2
        """
        default = {"n_splits": 5, "scaling_factor": 0.2, "maxfev": 25000}
        params = {**default, **kwargs}
        if "bounds" not in params.keys():
            params["bounds"] = np.hstack(
                (
                    np.array(
                        [
                            [
                                value - params["scaling_factor"] * abs(value),
                                value + params["scaling_factor"] * abs(value),
                            ]
                            for value in self.method["parameters"].values()
                        ]
                    ).T,
                    [[-np.inf], [np.inf]],
                )
            )
        popt_tot, popv_tot = [], []
        mae, rmse, r2, var, std = [], [], [], [], []
        self.df["skf_estimation"] = np.nan
        skf = StratifiedKFold(n_splits=params["n_splits"])
        for i, (train_index, test_index) in enumerate(
            skf.split(self.df[self.method["frequency"]], self.df.classes)
        ):
            trainset = self.df.iloc[train_index].dropna(
                subset=[self.method["frequency"], self.ground_truth]
            )
            testset = self.df.iloc[test_index].dropna(
                subset=[self.method["frequency"], self.ground_truth]
            )
            popt, popv = curve_fit(
                self.method["function"],
                trainset[self.method["frequency"]].to_numpy(),
                trainset[self.ground_truth].to_numpy(),
                bounds=params["bounds"],
                maxfev=params["maxfev"],
            )
            popt_tot.append(popt)
            popv_tot.append(popv)
            estimation = self.method["function"](
                testset[self.method["frequency"]].to_numpy(), *popt
            )
            mae.append(
                metrics.mean_absolute_error(testset[self.ground_truth], estimation)
            )
            rmse.append(
                metrics.root_mean_squared_error(testset[self.ground_truth], estimation)
            )
            r2.append(metrics.r2_score(testset[self.ground_truth], estimation))
            var.append(np.var(abs(testset[self.ground_truth]) - abs(estimation)))
            std.append(np.std(abs(testset[self.ground_truth]) - abs(estimation)))
            self.df.loc[testset.index, "skf_estimation"] = estimation
        self.popt["skf_fit"] = np.mean(popt_tot, axis=0)
        self.wind_model_stats.update(
            {
                "skf_mae": np.mean(mae),
                "skf_rmse": np.mean(rmse),
                "skf_r2": np.mean(r2),
                "skf_var": np.mean(var),
                "skf_std": np.mean(std),
            }
        )

    def lstm_fit(self, seq_length=10, **kwargs):
        default = {
            "learning_rate": 0.001,
            "epochs": 75,
            "weight_decay": 0.000,
            "hidden_dim": 512,
            "n_splits": 5,
            "n_cross_validation": 1,
        }
        params = {**default, **kwargs}
        self.df["lstm_estimation"] = np.nan
        self.df.dropna(
            subset=[self.method["frequency"], self.ground_truth], inplace=True
        )
        skf = StratifiedKFold(n_splits=params["n_splits"])
        split = skf.split(self.df[self.method["frequency"]], self.df.classes)
        for i in range(min(params["n_splits"], params["n_cross_validation"])):
            # set the device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # create the model and move it to the specified device
            model = dl_utils.RNNModel(1, params["hidden_dim"], 1, 1)
            model.to(device)
            # create the loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )

            train_indices, test_indices = next(split)
            trainset, testset = self.df.loc[train_indices], self.df.loc[test_indices]
            train_loader = utils.data.DataLoader(
                dl_utils.Wind_Speed(
                    trainset,
                    self.method["frequency"],
                    self.ground_truth,
                    seq_length=seq_length,
                ),
                batch_size=64,
                shuffle=True,
            )
            test_loader = utils.data.DataLoader(
                dl_utils.Wind_Speed(
                    testset,
                    self.method["frequency"],
                    self.ground_truth,
                    seq_length=seq_length,
                ),
                batch_size=64,
                shuffle=False,
            )
            estimation = dl_utils.train_rnn(
                model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                num_epochs=params["epochs"],
                device=device,
            )
            self.df.loc[test_indices, "lstm_estimation"] = estimation

    def plot_estimation(self):

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=self.ground_truth,
                mode="markers",
                marker=dict(size=3),
                name="Ground Truth",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["estimation"],
                mode="markers",
                marker=dict(size=4),
                name="Estimation",
            )
        )
        fig.update_layout(
            title="Wind Speed and Estimation Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Weather estimation",
            width=800,
            height=600,
        )
        fig.show()

    @classmethod
    def from_joined_dataframe(
        cls,
        path,
        ground_truth,
        dataset_path,
        method=None,
        weather_params=None,
        **kwargs,
    ):
        """
        Class method to instantiate a Weather object from an already joined dataframe.

        Parameters:
        path (str): Path to the joined dataframe (CSV format).
        method (str): Optional. Method or processing pipeline to be used.
        ground_truth (str): Column name for ground truth data.
        dataset_path (str): Path to the dataset (needed to call the init method properly).
        weather_params (dict): Optional weather parameters to be passed.
        kwargs: Any additional arguments to pass to the __init__ method.
        """
        instance = cls.__new__(cls)
        instance.df = check_epoch(pd.read_csv(path))
        if method:
            instance.method = empirical[method]
        else:
            instance.method = weather_params

        instance.ground_truth = ground_truth
        if instance.ground_truth not in instance.df:
            print(
                f"Ground truth data '{instance.ground_truth}' was not found in joined dataframe.\nPlease call the correct joining method or automatic_join()"
            )
        instance.popt, instance.wind_model_stats = {}, {}
        instance.df.columns = [
            int(col) if col.isdigit() else col for col in instance.df.columns
        ]

        return instance


"""	@classmethod
	def from_joined_dataframe(cls, path, method, ground_truth):
		instance = cls.__new__(cls)
		instance.df = check_epoch(pd.read_csv(path))
		instance.method = method
		instance.ground_truth = ground_truth
		instance.popt, instance.wind_model_stats = {}, {}
		return instance"""


"""    def save_all_welch(self):
=======
class Weather:
    def __init__(
        self,
        osmose_path_dataset,
        dataset,
        time_resolution_welch,
        sample_rate_welch,
        local=True,
    ):
        self.path = Path(os.path.join(osmose_path_dataset, dataset))
        self.dataset = dataset
        self.time_resolution_welch = time_resolution_welch
        self.sample_rate_welch = sample_rate_welch

    def save_all_welch(self):
>>>>>>> origin/join
        # get metadata from sepctrogram folder
        metadata_path = next(
            self.path.joinpath(
                OSMOSE_PATH.spectrogram,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
            ).rglob("metadata.csv"),
            None,
        )
        metadata_spectrogram = pd.read_csv(metadata_path)
	
        df = pd.read_csv(
            self.path.joinpath(
                OSMOSE_PATH.processed_auxiliary,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
                "aux_data.csv",
            ),
            header=0,
        )

        path_all_welch = self.path.joinpath(
            OSMOSE_PATH.processed_auxiliary,
            str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
            "all_welch.npz",
        )

        if not path_all_welch.exists():
            LTAS = np.empty((1, int(metadata_spectrogram["nfft"][0] / 2) + 1))
            time = []
            for file_npz in tqdm(list(df["fn"].values)):
                current_matrix = np.load(file_npz, allow_pickle=True)
                LTAS = np.vstack((LTAS, current_matrix["Sxx"]))
                time.append(current_matrix["Time"])
            LTAS = LTAS[1:, :]
            Freq = current_matrix["Freq"]

            time = np.array(time)

            # flatten time, which is currently a list of arrays
            if time.ndim == 2:
                time = list(itertools.chain(*time))
            else:
                time = [
                    tt.item() for tt in time
                ]  # suprinsingly , doing simply = list(time) was droping the Timestamp dtype, to be investigated in more depth...

            np.savez(
                path_all_welch, LTAS=LTAS, time=time, Freq=Freq, allow_pickle=True
            )  # careful data not sorted here! we should save them based on dataframe df below

        else:
            print(f"Your complete welch npz is already built! move on..")

    def wind_speed_estimation(
        self,
        show_fig: bool = False,
        percentile_outliers: int = None,
        threshold_SPL: [int, list] = None,
    ):
        if not self.path.joinpath(OSMOSE_PATH.weather).exists():
            make_path(self.path.joinpath(OSMOSE_PATH.weather), mode=DPDEFAULT)

        df = pd.read_csv(
            self.path.joinpath(
                OSMOSE_PATH.processed_auxiliary,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
                "aux_data.csv",
            ),
            header=0,
        )

        feature_matrix = pd.DataFrame(
            {
                "SPL_filtered": df["SPL_filtered"],
                "InSituWIND": np.sqrt(df["interp_u10"] ** 2 + df["interp_v10"] ** 2),
            }
        )

        feature_matrix = feature_matrix[pd.notnull(feature_matrix["InSituWIND"])]

        Y_wind = feature_matrix["InSituWIND"]
        X_wind = feature_matrix["SPL_filtered"]

        # Y_categorical = pd.cut(Y_wind, [0,2.2,3.6,6,np.inf], right=False)

        x_train = X_wind.values
        y_train = Y_wind

        if percentile_outliers:
            val_outlier = np.percentile(x_train, percentile_outliers)
            y_train = y_train[x_train < val_outlier]
            x_train = x_train[x_train < val_outlier]
        if threshold_SPL:
            if type(threshold_SPL) == int:
                y_train = y_train[x_train < threshold_SPL]
                x_train = x_train[x_train < threshold_SPL]
            else:
                y_train = y_train[
                    (x_train > threshold_SPL[0]) & (x_train < threshold_SPL[1])
                ]
                x_train = x_train[
                    (x_train > threshold_SPL[0]) & (x_train < threshold_SPL[1])
                ]

        z = np.polyfit(x_train, y_train, 2)
        fit = np.poly1d(z)

        # scatter_wind_model
        my_dpi = 100
        fact_x = 0.7
        fact_y = 1
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
            constrained_layout=True,
        )
        ax.scatter(x_train, y_train)
        ax.plot(
            np.sort(x_train),
            fit(np.sort(x_train)),
            label=fit,
            color="C3",
            alpha=1,
            lw=2.5,
        )
        ax.legend([fit, ""])
        plt.xlabel("Relative SPL (dB)")
        plt.ylabel("ECMWF w10 (m/s)")
        plt.savefig(
            self.path.joinpath(OSMOSE_PATH.weather, "scatter_wind_model.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        if show_fig:
            plt.show()
        else:
            plt.close()
        print(
            f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'scatter_wind_model.png')}"
        )

        with open(
            self.path.joinpath(OSMOSE_PATH.weather, "polynomial_law.txt"), "w"
        ) as f:
            for item in z:
                f.write("%s\n" % item)

        with open(self.path.joinpath(OSMOSE_PATH.weather, "min_max.txt"), "w") as f:
            for item in [np.min(X_wind), np.max(X_wind)]:
                f.write("%s\n" % item)

        # scatter_ecmwf_model
        my_dpi = 100
        fact_x = 0.7
        fact_y = 1
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
            constrained_layout=True,
        )
        ax.scatter(y_train.values, fit(x_train))
        plt.plot(
            np.linspace(
                np.min([np.min(y_train.values), np.min(fit(x_train))]) - 1,
                np.max([np.max(y_train.values), np.max(fit(x_train))]) + 1,
                100,
            ),
            np.linspace(
                np.min([np.min(y_train.values), np.min(fit(x_train))]) - 1,
                np.max([np.max(y_train.values), np.max(fit(x_train))]) + 1,
                100,
            ),
            "k--",
        )
        ax.set_xlabel("ERA5 wind speed (m/s)")
        ax.set_ylabel("Model wind speed (m/s)")
        plt.xlim(
            np.min([np.min(y_train.values), np.min(fit(x_train))]) - 1,
            np.max([np.max(y_train.values), np.max(fit(x_train))]) + 1,
        )
        plt.ylim(
            np.min([np.min(y_train.values), np.min(fit(x_train))]) - 1,
            np.max([np.max(y_train.values), np.max(fit(x_train))]) + 1,
        )
        plt.savefig(
            self.path.joinpath(OSMOSE_PATH.weather, "scatter_ecmwf_model.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        if show_fig:
            plt.show()
        else:
            plt.close()
        print(
            f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'scatter_ecmwf_model.png')}"
        )

        # temporal_ecmwf_model
        my_dpi = 100
        fact_x = 0.7
        fact_y = 1
        fig, ax1 = plt.subplots(
            1,
            1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
            constrained_layout=True,
        )
        color = "tab:red"
        ax1.set_xlabel("samples")
        ax1.set_ylabel("wind speed (m/s)", color=color)
        ax1.plot(y_train.values, color=color)
        ax1.plot(fit(x_train), color=color, linestyle="dotted")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend(["ecmwf", "model"])
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = "tab:blue"
        ax2.set_ylabel("SPL", color=color)  # we already handled the x-label with ax1
        ax2.plot(x_train, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax1.autoscale(enable=True, axis="x", tight=True)
        plt.savefig(
            self.path.joinpath(OSMOSE_PATH.weather, "temporal_ecmwf_model.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        if show_fig:from OSmOSE.utils import make_path

            plt.show()
        else:
            plt.close()
        print(
            f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'temporal_ecmwf_model.png')}"
        )

    def append_SPL_filtered(self, freq_min: int, freq_max: int):
        # get metadata from sepctrogram folder
        metadata_path = next(
            self.path.joinpath(
                OSMOSE_PATH.spectrogram,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
            ).rglob("metadata.csv"),
            None,
        )
        metadata_spectrogram = pd.read_csv(metadata_path)

        df = pd.read_csv(
            self.path.joinpath(
                OSMOSE_PATH.processed_auxiliary,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
                "aux_data.csv",
            ),
            header=0,
        )

        SPL_filtered = []
        for npz_path in tqdm(df["fn"]):
            ltas = np.load(npz_path, allow_pickle=True)

            if freq_min != freq_max:
                pre_SPL = np.mean(
                    ltas["Sxx"][
                        0,
                        np.argmin(abs(ltas["Freq"] - freq_min)) : np.argmax(
                            abs(ltas["Freq"] - freq_max)
                        ),
                    ]
                )
            else:
                pre_SPL = np.mean(
                    ltas["Sxx"][0, np.argmin(abs(ltas["Freq"] - freq_min))]
                )

            if metadata_spectrogram["spectro_normalization"][0] == "density":
                SPL_filtered.append(10 * np.log10((pre_SPL / (1e-12)) + (1e-20)))
            if metadata_spectrogram["spectro_normalization"][0] == "spectrum":
                SPL_filtered.append(10 * np.log10(pre_SPL + (1e-20)))

        df["SPL_filtered"] = SPL_filtered
        df.to_csv(
            self.path.joinpath(
                OSMOSE_PATH.processed_auxiliary,
                str(self.time_resolution_welch) + "_" + str(self.sample_rate_welch),
                "aux_data.csv",
            ),
            index=False,
            na_rep="NaN",
        )


class benchmark_weather:
    def __init__(self, osmose_path_dataset, dataset, local=True):
        if not isinstance(dataset, list):
            print(f"Dataset should be multiple and defined within a list")
            sys.exit(0)

        self.path = Path(osmose_path_dataset)
        self.dataset = dataset

        if not self.path.joinpath(OSMOSE_PATH.weather).exists():
            make_path(self.path.joinpath(OSMOSE_PATH.weather), mode=DPDEFAULT)

    def compare_wind_speed_models(self):
        my_dpi = 80
        fact_x = 0.5
        fact_y = 0.9
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
            constrained_layout=True,
        )
        # datasetID=[]
        # for path in self.path.joinpath(OSMOSE_PATH.weather).iterdir():
        #     if path.is_dir():
        #         datasetID.append(path)

        veccol = ["r", "b", "g"]

        print("Polynom coefs:")

        ct = 0
        for dd in self.dataset:
            f = open(
                self.path.joinpath(dd, OSMOSE_PATH.weather, "polynomial_law.txt"), "r"
            )
            xx = f.read()
            ll = [float(x) for x in xx.split("\n")[:-1]]

            p = np.poly1d(ll)

            print(
                "-",
                dd,
                " : ",
                "{:.3f}".format(p[0]),
                "/ {:.3f}".format(p[1]),
                "/ {:.3f}".format(p[2]),
            )

            x = np.arange(-20, 0, 0.1)
            y = p(x)
            plt.plot(x, y, c=veccol[ct])

            ct += 1

        plt.xlabel("Relative SPL (dB)")
        plt.ylabel("Estimated wind speed (m/s)")
        plt.legend(self.dataset)

        plt.savefig(
            self.path.joinpath(OSMOSE_PATH.weather, "compare_wind_model.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
"""
