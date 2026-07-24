import numpy as np
from matplotlib import pyplot as plt


def get_default_axes(nb_rows: int = 1, nb_cols: int = 1) -> plt.Axes | np.ndarray:
    """Return a default-formatted ``Axes`` on a new figure.

    By default, OSEkit plots on wide, borderless figures.
    This method set the default figure and axes parameters.

    Returns
    -------
    plt.Axes | np.ndarray:
        The default ``Axes`` on a new figure.
        If nb_rows > 1, returns a np.ndarray of plt.Axes.

    """
    # Legacy OSEkit behaviour.
    _, axs = plt.subplots(
        nrows=nb_rows,
        ncols=nb_cols,
        figsize=(1813 / 100, 512 / 100),
        dpi=100,
    )

    # Skim through both 1D and 2D ax arrays
    axs_array = axs if type(axs) is np.ndarray else [axs]
    for outer in axs_array:
        inner_axs_array = outer if type(outer) is np.ndarray else [outer]
        for ax in inner_axs_array:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
    plt.axis("off")
    plt.subplots_adjust(
        top=1,
        bottom=0,
        right=1,
        left=0,
        hspace=0,
        wspace=0,
    )
    return axs
