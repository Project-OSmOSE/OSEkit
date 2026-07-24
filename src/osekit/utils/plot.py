import numpy as np
from matplotlib import pyplot as plt


def get_default_axes(nb_rows: int = 1) -> plt.Axes:
    """Return a default-formatted ``Axes`` on a new figure.

    By default, OSEkit plots on wide, borderless figures.
    This method set the default figure and axes parameters.

    Returns
    -------
    plt.Axes:
        The default ``Axes`` on a new figure.

    """
    # Legacy OSEkit behaviour.
    _, axs = plt.subplots(
        nrows=nb_rows,
        ncols=1,
        figsize=(1813 / 100, 512 / 100),
        dpi=100,
    )

    axs_array = axs if type(axs) is np.ndarray else [axs]

    for ax in axs_array:
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
