"""APLOSE-related utilities."""

# Official APLOSE annotation color palette.
APLOSE_COLORS = {
    "teal": "#00B1B9",
    "plum": "#A23B72",
    "orange": "#F18F01",
    "red": "#C73E1D",
    "brown": "#BB7E5D",
    "yellow": "#EAC435",
    "lime": "#98CE00",
    "purple": "#6761A8",
    "green": "#009B72",
    "black": "#2A2D34",
}

APLOSE_PALETTE = tuple(APLOSE_COLORS.values())


def get_aplose_color(index: int) -> str:
    """Return an APLOSE color, cycling through the palette.

    Parameters
    ----------
    index: int
        Index of the color to return.

    Returns
    -------
    str:
        Hex value of the APLOSE color.

    """
    return APLOSE_PALETTE[index % len(APLOSE_PALETTE)]
