import warnings

from osekit import core

warnings.warn(
    "\nosekit.core_api is deprecated and will be removed in a future version. "
    "Use osekit.core instead.",
    FutureWarning,
    stacklevel=2,
)

__path__ = core.__path__
