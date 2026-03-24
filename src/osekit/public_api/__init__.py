import warnings

from osekit import public

warnings.warn(
    "\nosekit.public_api is deprecated and will be removed in a future version. "
    "Use osekit.public instead.",
    FutureWarning,
    stacklevel=2,
)

__path__ = public.__path__
