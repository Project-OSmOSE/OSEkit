"""Logging context used by util functions, settable using a context manager.

The OSmOSE package instantiates a LoggingContext on initialize in the config module.
Utils functions log records to this ``LoggingContext.logger`` logger.
The global logger can be replaced with a context manager:

>>> from osekit.config import global_logging_context as glc
>>> import logging
>>>
>>> @glc.set_logger(logging.getLogger("logger_to_use"))
>>> def do_something_with_util():
>>>     ... # calls to utils functions, which log records to glc.logger
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class LoggingContext:
    """Logging context with a global logger used within the OSEkit environment."""

    def __init__(self) -> None:
        """Initialize a LoggingContext object.

        The initializer sets up the global logger of the instance.
        """
        self.logger = logging.root

    @contextmanager
    def set_logger(self, logger: logging.Logger) -> Generator[None, Any, None]:
        """Set a contextmanager for calling utils functions with a specific logger.

        Parameters
        ----------
        logger: logging.Logger
            The logger to use in the function called within this context.
            The function called should import the ``LoggingContext``
            instance used for creating the context.

        Examples
        --------
        global_logging_context = LoggingContext()
        logger_to_use = logging.getLogger(__name__)

        def log_something():
            global_logging_context.logger.info("Info log")

        with global_logging_context.set_logger(logger_to_use):
            log_something() # This log will be sent to the logger_to_use logger

        """
        previous_logger = self.logger
        try:
            self.logger = logger
            yield
        finally:
            self.logger = previous_logger
