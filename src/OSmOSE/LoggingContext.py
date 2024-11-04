import logging
from contextlib import contextmanager


class LoggingContext:
    """Logging context with a global logger used within the OSEkit environment."""

    def __init__(self) -> None:
        """Initialize a LoggingContext object.

        The initializer sets up the global logger of the instance.
        """
        self.logger = logging.root

    @contextmanager
    def set_logger(self, logger: logging.Logger) -> None:
        """Set a contextmanager for calling utils functions with a specific logger.

        Parameters
        ----------
        logger: logging.Logger
            The logger to use in the function called within this context.
            The function called should import the LoggingContext instance used for creating the context.

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
