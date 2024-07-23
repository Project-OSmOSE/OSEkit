from loguru import logger
from rich.logging import RichHandler
from rich.console import Console


console = Console()

log_format = "{time:HH:mm:ss} | " "[red]{name}:{function}: {line}[/] | " "{message}"

logger.remove()
logger.add(
    RichHandler(console=console, rich_tracebacks=True, show_time=False, markup=True),
    colorize=True,
    format=log_format,
    level="DEBUG",
)
