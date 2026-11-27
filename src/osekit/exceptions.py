class ProjectError(Exception):
    """Base exception for project-related errors."""


class ExternalFilesError(ProjectError):
    """Raised when files outside the project folder prevent an operation."""
