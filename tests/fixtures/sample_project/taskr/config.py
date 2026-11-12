import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime configuration for taskr."""

    db_path: str = "tasks.db"
    max_tasks: int = 100
    default_priority: int = 1


def load_settings() -> Settings:
    """Load settings from the TASKR_DB_PATH and TASKR_MAX_TASKS environment variables."""
    return Settings(
        db_path=os.environ.get("TASKR_DB_PATH", "tasks.db"),
        max_tasks=int(os.environ.get("TASKR_MAX_TASKS", "100")),
    )
