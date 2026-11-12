class TaskStore:
    """In-memory store of tasks."""

    def __init__(self, limit: int = 100):
        self.limit = limit
        self._tasks: list[dict] = []

    def add_task(self, title: str, priority: int = 1) -> dict:
        """Add a task with the given title and priority."""
        task = {"title": title, "priority": priority, "status": "open"}
        self._tasks.append(task)
        return task

    def list_tasks(self, status: str = "open") -> list[dict]:
        """Return tasks that have the given status."""
        return [t for t in self._tasks if t["status"] == status]


def complete_task(store: TaskStore, title: str) -> bool:
    """Mark the task with this title as done. Returns True if found."""

    def _match(t: dict) -> bool:
        return t["title"] == title

    for task in store.list_tasks():
        if _match(task):
            task["status"] = "done"
            return True
    return False
