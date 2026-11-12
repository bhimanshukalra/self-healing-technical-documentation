from fastapi import FastAPI

app = FastAPI()


@app.get("/tasks")
async def get_tasks(status: str = "open"):
    """List tasks, filtered by status."""
    return []


@app.post("/tasks")
def create_task(title: str, priority: int = 1):
    """Create a new task."""
    return {"title": title, "priority": priority}
