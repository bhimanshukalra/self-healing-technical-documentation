from taskr.tasks import TaskStore


def test_add_task():
    assert TaskStore().add_task("x")["title"] == "x"
