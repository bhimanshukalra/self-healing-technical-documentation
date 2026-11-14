from doc_healer.changes import detect_chunk_changes
from doc_healer.gitutil import changed_files


def _summary(repo, base):
    head = repo.commit("change")
    return {(c.chunk_id, c.change_type) for c in detect_chunk_changes(repo.path, base, head)}


def test_no_changes(taskr_repo):
    repo, base = taskr_repo
    repo.write("NOTES.txt", "hi")  # unrelated file
    assert _summary(repo, base) == set()


def test_renamed_parameter_modifies_only_that_method(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    assert _summary(repo, base) == {("taskr/tasks.py::TaskStore.add_task", "modified")}


def test_method_edit_does_not_modify_the_class(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/tasks.py", 'status: str = "open"', 'status: str = "todo"')
    changes = _summary(repo, base)
    assert ("taskr/tasks.py::TaskStore.list_tasks", "modified") in changes
    assert ("taskr/tasks.py::TaskStore", "modified") not in changes


def test_config_default_change_modifies_the_config_class_only(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/config.py", "max_tasks: int = 100", "max_tasks: int = 500")
    assert _summary(repo, base) == {("taskr/config.py::Settings", "modified")}


def test_class_level_change_modifies_the_class(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/tasks.py", '"""In-memory store of tasks."""', '"""A store of tasks."""')
    assert _summary(repo, base) == {("taskr/tasks.py::TaskStore", "modified")}


def test_added_and_removed_chunks(taskr_repo):
    repo, base = taskr_repo
    repo.write("taskr/extra.py", "def helper():\n    return 1\n")
    repo.edit("taskr/tasks.py", "def complete_task(", "def finish_task(")
    assert _summary(repo, base) == {
        ("taskr/extra.py::helper", "added"),
        ("taskr/tasks.py::finish_task", "added"),
        ("taskr/tasks.py::complete_task", "removed"),
    }


def test_deleted_file_removes_all_its_chunks(taskr_repo):
    repo, base = taskr_repo
    (repo.path / "taskr/api.py").unlink()
    assert _summary(repo, base) == {
        ("taskr/api.py::get_tasks", "removed"),
        ("taskr/api.py::create_task", "removed"),
    }


def test_pure_rename_has_no_changes_but_rename_with_edit_does(taskr_repo):
    repo, base = taskr_repo
    repo.git("mv", "taskr/api.py", "taskr/http.py")
    moved = repo.commit("mv")
    assert changed_files(repo.path, base, moved)[0].status == "renamed"
    assert detect_chunk_changes(repo.path, base, moved) == []

    repo.edit("taskr/http.py", "def create_task(title: str, priority: int = 1)",
              "def create_task(title: str, priority: int = 2)")
    assert _summary(repo, base) == {("taskr/http.py::create_task", "modified")}


def test_comment_only_edit_is_reported_here_for_the_filter_to_judge(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    assert _summary(repo, base) == {("taskr/tasks.py::TaskStore.add_task", "modified")}


def test_test_files_docs_and_broken_files_are_ignored(taskr_repo):
    repo, base = taskr_repo
    repo.edit("tests/test_tasks.py", 'add_task("x")', 'add_task("y")')
    repo.edit("docs/usage.md", "Adding tasks", "Add tasks")
    repo.write("taskr/new_broken.py", "def (:\n")
    assert _summary(repo, base) == set()


def test_old_and_new_chunks_are_attached(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/config.py", "max_tasks: int = 100", "max_tasks: int = 500")
    head = repo.commit("c")
    (change,) = detect_chunk_changes(repo.path, base, head)
    assert "= 100" in change.old.source and "= 500" in change.new.source
