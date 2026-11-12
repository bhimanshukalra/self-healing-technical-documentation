# Usage

## Adding tasks

Use `add_task(title, priority=1)` on a `TaskStore` to create a task. The default priority is 1.

From the shell, run `taskr add "write docs" --priority 2`, which calls the `add` command.

## Listing tasks

`list_tasks(status="open")` returns tasks with the given status.

## Completing tasks

Call `complete_task(store, title)` to mark a task as done.

## HTTP API

`GET /tasks` lists tasks and `POST /tasks` creates one.
