import hashlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_PROJECT = Path(__file__).parent / "fixtures" / "sample_project"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder; texts sharing words are similar."""

    dim = 256

    def __init__(self):
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def embedder():
    return FakeEmbedder()


class GitRepo:
    """A throwaway git repository for diff tests."""

    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
            "HOME": str(self.path), "PATH": os.environ["PATH"],
        }
        out = subprocess.run(
            ["git", *args], cwd=self.path, env=env, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    def write(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def edit(self, rel: str, old: str, new: str) -> None:
        text = (self.path / rel).read_text()
        assert old in text, f"{old!r} not in {rel}"
        self.write(rel, text.replace(old, new, 1))

    def commit(self, message: str = "c") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def taskr_repo(tmp_path):
    """A git repo whose first commit is the taskr sample project; returns (repo, base_sha)."""
    repo = GitRepo(tmp_path / "repo")
    shutil.copytree(FIXTURE_PROJECT, repo.path, dirs_exist_ok=True)
    return repo, repo.commit("base")


class ScriptedProvider:
    """An LLM provider that replays outcomes: an exception to raise, or text to return.
    The last outcome repeats once the list runs out."""

    def __init__(self, name, *outcomes):
        from doc_healer.llm import LLMResponse

        self._response = LLMResponse
        self.name, self.model, self.outcomes, self.calls = name, "m", list(outcomes), 0
        self.last_messages, self.last_kwargs = None, None

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.last_messages, self.last_kwargs = list(messages), kwargs
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return self._response(outcome, self.name, self.model)
