"""A small GitHub REST client: open or update the fix pull request, upsert the review comment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

API_URL = "https://api.github.com"


class GitHubError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str  # the page people open
    created: bool  # False when an existing PR was updated


class GitHubClient:
    def __init__(self, token: str, repo: str, api_url: str = API_URL, client: httpx.Client | None = None):
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self._owner = repo.split("/")[0]
        self._api = api_url.rstrip("/")
        self._http = client or httpx.Client(timeout=30.0)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def __repr__(self) -> str:  # never print the token
        return f"GitHubClient(repo={self.repo!r})"

    # -- pull requests ---------------------------------------------------------------------

    def find_open_pr(self, head_branch: str) -> PullRequest | None:
        prs = self._request("GET", f"/repos/{self.repo}/pulls", params={"head": f"{self._owner}:{head_branch}", "state": "open"})
        return PullRequest(prs[0]["number"], prs[0]["html_url"], False) if prs else None

    def open_or_update_pr(self, title: str, body: str, head: str, base: str, draft: bool = False) -> PullRequest:
        """Open a PR from ``head`` into ``base``; if one is already open for ``head``, update its text.

        This makes a re-run on the same pull request refresh the fix PR instead of piling up duplicates.
        """
        if existing := self.find_open_pr(head):
            self._request("PATCH", f"/repos/{self.repo}/pulls/{existing.number}", json={"title": title, "body": body, "base": base})
            return existing
        data = self._request("POST", f"/repos/{self.repo}/pulls",
                             json={"title": title, "body": body, "head": head, "base": base, "draft": draft})
        return PullRequest(data["number"], data["html_url"], True)

    # -- comments --------------------------------------------------------------------------

    def upsert_comment(self, pr_number: int, body: str, marker: str) -> str:
        """Post ``body`` on a PR, or edit the earlier comment carrying ``marker``. Returns its URL."""
        for comment in self._comments(pr_number):
            if marker in (comment.get("body") or ""):
                try:
                    data = self._request("PATCH", f"/repos/{self.repo}/issues/comments/{comment['id']}", json={"body": body})
                    return data["html_url"]
                except GitHubError as exc:
                    if exc.status not in (403, 404):  # not our comment to edit: fall through and post our own
                        raise
        return self._request("POST", f"/repos/{self.repo}/issues/{pr_number}/comments", json={"body": body})["html_url"]

    def _comments(self, pr_number: int, max_pages: int = 5):
        for page in range(1, max_pages + 1):
            batch = self._request("GET", f"/repos/{self.repo}/issues/{pr_number}/comments", params={"per_page": 100, "page": page})
            yield from batch
            if len(batch) < 100:
                return

    # -- plumbing --------------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._http.request(method, self._api + path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise GitHubError(0, f"{exc.__class__.__name__}: could not reach GitHub") from exc
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text)
            except ValueError:
                message = resp.text
            raise GitHubError(resp.status_code, str(message)[:300])
        return resp.json()
