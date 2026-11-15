import json

import httpx
import pytest

from doc_healer.github import GitHubClient, GitHubError

MARKER = "<!-- doc-healer:review -->"


class FakeGitHub:
    """A tiny stand-in for the GitHub REST API; records every request."""

    def __init__(self, open_prs=(), comments=(), patch_status=200):
        self.requests, self.open_prs, self.comments, self.patch_status = [], list(open_prs), list(comments), patch_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, dict(request.url.params), body, dict(request.headers)))
        path, method = request.url.path, request.method
        if (method, path) == ("GET", "/repos/o/r/pulls"):
            return httpx.Response(200, json=self.open_prs)
        if (method, path) == ("POST", "/repos/o/r/pulls"):
            return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/o/r/pull/7"})
        if method == "PATCH" and path.startswith("/repos/o/r/pulls/"):
            return httpx.Response(200, json={})
        if (method, path) == ("GET", "/repos/o/r/issues/5/comments"):
            page = int(request.url.params.get("page", 1))
            return httpx.Response(200, json=self.comments[(page - 1) * 100: page * 100])
        if (method, path) == ("POST", "/repos/o/r/issues/5/comments"):
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/5#issuecomment-new"})
        if method == "PATCH" and path.startswith("/repos/o/r/issues/comments/"):
            if self.patch_status != 200:
                return httpx.Response(self.patch_status, json={"message": "Resource not accessible"})
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/5#issuecomment-" + path.rsplit("/", 1)[1]})
        return httpx.Response(404, json={"message": "Not Found"})

    def client(self):
        return GitHubClient("tok-secret", "o/r", client=httpx.Client(transport=httpx.MockTransport(self)))


def test_a_new_pull_request_is_opened_with_the_right_payload_and_headers():
    fake = FakeGitHub()
    pr = fake.client().open_or_update_pr("T", "B", head="doc-healer/fix-5", base="feature", draft=False)
    assert (pr.number, pr.url, pr.created) == (7, "https://github.com/o/r/pull/7", True)
    method, path, params, _, _ = fake.requests[0]
    assert (method, path, params) == ("GET", "/repos/o/r/pulls", {"head": "o:doc-healer/fix-5", "state": "open"})
    method, path, _, body, headers = fake.requests[1]
    assert (method, path) == ("POST", "/repos/o/r/pulls")
    assert body == {"title": "T", "body": "B", "head": "doc-healer/fix-5", "base": "feature", "draft": False}
    assert headers["authorization"] == "Bearer tok-secret" and headers["accept"] == "application/vnd.github+json"


def test_an_existing_open_pull_request_is_updated_not_duplicated():
    fake = FakeGitHub(open_prs=[{"number": 3, "html_url": "https://github.com/o/r/pull/3"}])
    pr = fake.client().open_or_update_pr("T2", "B2", head="doc-healer/fix-5", base="feature")
    assert (pr.number, pr.created) == (3, False)
    assert [r[0] for r in fake.requests] == ["GET", "PATCH"]
    assert fake.requests[1][1] == "/repos/o/r/pulls/3" and fake.requests[1][3] == {"title": "T2", "body": "B2", "base": "feature"}


def test_a_new_comment_is_posted_when_none_carries_the_marker():
    fake = FakeGitHub(comments=[{"id": 1, "body": "unrelated"}])
    url = fake.client().upsert_comment(5, f"{MARKER}\nhello", MARKER)
    assert url.endswith("issuecomment-new") and fake.requests[-1][:2] == ("POST", "/repos/o/r/issues/5/comments")


def test_the_earlier_comment_is_edited_in_place():
    fake = FakeGitHub(comments=[{"id": 1, "body": "unrelated"}, {"id": 2, "body": f"{MARKER}\nold"}])
    url = fake.client().upsert_comment(5, f"{MARKER}\nnew", MARKER)
    assert url.endswith("issuecomment-2")
    assert fake.requests[-1][:2] == ("PATCH", "/repos/o/r/issues/comments/2") and fake.requests[-1][3] == {"body": f"{MARKER}\nnew"}


def test_a_marker_comment_we_cannot_edit_is_not_fatal():
    fake = FakeGitHub(comments=[{"id": 2, "body": f"{MARKER} planted by someone else"}], patch_status=403)
    assert fake.client().upsert_comment(5, "mine", MARKER).endswith("issuecomment-new")


def test_other_edit_failures_are_raised():
    fake = FakeGitHub(comments=[{"id": 2, "body": MARKER}], patch_status=500)
    with pytest.raises(GitHubError) as exc:
        fake.client().upsert_comment(5, "x", MARKER)
    assert exc.value.status == 500


def test_the_marker_is_found_on_a_later_page_of_comments():
    comments = [{"id": i, "body": "x"} for i in range(1, 101)] + [{"id": 101, "body": MARKER}]
    assert FakeGitHub(comments=comments).client().upsert_comment(5, "n", MARKER).endswith("issuecomment-101")


def test_api_errors_carry_status_and_message():
    class Denied(FakeGitHub):
        def __call__(self, request):
            return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    with pytest.raises(GitHubError, match="403.*not accessible") as exc:
        Denied().client().find_open_pr("x")
    assert exc.value.status == 403


def test_network_failures_become_github_errors():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    client = GitHubClient("t", "o/r", client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(GitHubError) as exc:
        client.find_open_pr("x")
    assert exc.value.status == 0


def test_repr_hides_the_token_and_the_repo_must_be_owner_slash_name():
    assert "tok-secret" not in repr(FakeGitHub().client())
    with pytest.raises(ValueError):
        GitHubClient("t", "just-a-name")
