"""Reads Reddit posts and comments from the Arctic Shift archive (Reddit refuses automated clients).

The archive is a third-party copy of public Reddit, free and without login (docs/adr/0007); it can
lag behind Reddit by a little. Every title, body and comment is someone else's text.
"""

import re
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from collections.abc import Callable

API = "https://arctic-shift.photon-reddit.com/api"
POST_ID = re.compile(r"[a-z0-9]{1,12}")
SUBREDDIT = re.compile(r"[A-Za-z0-9_]{2,21}")
MAX_IDS = 10
COMMENT_FETCH = 40


class RedditUnavailableError(Exception):
    """The archive did not answer, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class Post:
    """A Reddit post as the archive holds it."""

    id: str
    subreddit: str
    title: str
    score: int
    comments: int
    body: str


@dataclass(frozen=True, slots=True)
class Comment:
    """A top-level comment and its score."""

    score: int
    body: str


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


class ArcticShift:
    """The archive's read-only API."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        """Use `http` (tests) or a client `factory` makes on first use (no socket while idle)."""
        super().__init__()
        self._http = http
        self._factory = factory or httpx.Client

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if self._http is None:
            self._http = self._factory()
        try:
            response = self._http.get(f"{API}{path}", params=params)
        except httpx.HTTPError as exc:
            msg = "the Reddit archive didn't answer"
            raise RedditUnavailableError(msg) from exc
        if response.status_code != HTTPStatus.OK:
            msg = f"the Reddit archive answered {response.status_code}"
            raise RedditUnavailableError(msg)
        try:
            return objs(obj(response.json()).get("data"))
        except ValueError as exc:
            msg = "the Reddit archive answered something that is not JSON"
            raise RedditUnavailableError(msg) from exc

    def posts(self, ids: list[str]) -> dict[str, Post]:
        """Return the archived posts among `ids` (base-36 post ids), keyed by id.

        A failing archive raises `RedditUnavailableError`.
        """
        wanted = [i for i in dict.fromkeys(ids) if POST_ID.fullmatch(i)][:MAX_IDS]
        if not wanted:
            return {}
        found: dict[str, Post] = {}
        for item in self._get("/posts/ids", {"ids": ",".join(wanted)}):
            post_id = str(item.get("id") or "")
            subreddit = str(item.get("subreddit") or "")
            if post_id not in wanted or not SUBREDDIT.fullmatch(subreddit):
                continue
            found[post_id] = Post(
                post_id,
                subreddit,
                _text(item.get("title")),
                _int(item.get("score")),
                _int(item.get("num_comments")),
                _text(item.get("selftext")),
            )
        return found

    def top_comments(self, post_id: str, limit: int) -> list[Comment]:
        """Return the post's `limit` highest-scored top-level comments, best first.

        A failing archive raises `RedditUnavailableError`.
        """
        if not POST_ID.fullmatch(post_id):
            return []
        tree = self._get("/comments/tree", {"link_id": post_id, "limit": COMMENT_FETCH})
        comments = [
            Comment(_int(data.get("score")), _text(data.get("body")))
            for data in (obj(node.get("data")) for node in tree if node.get("kind") == "t1")
            if _text(data.get("body")) not in {"", "[deleted]", "[removed]"}
        ]
        comments.sort(key=lambda c: c.score, reverse=True)
        return comments[:limit]
