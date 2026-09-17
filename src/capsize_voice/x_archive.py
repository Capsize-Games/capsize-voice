"""Read posts out of an X (Twitter) data export.

X's downloadable archive (Settings -> Your account -> Download an
archive of your data) includes a `tweets.js` file: a JavaScript
assignment wrapping a JSON array, not a plain JSON file. This reads it
as data only — it is never executed as code.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CANDIDATE_PATHS = ("tweets.js", "data/tweets.js")


@dataclass(frozen=True)
class ArchivePost:
    """One post from an X archive export."""

    id: str
    text: str
    created_at: str
    is_reply: bool
    is_retweet: bool


def find_tweets_file(archive_dir: Path | str) -> Path:
    """Locate `tweets.js` inside an X archive folder."""
    base = Path(archive_dir)
    for candidate in _CANDIDATE_PATHS:
        path = base / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find tweets.js under {base} "
        f"(looked for {', '.join(_CANDIDATE_PATHS)}). Download your "
        "archive from X: Settings -> Your account -> "
        "Download an archive of your data."
    )


def load_tweets_js(path: Path | str) -> list[dict[str, object]]:
    """Parse `tweets.js` into its raw list of tweet objects.

    The file looks like `window.YTD.tweets.part0 = [ {...}, ... ];` — the
    leading assignment and trailing semicolon are stripped, and the rest
    is parsed as JSON. Nothing in this file is ever executed.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    idx = raw.find("=")
    if idx == -1:
        raise ValueError(f"unrecognized tweets.js format in {path}")
    payload = raw[idx + 1 :].strip()
    if payload.endswith(";"):
        payload = payload[:-1].strip()
    tweets = json.loads(payload)
    if not isinstance(tweets, list):
        raise ValueError("expected tweets.js to contain a JSON array")
    return tweets


def _normalize(raw: dict[str, object]) -> ArchivePost:
    inner_val = raw.get("tweet")
    inner: dict[str, object] = (
        inner_val if isinstance(inner_val, dict) else raw
    )
    text = str(inner.get("full_text") or inner.get("text") or "")
    tweet_id = str(inner.get("id_str") or inner.get("id") or "")
    created_at = str(inner.get("created_at") or "")
    is_reply = bool(inner.get("in_reply_to_status_id_str"))
    is_retweet = bool(inner.get("retweeted_status")) or text.startswith(
        "RT @"
    )
    return ArchivePost(
        id=tweet_id,
        text=text,
        created_at=created_at,
        is_reply=is_reply,
        is_retweet=is_retweet,
    )


def load_archive(archive_dir: Path | str) -> list[ArchivePost]:
    """Load every post from an X archive folder."""
    path = find_tweets_file(archive_dir)
    return [_normalize(t) for t in load_tweets_js(path)]


_MENTION_PREFIX = re.compile(r"^(@\w+[\s,]*)+")


def own_post_texts(
    posts: list[ArchivePost],
    include_replies: bool = False,
) -> list[str]:
    """Return original post text: no retweets, replies excluded by default.

    Replies are excluded by default because a reply's text often only
    makes sense next to the post it replies to, which makes it a poor
    example of unprompted, standalone writing.
    """
    return [
        p.text
        for p in posts
        if not p.is_retweet
        and (include_replies or not p.is_reply)
        and _MENTION_PREFIX.sub("", p.text).strip()
    ]
