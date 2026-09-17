import json
from pathlib import Path

import pytest

from capsize_voice.x_archive import (
    find_tweets_file,
    load_archive,
    load_tweets_js,
    own_post_texts,
)

_SAMPLE_TWEETS = [
    {
        "tweet": {
            "id_str": "1",
            "full_text": "Shipped a new feature today.",
            "created_at": "Mon Jan 01 00:00:00 +0000 2026",
        }
    },
    {
        "tweet": {
            "id_str": "2",
            "full_text": "@alice yes exactly",
            "created_at": "Mon Jan 01 00:00:00 +0000 2026",
            "in_reply_to_status_id_str": "999",
        }
    },
    {
        "tweet": {
            "id_str": "3",
            "full_text": "RT @bob: some retweeted content",
            "created_at": "Mon Jan 01 00:00:00 +0000 2026",
        }
    },
]


def _write_archive(tmp_path: Path) -> Path:
    tweets_js = tmp_path / "tweets.js"
    payload = json.dumps(_SAMPLE_TWEETS)
    tweets_js.write_text(f"window.YTD.tweets.part0 = {payload};")
    return tmp_path


def test_find_tweets_file_at_root(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)
    assert find_tweets_file(archive) == archive / "tweets.js"


def test_find_tweets_file_under_data(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    tweets_js = tmp_path / "data" / "tweets.js"
    tweets_js.write_text("window.YTD.tweets.part0 = [];")
    assert find_tweets_file(tmp_path) == tweets_js


def test_find_tweets_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_tweets_file(tmp_path)


def test_load_tweets_js_never_executes_content(tmp_path: Path) -> None:
    tweets_js = tmp_path / "tweets.js"
    tweets_js.write_text('window.YTD.tweets.part0 = [{"id": "1"}];')
    result = load_tweets_js(tweets_js)
    assert result == [{"id": "1"}]


def test_load_archive_normalizes_posts(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)
    posts = load_archive(archive)
    assert len(posts) == 3
    assert posts[0].text == "Shipped a new feature today."
    assert posts[1].is_reply is True
    assert posts[2].is_retweet is True


def test_own_post_texts_excludes_replies_and_retweets(
    tmp_path: Path,
) -> None:
    archive = _write_archive(tmp_path)
    posts = load_archive(archive)
    texts = own_post_texts(posts)
    assert texts == ["Shipped a new feature today."]


def test_own_post_texts_can_include_replies(tmp_path: Path) -> None:
    # own_post_texts only filters; mention-stripping happens in clean(),
    # applied later by analyze()/curate() — so the raw reply text with
    # its @-mention intact is the correct, unmodified output here.
    archive = _write_archive(tmp_path)
    posts = load_archive(archive)
    texts = own_post_texts(posts, include_replies=True)
    assert "@alice yes exactly" in texts
