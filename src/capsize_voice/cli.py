"""Command-line entry point: `capsize-voice analyze` / `generate`.

    capsize-voice analyze ~/Downloads/twitter-2026-01-01 --out voice.json
    capsize-voice generate voice.json --context "shipped a new feature"

`analyze` writes one JSON file holding the style guide and the curated
exemplar pool; `generate` reads that file back, so the two commands
never need to agree on a schema beyond what this module itself writes
and reads.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from capsize_voice.exemplars import curate
from capsize_voice.generate import DEFAULT_BASE_URL, generate_candidates
from capsize_voice.profile import analyze
from capsize_voice.render import render_style_guide
from capsize_voice.x_archive import load_archive, own_post_texts


def _cmd_analyze(args: argparse.Namespace) -> None:
    posts = load_archive(args.archive_dir)
    texts = own_post_texts(posts, include_replies=args.include_replies)
    if not texts:
        sys.exit("no usable posts found in that archive")

    style_profile = analyze(texts, exclude_words=args.banned_terms)
    exemplars = curate(
        texts,
        banned_terms=args.banned_terms,
        onbrand_terms=args.onbrand_terms,
    )
    style_guide = render_style_guide(
        style_profile, persona=args.persona, rules=args.rules
    )

    out = {
        "style_guide": style_guide,
        "exemplars": [e.text for e in exemplars],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"{style_profile.n_posts} posts analyzed, "
        f"{len(exemplars)} exemplars kept -> {args.out}"
    )


def _cmd_generate(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.voice_file).read_text())
    api_key = os.environ.get(args.api_key_env, "")
    extra_body: dict[str, object] | None = (
        {"provider": {"order": args.provider, "allow_fallbacks": False}}
        if args.provider
        else None
    )
    candidates = generate_candidates(
        api_key,
        data["style_guide"],
        data["exemplars"],
        args.context,
        args.count,
        model=args.model,
        base_url=args.base_url,
        extra_body=extra_body,
    )
    for candidate in candidates:
        print(f"[{len(candidate):>3}c] {candidate}")


def build_parser() -> argparse.ArgumentParser:
    """Build the `capsize-voice` argument parser."""
    parser = argparse.ArgumentParser(prog="capsize-voice")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser(
        "analyze", help="build a style guide + exemplar pool from an archive"
    )
    analyze_cmd.add_argument("archive_dir", help="your X data export folder")
    analyze_cmd.add_argument("--out", default="voice.json")
    analyze_cmd.add_argument("--persona", default="")
    analyze_cmd.add_argument(
        "--rules", nargs="*", default=[], help="hard content prohibitions"
    )
    analyze_cmd.add_argument("--banned-terms", nargs="*", default=[])
    analyze_cmd.add_argument("--onbrand-terms", nargs="*", default=[])
    analyze_cmd.add_argument(
        "--include-replies", action="store_true", default=False
    )
    analyze_cmd.set_defaults(func=_cmd_analyze)

    generate_cmd = sub.add_parser(
        "generate", help="generate candidate posts from a voice file"
    )
    generate_cmd.add_argument("voice_file", help="output of `analyze`")
    generate_cmd.add_argument("--context", required=True)
    generate_cmd.add_argument("-n", "--count", type=int, default=6)
    generate_cmd.add_argument(
        "--model", required=True, help="e.g. deepseek/deepseek-v4-flash-0731"
    )
    generate_cmd.add_argument("--base-url", default=DEFAULT_BASE_URL)
    generate_cmd.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="env var to read the API key from",
    )
    generate_cmd.add_argument(
        "--provider",
        nargs="*",
        default=[],
        help="pin the OpenRouter upstream provider order, e.g. deepinfra",
    )
    generate_cmd.set_defaults(func=_cmd_generate)

    return parser


def main() -> None:
    """Parse argv and run the selected subcommand."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
