# Fleet consolidation boundary

## Repository identity and release state

`capsize-voice` is the public Capsize-Games repository at
https://github.com/Capsize-Games/capsize-voice. It is an MIT-licensed Python
library and CLI supporting Python 3.11 and newer. Its package version is
`0.1.0`.

At audit time, GitHub has no release for this repository and PyPI returns 404
for `capsize-voice`. Consumers therefore resolve the package from a local
editable checkout or Git `main` as an explicit pre-release source. The
trusted-publishing and published-artifact gate in hq#23 remains open; this
document does not claim a registry release.

## Domain boundary decisions

| Surface | Decision | Authority and compatibility contract |
| --- | --- | --- |
| Voice measurement and profile schema | Retain-local | NLTK-backed measurements and `StyleProfile` are the domain API. No generic text-analysis extraction is proposed. |
| Style-guide rendering and prompt construction | Retain-local | Persona rules, exemplar selection, history formatting, and anti-fabrication instructions are product semantics. |
| OpenAI-compatible model client | Retain-local | `openai.OpenAI` is used with caller-supplied `base_url`, required `model`, and opaque `extra_body` passthrough. Provider routing stays outside the package's domain boundary. |
| Provider retry, timeout, and rate-limit policy | Retain-local | The package does not add a generic retry adapter or claim provider-independent retry semantics. `GenerationError` preserves the current failure boundary. |
| API keys and secrets | Retain-local | Callers pass an API key to the function/CLI boundary; the package does not persist, rotate, or share credentials. |
| NLTK data | Retain-local | `ensure_nltk_data()` owns the two required downloads and uses NLTK's standard cache. It is not moved into a shared service package. |
| X archive parsing and curation | Retain-local | Archive wrapper parsing, own-post filtering, completeness checks, and exemplar ranking are library-specific. |
| Safety categories and scoring | Retain-local | Weighted regex categories and threshold decisions are domain behavior, not generic platform policy. |
| Audio boundaries | Not applicable | This package measures and generates text; no audio provider or audio persistence surface exists. |
| `capsize-commons` config/db/web/http | Not applicable | This is a stateless library with explicit-call APIs, no service settings/database/HTTP server, and provider-specific model transport. |

## Dependency and consumer authority

The declared ranges below are the authority in `pyproject.toml`; this
repository has no committed lockfile. The versions shown are the versions
used for local verification at audit time.

| Dependency | Declared version | Verified version | Authority or exception |
| --- | --- | --- | --- |
| Python | `>=3.11` | Python `3.13.5` environment | Library compatibility floor and CI matrix. |
| NLTK | `>=3.8` | `3.10.3` | Text measurement and tokenizer/tagger data remain local. |
| OpenAI | `>=1.50` | `3.16.2` | OpenAI-compatible transport boundary; `base_url` and `extra_body` remain caller-controlled. |
| typing-extensions | `>=4.7` | `4.16.0` | Direct runtime dependency for Python 3.11-compatible public records. |
| pytest | `>=8.2` | `9.1.1` | Test contract. |
| Ruff | `>=0.5` | `0.16.8` | Lint contract. |
| mypy | `>=1.10` | `2.3.1` | Type-check contract over `src/capsize_voice`. |

The known Python service consumer declares `capsize-voice>=0.1.0` while its
local development and GitHub/Docker paths still use a local checkout or Git
`main`, matching the absence of a published artifact. That consumer must not
replace the source with a fabricated PyPI pin; hq#23 is the release gate.

## Provider and behavior compatibility

The public behavior stays above any platform package:

- `generate_candidates` requires an API key, exemplars, and an explicit model,
  returns parsed candidate strings, and passes `base_url`/`extra_body` through
  unchanged.
- `generate_reply` requires the same provider inputs, keeps recent-turn
  formatting and quote stripping, and returns one reply string.
- OpenAI SDK failures, empty choices, and malformed JSON remain
  `GenerationError` outcomes; they are not normalized into a service error
  envelope.
- NLTK downloads are data-cache side effects of `analyze`, not package or
  repository writes. The X archive is read as data and never executed.
- Safety scoring remains weighted, regex-based, and inspectable; the package
  does not claim a trained classifier or provider-independent moderation.

Existing provider/prompt tests plus the fleet contract tests characterize these
boundaries. No duplicate helper is removed in this audit, so there is no
before/after behavior delta to hide.

## Verification and rollback

The existing CI matrix runs Python 3.11, 3.12, and 3.13 with NLTK data
installed, then Ruff, mypy, and the full test suite. The local `just` contract
maps to those checks plus `uv build`; no release upload or trusted-publishing
claim is made here.

Reverting the audit commit removes only metadata, documentation, task aliases,
and characterization tests. It does not change generation, prompts, provider
requests, NLTK behavior, consumer source resolution, or any user archive.

No repository, directory, branch, issue history, generated output, secret,
private dataset, model artifact, or deployment copy is deleted, archived,
retired, or overwritten.
