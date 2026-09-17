# capsize-voice

Measure how you actually write — from your own X (Twitter) archive or any
list of your own text — and use that measurement to get an LLM to write
new text that sounds like you, instead of like an LLM.

```bash
pip install capsize-voice
```

## What it does

1. **Measures** your writing with NLTK: sentence length and variance,
   coordination vs. subordination, part-of-speech mix, function-word
   rates, punctuation habits, lexical variety. The signals that actually
   identify an author, not surface tics.
2. **Renders** those measurements into a markdown style guide an LLM can
   follow, alongside your own persona description and any hard rules you
   want enforced (topics to avoid, formats to avoid).
3. **Curates** a pool of your own real posts as few-shot exemplars —
   filtered for completeness and standalone-ness, ranked by how well
   they match your stated topics.
4. **Generates** new candidate text in your voice from a topic/context,
   via any OpenAI-compatible endpoint (OpenRouter, DeepInfra, OpenAI
   itself, ...), seeded with your style guide and a random sample of
   your real exemplars each time.

Every step is a plain function you can call directly; none of it needs
the others; none of it needs an X archive specifically, either — `analyze`
and `curate` take a plain `list[str]` of your own text from anywhere.

## Quickstart from an X archive

Download your archive: X → Settings → Your account → Download an archive
of your data.

```bash
capsize-voice analyze ~/Downloads/twitter-2026-01-01 \
  --persona "A software engineer who writes short, plain posts." \
  --rules "No politics or religion" "No hashtags or emoji" \
  --banned-terms politics religion trump biden \
  --onbrand-terms code python software open-source \
  --out voice.json

export OPENROUTER_API_KEY=sk-or-...
capsize-voice generate voice.json --context "shipped a new feature today" \
  --model deepseek/deepseek-v4-flash-0731 --provider deepinfra
```

`--api-key-env` reads a different environment variable if you'd rather
not use `OPENROUTER_API_KEY`; `--base-url` points at a different
OpenAI-compatible endpoint entirely (OpenAI itself, a self-hosted
vLLM/DeepInfra deployment, ...).

`analyze` never executes the archive's `tweets.js` — it's read as data,
stripped of its JS assignment wrapper, and parsed as JSON.

## As a library

```python
from capsize_voice import analyze, render_style_guide, curate, generate_candidates

profile = analyze(my_posts)                       # -> StyleProfile
guide = render_style_guide(
    profile,
    persona="A software engineer who writes short posts.",
    rules=["No politics.", "No hashtags."],
)
exemplars = curate(my_posts, banned_terms=["politics"])

candidates = generate_candidates(
    api_key, guide, [e.text for e in exemplars],
    context="shipped a new feature today", count=6,
    model="deepseek/deepseek-v4-flash-0731",
    extra_body={"provider": {"order": ["deepinfra"]}},  # OpenRouter-specific, optional
)
```

## What this package has no opinion on

- What your hard rules are. `render_style_guide`'s `rules` and
  `curate`'s `banned_terms`/`onbrand_terms` are plain lists you supply —
  there is no built-in content policy beyond basic post hygiene
  (dropping fragments, duplicates, and dependent replies).
- Where your data lives, or whether it's backed up. This package reads
  whatever text you hand it and writes whatever file you tell `analyze`
  to. Keeping your own archive and your generated `voice.json` somewhere
  durable is on you.
- Which model generates your text, or which provider serves it.
  `generate_candidates` takes a required `model` and an optional
  `base_url` (default `https://openrouter.ai/api/v1`) plus an
  `extra_body` passthrough for provider-specific routing — pointing this
  at a different OpenAI-compatible endpoint needs no other change.

## NLTK data

The first call to `analyze` downloads `punkt_tab` and
`averaged_perceptron_tagger_eng` into NLTK's standard data directory if
they aren't already there. No manual setup step required.

## Licence

MIT. See [LICENSE](LICENSE).
