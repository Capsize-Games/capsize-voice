"""Fetch the NLTK data this package's analysis needs, once.

NLTK ships its models and corpora separately from the library itself.
Rather than making every caller run a downloader command before their
first import works, this looks the data up and fetches anything missing
into NLTK's standard cache directory, exactly what
`python -m nltk.downloader ...` would do.
"""

import nltk

_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger_eng": (
        "averaged_perceptron_tagger_eng"
    ),
}

_checked = False


def ensure_nltk_data() -> None:
    """Download any of `_RESOURCES` not already present. Idempotent."""
    global _checked
    if _checked:
        return
    for path, package in _RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    _checked = True
