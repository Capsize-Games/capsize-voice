"""Characterize the package boundary and release-source contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_manifest_identifies_a_python_library_and_cli() -> None:
    manifest = json.loads((ROOT / "capsize.json").read_text())

    assert manifest["name"] == "capsize-voice"
    assert manifest["type"] == "library"
    assert manifest["runtimes"] == {"python": ">=3.11"}
    assert manifest["entrypoints"] == {
        "python": "capsize_voice",
        "cli": "capsize-voice",
    }
    assert manifest["owner"] == "Capsize-Games"
    assert manifest["license"] == "MIT"


def test_runtime_dependencies_keep_provider_policy_outside_commons() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert "nltk>=3.8" in dependencies
    assert "openai>=1.50" in dependencies
    assert "typing-extensions>=4.7" in dependencies
    shared_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.startswith("capsize-commons")
    ]
    assert not shared_dependencies


def test_documented_provider_boundary_matches_the_public_api() -> None:
    document = (ROOT / "docs" / "FLEET_CONSOLIDATION.md").read_text()

    for phrase in (
        "caller-supplied `base_url`",
        "opaque `extra_body` passthrough",
        "Provider retry, timeout, and rate-limit policy",
        "trusted-publishing",
        "hq#23 remains open",
        "No duplicate helper is removed in this audit",
    ):
        assert phrase in document
