"""Verify that deployed frontend data matches the repository prediction assets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

if __package__:
    from scripts.build_frontend_data import ROOT, compile_dataset, source_commit
else:
    from build_frontend_data import ROOT, compile_dataset, source_commit


DEFAULT_URL = "https://ryantjx.github.io/cuthberto-carlos/data/tournament.json"


def comparable_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Remove metadata that does not describe the underlying model assets."""
    comparable = copy.deepcopy(dataset)
    comparable.pop("generatedAt", None)
    comparable.pop("sourceCommit", None)
    return comparable


def dataset_digest(dataset: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for source-derived tournament data."""
    payload = json.dumps(
        comparable_dataset(dataset),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def first_difference(expected: Any, actual: Any, path: str = "$.") -> str | None:
    """Return the first structural difference between two JSON-compatible values."""
    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, found {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: missing keys {sorted(expected_keys - actual_keys)}, "
                f"unexpected keys {sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}{key}.")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, found {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = first_difference(
                expected_item,
                actual_item,
                f"{path}[{index}].",
            )
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, found {actual!r}"
    return None


def cache_busted_url(url: str, commit: str) -> str:
    """Add a source commit query parameter to bypass stale Pages CDN responses."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["source"] = commit
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON document from an HTTPS deployment URL."""
    if urlsplit(url).scheme != "https":
        raise ValueError("Deployed data URL must use HTTPS")
    request = Request(url, headers={"User-Agent": "cuthberto-carlos-data-validator"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - validated HTTPS URL
        return json.load(response)


def validate_dataset(
    expected: dict[str, Any],
    deployed: dict[str, Any],
    expected_commit: str,
) -> None:
    """Raise when deployed data differs from freshly compiled repository assets."""
    deployed_commit = str(deployed.get("sourceCommit", ""))
    if deployed_commit != expected_commit:
        raise ValueError(
            f"Source commit mismatch: expected {expected_commit}, found {deployed_commit or 'missing'}"
        )

    expected_comparable = comparable_dataset(expected)
    deployed_comparable = comparable_dataset(deployed)
    difference = first_difference(expected_comparable, deployed_comparable)
    if difference:
        raise ValueError(f"Tournament data mismatch at {difference}")


def validate_deployment(
    url: str,
    expected: dict[str, Any],
    expected_commit: str,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Fetch and validate deployed data, retrying while Pages propagates."""
    request_url = cache_busted_url(url, expected_commit)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            deployed = fetch_json(request_url)
            validate_dataset(expected, deployed, expected_commit)
            return deployed
        except Exception as error:  # noqa: BLE001 - report final network or data error
            last_error = error
            if attempt < retries:
                print(f"Validation attempt {attempt}/{retries} failed: {error}")
                time.sleep(retry_delay)
    raise RuntimeError(f"Deployment validation failed after {retries} attempts: {last_error}")


def main() -> None:
    """Compile repository assets and compare them with the deployed JSON sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--source-commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args()

    if args.retries < 1:
        parser.error("--retries must be at least 1")

    expected_commit = (
        args.source_commit[:12] if args.source_commit else source_commit(ROOT)
    )
    expected = compile_dataset(ROOT)
    deployed = validate_deployment(
        args.url,
        expected,
        expected_commit,
        args.retries,
        args.retry_delay,
    )
    print(
        "Validated deployed frontend data: "
        f"snapshot {deployed['snapshotDate']}, "
        f"{len(deployed['groupMatches'])} group matches, "
        f"{len(deployed['knockoutMatches'])} knockout matches, "
        f"SHA-256 {dataset_digest(deployed)}"
    )


if __name__ == "__main__":
    main()
