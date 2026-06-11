"""Validate deployed frontend tournament data against local expected output."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import ipaddress
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from scripts.build_frontend_data import DEFAULT_OUTPUT, compile_dataset, source_commit
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    build_module_path = Path(__file__).with_name("build_frontend_data.py")
    spec = importlib.util.spec_from_file_location("build_frontend_data", build_module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_frontend_data", module)
    spec.loader.exec_module(module)
    DEFAULT_OUTPUT = module.DEFAULT_OUTPUT
    compile_dataset = module.compile_dataset
    source_commit = module.source_commit


ROOT = Path(__file__).resolve().parents[1]
SHORT_HASH_LENGTH = 12


def cache_busted_url(url: str, commit: str) -> str:
    """Append a cache-busting source query parameter to a URL."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["source"] = commit
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def comparable_dataset(dataset: Any) -> Any:
    """Strip volatile fields before deterministic comparison."""
    if isinstance(dataset, Mapping):
        return {
            key: comparable_dataset(value)
            for key, value in dataset.items()
            if key != "generatedAt"
        }
    if isinstance(dataset, Sequence) and not isinstance(dataset, (str, bytes, bytearray)):
        return [comparable_dataset(value) for value in dataset]
    return dataset


def dataset_digest(dataset: Any) -> str:
    """Hash a dataset after normalizing volatile fields."""
    canonical = comparable_dataset(dataset)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    """Return the first differing JSON path between two objects."""
    if type(expected) is not type(actual):
        return path or "<root>"

    if isinstance(expected, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            return f"{path}.{key}" if path else key
        for key in sorted(actual_keys - expected_keys):
            return f"{path}.{key}" if path else key
        for key in sorted(expected_keys):
            child_path = f"{path}.{key}" if path else key
            difference = first_difference(expected[key], actual[key], child_path)
            if difference:
                return difference
        return None

    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if len(expected) != len(actual):
            return f"{path}.length" if path else "length"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            child_path = f"{path}.[{index}]" if path else f"[{index}]"
            difference = first_difference(expected_item, actual_item, child_path)
            if difference:
                return difference
        return None

    if expected != actual:
        return path or "<root>"
    return None


def validate_dataset(
    expected_dataset: dict[str, Any],
    deployed_dataset: dict[str, Any],
    expected_source_commit: str,
) -> None:
    """Validate source commit and structural equivalence of datasets."""
    source_commit_expected = expected_dataset.get("sourceCommit")
    source_commit_deployed = deployed_dataset.get("sourceCommit")

    if source_commit_expected != expected_source_commit:
        raise ValueError(
            "Source commit mismatch: "
            f"expected dataset has {source_commit_expected}, "
            f"but expected {expected_source_commit}"
        )

    if source_commit_deployed != expected_source_commit:
        raise ValueError(
            "Source commit mismatch: "
            f"deployed dataset has {source_commit_deployed}, "
            f"but expected {expected_source_commit}"
        )

    comparable_expected = comparable_dataset(expected_dataset)
    comparable_deployed = comparable_dataset(deployed_dataset)
    difference = first_difference(comparable_expected, comparable_deployed)
    if difference:
        raise ValueError(f"Dataset mismatch at {difference}")


def load_expected_dataset(expected_path: Path) -> dict[str, Any]:
    """Load expected dataset from disk, generating one when missing."""
    if expected_path.exists():
        return json.loads(expected_path.read_text())
    return compile_dataset(ROOT)


def fetch_dataset(url: str) -> dict[str, Any]:
    """Fetch JSON dataset from deployed URL."""
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    is_loopback_host = hostname == "localhost"
    if not is_loopback_host:
        try:
            is_loopback_host = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback_host = False
    if parts.scheme == "http" and not is_loopback_host:
        raise ValueError("Insecure URL scheme: use HTTPS for non-local validation")
    if parts.scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported URL scheme: {parts.scheme}")
    request = Request(url, headers={"User-Agent": "cuthberto-carlos-validator"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - scheme validated above
        return json.load(response)


def main() -> None:
    """Run deployed frontend dataset validation."""
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("retry count must be >= 1")
        return parsed

    def normalized_commit(value: str) -> str:
        commit = value.strip()
        if len(commit) < 7:
            raise argparse.ArgumentTypeError("commit hash must be at least 7 characters")
        return commit

    def non_negative_float(value: str) -> float:
        parsed = float(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("retry delay must be >= 0")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="URL to deployed tournament.json",
    )
    parser.add_argument(
        "--source-commit",
        type=normalized_commit,
        default=source_commit(ROOT),
        help="Expected source commit (defaults to current/GITHUB_SHA short hash)",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to local expected tournament.json",
    )
    parser.add_argument("--retries", type=positive_int, default=6)
    parser.add_argument("--retry-delay", type=non_negative_float, default=5.0)
    args = parser.parse_args()

    expected_dataset = load_expected_dataset(args.expected)
    expected_commit = normalized_commit(args.source_commit)[:SHORT_HASH_LENGTH]
    deployed_url = cache_busted_url(args.url, expected_commit)
    retries = args.retries

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            deployed_dataset = fetch_dataset(deployed_url)
            validate_dataset(expected_dataset, deployed_dataset, expected_commit)
            print(
                "Validated deployed dataset successfully "
                "(sourceCommit="
                f"{expected_commit}, "
                f"digest={dataset_digest(deployed_dataset)[:SHORT_HASH_LENGTH]}...)"
            )
            return
        except Exception as error:  # noqa: BLE001 - report exact validation failure
            last_error = error
            if attempt == retries:
                break
            print(
                f"Validation attempt {attempt}/{retries} failed: {error}. "
                f"Retrying in {args.retry_delay}s..."
            )
            time.sleep(args.retry_delay)

    raise SystemExit(f"Validation failed after {retries} attempts: {last_error}")


if __name__ == "__main__":
    main()
