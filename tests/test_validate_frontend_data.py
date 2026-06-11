"""Tests for deployed frontend data validation."""

import unittest

from scripts.validate_deployed_frontend_data import (
    cache_busted_url,
    comparable_dataset,
    dataset_digest,
    first_difference,
    validate_dataset,
)


class DeployedFrontendDataTests(unittest.TestCase):
    """Validate canonical comparisons without requiring network access."""

    def setUp(self):
        self.dataset = {
            "snapshotDate": "2026-06-11",
            "generatedAt": "2026-06-11T12:00:00Z",
            "sourceCommit": "abc123abc123",
            "groupMatches": [{"id": "match-a", "probability": 0.5}],
            "knockoutMatches": [],
        }

    def test_volatile_generation_timestamp_does_not_change_digest(self):
        changed = {**self.dataset, "generatedAt": "2026-06-11T13:00:00Z"}
        self.assertEqual(dataset_digest(self.dataset), dataset_digest(changed))
        self.assertNotIn("generatedAt", comparable_dataset(changed))

    def test_data_difference_reports_json_path(self):
        changed = {
            **self.dataset,
            "groupMatches": [{"id": "match-a", "probability": 0.6}],
        }
        difference = first_difference(
            comparable_dataset(self.dataset),
            comparable_dataset(changed),
        )
        self.assertIn("groupMatches.[0].probability", difference)

    def test_source_commit_and_asset_data_must_match(self):
        validate_dataset(self.dataset, dict(self.dataset), "abc123abc123")
        with self.assertRaisesRegex(ValueError, "Source commit mismatch"):
            validate_dataset(self.dataset, dict(self.dataset), "def456def456")

    def test_cache_buster_preserves_existing_query(self):
        self.assertEqual(
            cache_busted_url("https://example.test/data.json?x=1", "abc123"),
            "https://example.test/data.json?x=1&source=abc123",
        )


if __name__ == "__main__":
    unittest.main()
