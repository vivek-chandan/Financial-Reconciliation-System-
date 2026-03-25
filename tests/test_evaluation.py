from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recon.evaluation import (
    build_ground_truth,
    compute_classification_metrics,
    extract_id_suffix,
    load_ground_truth_csv,
)

# These tests validate the core evaluation logic, ensuring that ID suffix extraction,
# ground truth construction, and metric computation work as expected.
class EvaluationTests(unittest.TestCase):
    def test_extract_id_suffix_returns_numeric_tail(self) -> None:
        self.assertEqual(extract_id_suffix("B0232"), "0232")
        self.assertEqual(extract_id_suffix("R7"), "7")
        self.assertIsNone(extract_id_suffix("BANK"))

# The ground truth construction relies on matching numeric suffixes in the synthetic IDs,
#  so this test ensures that the function correctly identifies pairs of transactions 
# that should be considered matches for evaluation purposes.
    def test_build_ground_truth_pairs_matching_suffixes(self) -> None:
        bank_df = pd.DataFrame({"transaction_id": ["B0001", "B0002", "B9999"]})
        register_df = pd.DataFrame({"transaction_id": ["R0001", "R0002", "R0003"]})

        truth_df = build_ground_truth(bank_df, register_df)

        expected = pd.DataFrame(
            [
                {"bank_id": "B0001", "reg_id": "R0001"},
                {"bank_id": "B0002", "reg_id": "R0002"},
            ]
        )
        pd.testing.assert_frame_equal(truth_df.reset_index(drop=True), expected)

# The metric computation treats reconciliation as a pair-classification problem,
#  so this test checks that the function correctly computes precision, recall,
#  and F1 score based on the overlap of predicted and true pairs of matches.
    def test_compute_classification_metrics_uses_pair_overlap(self) -> None:
        report_df = pd.DataFrame(
            [
                {"bank_id": "B0001", "reg_id": "R0001"},
                {"bank_id": "B0002", "reg_id": "R0009"},
                {"bank_id": "B0003", "reg_id": None},
            ]
        )
        ground_truth_df = pd.DataFrame(
            [
                {"bank_id": "B0001", "reg_id": "R0001"},
                {"bank_id": "B0002", "reg_id": "R0002"},
                {"bank_id": "B0003", "reg_id": "R0003"},
            ]
        )

        metrics = compute_classification_metrics(report_df, ground_truth_df)

        self.assertEqual(metrics["correct_matches"], 1.0)
        self.assertEqual(metrics["predicted_matches"], 2.0)
        self.assertEqual(metrics["true_matches"], 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["f1_score"], 0.4)

    def test_load_ground_truth_csv_normalizes_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "ground_truth.csv"
            pd.DataFrame(
                [
                    {"bank_transaction_id": "B0001", "register_transaction_id": "R0001"},
                    {"bank_transaction_id": "B0002", "register_transaction_id": "R0002"},
                ]
            ).to_csv(csv_path, index=False)

            ground_truth_df = load_ground_truth_csv(csv_path)

        expected = pd.DataFrame(
            [
                {"bank_id": "B0001", "reg_id": "R0001"},
                {"bank_id": "B0002", "reg_id": "R0002"},
            ]
        )
        pd.testing.assert_frame_equal(ground_truth_df, expected)


if __name__ == "__main__":
    unittest.main()
