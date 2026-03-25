from __future__ import annotations

import unittest

import pandas as pd

from recon.learning_curve import evaluate_learning_curve


class LearningCurveTests(unittest.TestCase):
    def test_evaluate_learning_curve_returns_expected_columns(self) -> None:
        bank_df = pd.DataFrame(
            {
                "transaction_id": ["B0001", "B0002", "B0003", "B0004", "B0005", "B0006"],
                "date": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
                ),
                "description": ["BP GAS", "KROGER", "NETFLIX", "CAFE", "SHELL", "SAFEWAY"],
                "amount": [10, 20, 30, 40, 50, 60],
                "type": ["DEBIT"] * 6,
            }
        )
        register_df = pd.DataFrame(
            {
                "transaction_id": ["R0001", "R0002", "R0003", "R0004", "R0005", "R0006"],
                "date": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
                ),
                "description": ["Gas", "Groceries", "Subscription", "Lunch", "Fuel", "Food shopping"],
                "amount": [10, 20, 30, 40, 50, 60],
                "type": ["DR"] * 6,
            }
        )
        unique_matches_df = pd.DataFrame(
            {
                "bank_id": ["B0001", "B0002", "B0003", "B0004", "B0005", "B0006"],
                "reg_id": ["R0001", "R0002", "R0003", "R0004", "R0005", "R0006"],
            }
        )

        learning_curve_df = evaluate_learning_curve(
            bank_df,
            register_df,
            unique_matches_df,
            train_fractions=(0.5, 1.0),
            holdout_size=2,
        )

        self.assertEqual(
            list(learning_curve_df.columns),
            ["training_fraction", "training_pairs", "validation_pairs", "correct_predictions", "validation_accuracy"],
        )
        self.assertEqual(len(learning_curve_df), 2)


if __name__ == "__main__":
    unittest.main()
