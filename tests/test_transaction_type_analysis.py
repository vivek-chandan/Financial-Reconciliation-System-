from __future__ import annotations

import unittest

import pandas as pd

from recon.output_module import analyze_hardest_transaction_types


class TransactionTypeAnalysisTests(unittest.TestCase):
    def test_analyze_hardest_transaction_types_ranks_flagged_categories_higher(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "bank_id": "B1",
                    "reg_id": "R1",
                    "category": "Utility",
                    "match_method": "ML (SVD+Cosine)",
                    "confidence": 0.62,
                    "issue_flags": "LOW_CONFIDENCE",
                },
                {
                    "bank_id": "B2",
                    "reg_id": "R2",
                    "category": "Utility",
                    "match_method": "ML (SVD+Cosine)",
                    "confidence": 0.70,
                    "issue_flags": "MEDIUM_CONFIDENCE",
                },
                {
                    "bank_id": "B3",
                    "reg_id": "R3",
                    "category": "Subscription",
                    "match_method": "Unique Amount",
                    "confidence": 1.0,
                    "issue_flags": "OK",
                },
            ]
        )

        analysis_df = analyze_hardest_transaction_types(report_df)

        self.assertEqual(analysis_df.iloc[0]["category"], "Utility")
        self.assertGreater(analysis_df.iloc[0]["difficulty_score"], analysis_df.iloc[1]["difficulty_score"])


if __name__ == "__main__":
    unittest.main()
