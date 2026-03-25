from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from recon.analysis_report import generate_analysis_markdown


class AnalysisReportTests(unittest.TestCase):
    def test_generate_analysis_markdown_writes_expected_sections(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "bank_id": "B1",
                    "reg_id": "R1",
                    "confidence": 0.91,
                    "match_method": "Unique Amount",
                    "category": "Utility",
                    "issue_flags": "OK",
                    "description_bank": "ONLINE PMT WATER",
                    "description_reg": "Water bill",
                    "amount_bank": 100.0,
                },
                {
                    "bank_id": "B2",
                    "reg_id": None,
                    "confidence": 0.0,
                    "match_method": "Unmatched",
                    "category": "Unknown",
                    "issue_flags": "UNMATCHED",
                    "description_bank": "UNKNOWN TX",
                    "description_reg": None,
                    "amount_bank": 25.0,
                },
            ]
        )
        hardest_types_df = pd.DataFrame(
            [
                {
                    "category": "Utility",
                    "difficulty_score": 0.15,
                    "flagged_transactions": 1,
                    "total_transactions": 2,
                    "average_confidence": 0.455,
                }
            ]
        )
        learning_curve_df = pd.DataFrame(
            [
                {"training_pairs": 10, "validation_accuracy": 0.5},
                {"training_pairs": 20, "validation_accuracy": 0.7},
            ]
        )
        metrics = {"precision": 1.0, "recall": 0.5, "f1_score": 0.6667}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "analysis.md"
            written_path = generate_analysis_markdown(
                report_df,
                hardest_types_df,
                learning_curve_df,
                metrics,
                output_path,
            )
            content = written_path.read_text()

        self.assertIn("# Analysis & Documentation", content)
        self.assertIn("## Performance Analysis", content)
        self.assertIn("## Design Decisions", content)
        self.assertIn("## Limitations & Future Improvements", content)
        self.assertIn("Precision: `1.0000`", content)


if __name__ == "__main__":
    unittest.main()
