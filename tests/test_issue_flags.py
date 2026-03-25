from __future__ import annotations

import unittest

import pandas as pd

from recon.ml_module import FinancialReconciler


class IssueFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reconciler = FinancialReconciler(date_tolerance_days=5)

    def test_build_issue_flags_marks_rounding_and_medium_confidence(self) -> None:
        bank_row = pd.Series({"amount": 99.83, "type": "DEBIT"})
        register_row = pd.Series({"amount": 99.78, "type": "DR"})

        flags = self.reconciler.build_issue_flags(bank_row, register_row, 0.72, 1)

        self.assertEqual(flags, "ROUNDING_DIFFERENCE|MEDIUM_CONFIDENCE")

    def test_build_issue_flags_marks_date_difference_and_low_confidence(self) -> None:
        bank_row = pd.Series({"amount": 57.56, "type": "DEBIT"})
        register_row = pd.Series({"amount": 57.56, "type": "DR"})

        flags = self.reconciler.build_issue_flags(bank_row, register_row, 0.54, 7)

        self.assertEqual(flags, "DATE_DIFFERENCE|LOW_CONFIDENCE")

    def test_build_issue_flags_marks_unmatched(self) -> None:
        bank_row = pd.Series({"amount": 10.0, "type": "DEBIT"})

        flags = self.reconciler.build_issue_flags(bank_row, None, 0.0, None)

        self.assertEqual(flags, "UNMATCHED")


if __name__ == "__main__":
    unittest.main()
