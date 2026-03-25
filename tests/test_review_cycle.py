from __future__ import annotations

import unittest

import pandas as pd

from recon.review_cycle import build_review_queue

# These tests validate the review cycle logic, ensuring that high-confidence ML matches are correctly marked as validated and that the review queue is constructed with the appropriate statuses and round numbers.
class ReviewCycleTests(unittest.TestCase):
    def test_build_review_queue_marks_high_confidence_as_validated(self) -> None:
        matches_df = pd.DataFrame(
            [
                {"bank_id": "B1", "reg_id": "R1", "confidence": 0.91, "match_method": "ML (SVD+Cosine)"},
                {"bank_id": "B2", "reg_id": "R2", "confidence": 0.62, "match_method": "ML (SVD+Cosine)"},
                {"bank_id": "B3", "reg_id": "R3", "confidence": 1.0, "match_method": "Unique Amount"},
            ]
        )

# The test checks that the `build_review_queue` function correctly identifies ML matches with a confidence score of 0.85
#  or higher as "validated", while those below the threshold are marked as "needs_review". 
# It also verifies that the review round is set to 1 for all candidates in the review queue.
        review_df = build_review_queue(matches_df, confidence_threshold=0.85)

        self.assertEqual(len(review_df), 2)
        self.assertEqual(review_df.iloc[0]["review_status"], "validated")
        self.assertEqual(review_df.iloc[1]["review_status"], "needs_review")
        self.assertTrue((review_df["review_round"] == 1).all())


if __name__ == "__main__":
    unittest.main()
