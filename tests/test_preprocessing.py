from __future__ import annotations

import unittest

from recon.preprocessing import tokenize_description

# These tests validate the core preprocessing logic, ensuring that transaction descriptions 
# are tokenized and normalized correctly to improve matching performance.
class PreprocessingTests(unittest.TestCase):
    def test_tokenize_description_removes_numeric_noise(self) -> None:
        self.assertEqual(tokenize_description("BP GAS #5199"), ["gas", "gas"])

# The normalization rules are designed to standardize common variants of terms that are
#  likely to appear in transaction descriptions. This test checks that the function correctly maps 
# different lexical variants to their normalized forms, which helps create a more consistent set of attributes for matching.
    def test_tokenize_description_normalizes_grocery_and_utility_terms(self) -> None:
        self.assertEqual(tokenize_description("SAFEWAY #8208"), ["grocery"])
        self.assertEqual(tokenize_description("ONLINE PMT WATER"), ["utility", "utility"])


if __name__ == "__main__":
    unittest.main()
