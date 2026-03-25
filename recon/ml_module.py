from __future__ import annotations

import numpy as np
import pandas as pd

from recon.preprocessing import get_attributes, preprocess_data
from recon.similarity import choose_best_candidate
from recon.vectorization import train_svd_model


class FinancialReconciler:
    """
    Financial reconciliation system using unsupervised learning.
    Matches bank transactions to check register entries using:
    1. Unique amount matching (high confidence)
    2. SVD-based semantic matching (ML approach)
    """
# The class maintains state for matched transactions and training data,
# allowing for iterative improvement through review and retraining.
    def __init__(self, n_components: int = 15, date_tolerance_days: int = 5):
        self.n_components = n_components
        self.date_tolerance_days = date_tolerance_days
        self.u_matrix = None      # SVD projection matrix
        self.attribute_columns = None # Columns used for ML matching
        self.parallel_corpus = []  # List of (bank_attrs, reg_attrs, date_lag) for matched pairs
        self.matched_bank_ids = set() # Track matched bank transaction IDs for one-to-one matching
        self.matched_reg_ids = set()  # Track matched register transaction IDs for one-to-one matching
        self.stats = {
            "total_bank": 0,
            "total_register": 0,
            "unique_matches": 0,
            "ml_matches": 0,
            "unmatched": 0,
            "date_warnings": 0,
            "duplicates_removed": 0,
        }

    def build_issue_flags(
        self,
        bank_row: pd.Series,
        register_row: pd.Series | None,
        confidence: float,
        date_lag_days: int | None,
        amount_threshold: float = 0.05,
    ) -> str:
        """
        Build concise issue flags for output/reporting.

        These flags make potential review concerns explicit instead of leaving
        them implicit in confidence scores and free-form notes.
        """
        if register_row is None:
            return "UNMATCHED"

        flags: list[str] = []
        if date_lag_days is not None and abs(date_lag_days) > self.date_tolerance_days:
            flags.append("DATE_DIFFERENCE")

        amount_gap = abs(float(bank_row["amount"]) - float(register_row["amount"]))
        if 0 < amount_gap <= amount_threshold:
            flags.append("ROUNDING_DIFFERENCE")
        elif amount_gap > amount_threshold:
            flags.append("AMOUNT_DIFFERENCE")

        if confidence < 0.70:
            flags.append("LOW_CONFIDENCE")
        elif confidence < 0.85:
            flags.append("MEDIUM_CONFIDENCE")

        bank_type = "dr" if "debit" in str(bank_row["type"]).lower() or "dr" in str(bank_row["type"]).lower() else "cr"
        register_type = "dr" if "debit" in str(register_row["type"]).lower() or "dr" in str(register_row["type"]).lower() else "cr"
        if bank_type != register_type:
            flags.append("TYPE_MISMATCH")

        return "|".join(flags) if flags else "OK"

    def reset_matching_state(self) -> None:
        """Reset one-to-one bookkeeping for a fresh reconciliation pass."""
        self.matched_bank_ids = set() # Reset matched bank transaction IDs
        self.matched_reg_ids = set()  # Reset matched register transaction IDs

    def match_unique_amounts(self, bank_df: pd.DataFrame, register_df: pd.DataFrame) -> list[dict]:
        """Phase 1: Match transactions with unique amounts."""
        print("\n" + "=" * 70)
        print("PHASE 1: UNIQUE AMOUNT MATCHING")
        print("=" * 70)

        matches = []
        bank_counts = bank_df["amount"].value_counts()  # Count unique amounts in bank
        register_counts = register_df["amount"].value_counts() # Count unique amounts in register

        common_unique = np.intersect1d(
            bank_counts[bank_counts == 1].index,  # Filter to unique amounts
            register_counts[register_counts == 1].index, # Filter to unique amounts
        )

        print(f"Found {len(common_unique)} unique amounts to match...")
  # For each unique amount, we find the corresponding transactions in both datasets,
  # and create a matched pair with confidence 1.0. We also check for date lag
  #  and log warnings if it exceeds the tolerance.
        for amount in common_unique: 
            bank_row = bank_df[bank_df["amount"] == amount].iloc[0]
            register_row = register_df[register_df["amount"] == amount].iloc[0]

            bank_id = bank_row["transaction_id"]
            register_id = register_row["transaction_id"]
            lag = (bank_row["date"] - register_row["date"]).days

            self.matched_bank_ids.add(bank_id)   # Mark bank transaction as matched
            self.matched_reg_ids.add(register_id) # Mark register transaction as matched
            self.parallel_corpus.append((get_attributes(bank_row), get_attributes(register_row), lag)) # Add to training corpus for ML phase

            note = "Unique Amount Match"
            if abs(lag) > self.date_tolerance_days:
                note += f" (Date lag: {lag} days)"
                self.stats["date_warnings"] += 1

            matches.append(
                {
                    "bank_id": bank_id,
                    "reg_id": register_id,
                    "confidence": 1.0,
                    "match_method": "Unique Amount",
                    "date_lag_days": lag,
                    "notes": note,
                    "issue_flags": self.build_issue_flags(bank_row, register_row, 1.0, lag),
                }
            )

        self.stats["unique_matches"] = len(matches)
        print(f" Matched {len(matches)} transaction pairs with unique amounts")
        return matches

# The ML training phase uses the matched pairs from 
# unique amount matching to train an SVD model.
    def train_ml_model(self) -> None:
        """Phase 2: Train the SVD model from matched pairs."""
        print("\n" + "=" * 70)
        print("PHASE 2: TRAINING ML MODEL")
        print("=" * 70)
# The training function returns the projection matrix, the attribute columns used for matching,
        projection_matrix, attribute_columns, explained_variance = train_svd_model(
            self.parallel_corpus, # List of (bank_attrs, reg_attrs, date_lag) for matched pairs
            self.n_components,    # Number of SVD components to retain
        )
        self.u_matrix = projection_matrix  # Store the SVD projection matrix for later use in matching
        self.attribute_columns = attribute_columns  # Store the attribute columns used for matching (e.g., description, category)

        if self.attribute_columns is None:
            print("  No training data available. ML matching will use basic similarity.")
            return

        if self.u_matrix is None:
            print("  Limited training data available. ML matching will use basic similarity.")
            print(f"  Training examples: {len(self.parallel_corpus)}")
            return

        component_count = self.u_matrix.shape[1]
        print(f" Trained SVD model with {component_count} components")
        print(f"  Variance explained: {explained_variance:.1%}")
        print(f"  Training examples: {len(self.parallel_corpus)}")

# The ML reconciliation phase applies the trained SVD model to find the best match for each unmatched bank transaction,
    def reconcile_ml(self, bank_row: pd.Series, register_candidates: pd.DataFrame) -> dict | None:
        """Phase 3: Match using ML similarity."""
        bank_id = bank_row["transaction_id"]
        if bank_id in self.matched_bank_ids: # Skip if already matched in unique amount phase
            return None

        amount_threshold = 0.05
        date_window_days = self.date_tolerance_days # We allow a wider date window for ML matches, but will log warnings if lag exceeds tolerance
 # We filter the register candidates to those that are within the amount threshold and date window,
 #  and that haven't already been matched to ensure one-to-one matching.       
        valid_pool = register_candidates[
            (abs(register_candidates["amount"] - bank_row["amount"]) <= amount_threshold)
            & (~register_candidates["transaction_id"].isin(self.matched_reg_ids))
            & (abs((register_candidates["date"] - bank_row["date"]).dt.days) <= date_window_days)
        ]

        if valid_pool.empty:
            return None

        best_match, best_similarity = choose_best_candidate(
            bank_row,
            valid_pool,
            self.attribute_columns,
            self.u_matrix,
            self.date_tolerance_days,
        )

        if best_match is None:
            return None

        register_id = best_match["transaction_id"]
        self.matched_bank_ids.add(bank_id)
        self.matched_reg_ids.add(register_id)

        lag = (bank_row["date"] - best_match["date"]).days
        note = "ML Matched"
        # We log a warning if the date lag exceeds the tolerance, as this may indicate a less reliable match.
        if abs(lag) > self.date_tolerance_days:
            note += f" (Date lag: {lag} days)"
            self.stats["date_warnings"] += 1

        return {
            "bank_id": bank_id,
            "reg_id": register_id,
            "confidence": best_similarity,
            "match_method": "ML (SVD+Cosine)",
            "date_lag_days": lag,
            "notes": note,
            "issue_flags": self.build_issue_flags(bank_row, best_match, best_similarity, lag),
        }
# The main reconciliation function runs the entire pipeline: it preprocesses the data, performs unique amount matching,
# trains the ML model, and then applies ML-based matching to the remaining unmatched transactions. It also collects statistics on the process.
    def reconcile_all(self, bank_df: pd.DataFrame, register_df: pd.DataFrame):
        """Run the complete reconciliation pipeline."""
        print("\n" + "=" * 70)
        print("FINANCIAL RECONCILIATION SYSTEM")
        print("=" * 70)

        bank_clean = preprocess_data(bank_df, "Bank Statements", self.stats)
        register_clean = preprocess_data(register_df, "Check Register", self.stats)

        self.stats["total_bank"] = len(bank_clean)
        self.stats["total_register"] = len(register_clean)

        unique_matches = self.match_unique_amounts(bank_clean, register_clean)
        self.train_ml_model()

        print("\n" + "=" * 70)
        print("PHASE 3: ML-BASED MATCHING")
        print("=" * 70)

        ml_matches = []
        unmatched = []
# We iterate over the bank transactions that were not matched in the unique amount 
# phase, and attempt to find matches using the ML model. If a match is found, 
# we add it to the ml_matches list; if not, we log it as unmatched.
        for _, row in bank_clean.iterrows():
            if row["transaction_id"] not in self.matched_bank_ids:
                match = self.reconcile_ml(row, register_clean)
                if match:
                    ml_matches.append(match)
                else:
                    unmatched.append(
                        {
                            "bank_id": row["transaction_id"],
                            "reg_id": None,
                            "confidence": 0.0,
                            "match_method": "Unmatched",
                            "date_lag_days": None,
                            "notes": "No available match found",
                            "issue_flags": "UNMATCHED",
                        }
                    )

        self.stats["ml_matches"] = len(ml_matches)
        self.stats["unmatched"] = len(unmatched)

        print(f" Matched {len(ml_matches)} transactions using ML")
        print(f" {len(unmatched)} transactions remain unmatched")

        all_matches = unique_matches + ml_matches + unmatched
        return pd.DataFrame(all_matches), bank_clean, register_clean
# The improvement function allows for retraining the ML model with reviewed matches,
# and then rerunning the reconciliation process to see if more matches can be found.
    def improve_with_review(
        self,
        bank_df: pd.DataFrame,
        register_df: pd.DataFrame,
        validated_match_count: int,
    ):
        """Retrain with reviewed matches and rerun the pipeline."""
        print("\n" + "=" * 70)
        print("PHASE 4: REVIEW -> IMPROVE")
        print("=" * 70)
        print(f"Adding {validated_match_count} validated ML matches back into training data...")

        self.reset_matching_state()
        self.train_ml_model()

        unique_matches = self.match_unique_amounts(bank_df, register_df)

        ml_matches = []
        unmatched = []
        for _, row in bank_df.iterrows():
            if row["transaction_id"] not in self.matched_bank_ids:
                match = self.reconcile_ml(row, register_df)
                if match:
                    ml_matches.append(match)
                else:
                    unmatched.append(
                        {
                            "bank_id": row["transaction_id"],
                            "reg_id": None,
                            "confidence": 0.0,
                            "match_method": "Unmatched",
                            "date_lag_days": None,
                            "notes": "No available match found",
                            "issue_flags": "UNMATCHED",
                        }
                    )

        self.stats["unique_matches"] = len(unique_matches)
        self.stats["ml_matches"] = len(ml_matches)
        self.stats["unmatched"] = len(unmatched)

        print(f" Improved run complete with {len(ml_matches)} ML matches and {len(unmatched)} unmatched")
        return pd.DataFrame(unique_matches + ml_matches + unmatched)
# Finally, the print_summary function provides a comprehensive overview of the reconciliation process, including input data statistics, matching results, and any warnings about date lags.
    def print_summary(self) -> None:
        """Print final summary statistics."""
        print("\n" + "=" * 70)
        print("RECONCILIATION SUMMARY")
        print("=" * 70)
        print("\nInput Data:")
        print(f"  Bank transactions: {self.stats['total_bank']}")
        print(f"  Register transactions: {self.stats['total_register']}")
        print(f"  Duplicates removed: {self.stats['duplicates_removed']}")

        print("\nMatching Results:")
        print(f"  Unique amount matches: {self.stats['unique_matches']}")
        print(f"  ML matches: {self.stats['ml_matches']}")
        print(f"  Unmatched: {self.stats['unmatched']}")
        print(f"  Total matched: {self.stats['unique_matches'] + self.stats['ml_matches']}")

        total_matched = self.stats["unique_matches"] + self.stats["ml_matches"]
        if self.stats["total_bank"] > 0:
            match_rate = total_matched / self.stats["total_bank"] * 100
            print(f"\n  Match Rate: {match_rate:.1f}%")

        if self.stats["date_warnings"] > 0:
            print("\nWarnings:")
            print(f"  Transactions with date lag > {self.date_tolerance_days} days: {self.stats['date_warnings']}")
