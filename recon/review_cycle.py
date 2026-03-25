from __future__ import annotations

import pandas as pd

from recon.preprocessing import get_attributes

# The review cycle functions implement the "review -> improve" part of the assignment,
#  allowing for a simulated analyst review process and then feeding validated matches back 
# into the learning corpus to improve future reconciliation performance.
def build_review_queue(matches_df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    """
    Create a review queue for ML matches.

    High-confidence ML matches are auto-validated to simulate analyst approval,
    while the remainder are surfaced as `needs_review` so the report shows the
    full `match -> review -> improve` lifecycle required by the assignment.
    """
    review_candidates = matches_df[matches_df["match_method"] == "ML (SVD+Cosine)"].copy() # We focus the review process on ML matches, as these are the ones that benefit most from feedback and improvement. Unique amount matches are already considered high-confidence and don't require review.
    if review_candidates.empty:
        return review_candidates
# The confidence threshold is set to 0.8, which means that any ML match with a confidence score of 0.8 or higher will be automatically validated, 
# simulating an analyst's approval of high-confidence matches. Matches below this threshold will be marked as `needs_review`,
#  indicating that they require further scrutiny before being validated. This allows us to simulate a realistic review process 
# where not all ML matches are accepted without question, and provides a mechanism for continuous improvement by feeding validated matches back into the learning corpus.
    review_candidates["review_status"] = review_candidates["confidence"].apply(
        lambda value: "validated" if value >= confidence_threshold else "needs_review"
    )
    review_candidates["review_round"] = 1
    return review_candidates

def ingest_validated_matches(
    reconciler,
    review_queue_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
) -> int:
    """
    Feed validated review decisions back into the learning corpus.

    This is the mechanism that lets the system improve over time: once a match
    is trusted, its aligned descriptors and date lag become additional training
    evidence for later reconciliation passes.
    """
    validated = review_queue_df[review_queue_df["review_status"] == "validated"]
    if validated.empty:
        return 0

    bank_lookup = bank_df.set_index("transaction_id")
    register_lookup = register_df.set_index("transaction_id")
    added = 0
# We iterate over the validated matches and extract their attributes to add to the reconciler's parallel corpus. This corpus is used for training the ML model, so by adding validated matches, we are effectively improving the model's training data with real examples of successful reconciliations. The date lag is also included as a feature, which can help the model learn temporal patterns in the data.
    for row in validated.itertuples(index=False):
        if pd.isna(row.reg_id):
            continue
        bank_row = bank_lookup.loc[row.bank_id]
        register_row = register_lookup.loc[row.reg_id]
        lag = int((bank_row["date"] - register_row["date"]).days)
        reconciler.parallel_corpus.append((get_attributes(bank_row), get_attributes(register_row), lag))
        added += 1

    return added
