from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recon.preprocessing import get_attributes
from recon.vectorization import vectorize_attributes


def compute_similarity(bank_row, register_row, attribute_columns, projection_matrix, date_tolerance_days: int) -> float:
    """
    Calculate match similarity with an explicit date-gap penalty.

    Raw text similarity alone can create cross-swaps when two merchants look
    alike but occurred months apart. Penalizing larger date gaps keeps the ML
    stage aligned with the reconciliation domain assumptions.
    """
    bank_attributes = get_attributes(bank_row)
    date_gap = abs((bank_row["date"] - register_row["date"]).days)
    date_penalty = min(date_gap / max(date_tolerance_days, 1), 1.0) * 0.35
# The date penalty is calculated as a proportion of the date gap relative to the specified tolerance, capped at 1.0, and then scaled by a factor (0.35 in this case) 
# to determine how much it should reduce the similarity score. This approach allows for a gradual decrease in similarity as the date gap increases,
#  while still giving some credit for matches that are close in time.
    if projection_matrix is not None and attribute_columns is not None:
        bank_vector = vectorize_attributes(bank_attributes, attribute_columns)
        bank_semantic = (bank_vector @ projection_matrix).reshape(1, -1)
        register_vector = vectorize_attributes(get_attributes(register_row), attribute_columns)
        register_semantic = (register_vector @ projection_matrix).reshape(1, -1)
        base_similarity = float(cosine_similarity(bank_semantic, register_semantic)[0][0])
        return max(0.0, min(1.0, base_similarity - date_penalty))
# If no projection matrix is available, we fall back to a simple token overlap similarity, which is less sophisticated but 
# still provides a basic measure of similarity based on shared attributes. 
# The date penalty is applied in the same way to ensure that temporal proximity is still factored into the similarity score.
    register_attributes = get_attributes(register_row)
    common = set(bank_attributes.keys()) & set(register_attributes.keys())
    denominator = max(len(bank_attributes), len(register_attributes))
    base_similarity = len(common) / denominator if denominator > 0 else 0.5
    return max(0.0, min(1.0, base_similarity - date_penalty))


def choose_best_candidate(bank_row, valid_pool, attribute_columns, projection_matrix, date_tolerance_days: int):
    """Select the strongest candidate and prefer the closest date on exact ties."""
    best_similarity = -1.0
    best_match = None
    ties = []

    for _, register_row in valid_pool.iterrows():
        similarity = compute_similarity(
            bank_row,
            register_row,
            attribute_columns,
            projection_matrix,
            date_tolerance_days,
        )
        if similarity > best_similarity + 1e-7: # We use a small epsilon to avoid floating-point precision issues when comparing similarity scores. This ensures that we only consider a new best match if it is meaningfully better than the current best, rather than being a negligible difference due to rounding.
            best_similarity = similarity
            best_match = register_row
            ties = [register_row]
        elif abs(similarity - best_similarity) < 1e-7: # If the similarity score is effectively the same as the current best (within a small epsilon), we consider it a tie and add it to the list of ties. This allows us to handle cases where multiple candidates have very similar similarity scores, and we can then apply a secondary criterion (date proximity) to break the tie.
            ties.append(register_row)

    if len(ties) > 1:
        ties.sort(key=lambda row: abs((bank_row["date"] - row["date"]).days))
        best_match = ties[0]

    return best_match, round(float(best_similarity), 4)
