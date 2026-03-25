from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from recon.input_loader import resolve_input_path


ID_SUFFIX_PATTERN = re.compile(r"(\d+)$")

# The evaluation approach here is intentionally simple and transparent,
#  to focus on the core reconciliation logic.

def extract_id_suffix(transaction_id: str) -> str | None:
    """Extract the numeric suffix used by the synthetic dataset IDs."""
    match = ID_SUFFIX_PATTERN.search(str(transaction_id))
    return match.group(1) if match else None


def load_ground_truth_csv(ground_truth_file: str | Path) -> pd.DataFrame:
    """
    Load a labeled ground-truth CSV and normalize it to `bank_id` / `reg_id`.

    Accepted column pairs:
    - `bank_id`, `reg_id`
    - `bank_transaction_id`, `register_transaction_id`
    """
    ground_truth_path = resolve_input_path(str(ground_truth_file))
    ground_truth_df = pd.read_csv(ground_truth_path)

    rename_map = {}
    if "bank_transaction_id" in ground_truth_df.columns:
        rename_map["bank_transaction_id"] = "bank_id"
    if "register_transaction_id" in ground_truth_df.columns:
        rename_map["register_transaction_id"] = "reg_id"
    ground_truth_df = ground_truth_df.rename(columns=rename_map)

    required_columns = {"bank_id", "reg_id"}
    missing_columns = required_columns - set(ground_truth_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Ground truth file must contain bank/register match columns. Missing: {missing_list}"
        )

    return ground_truth_df[["bank_id", "reg_id"]].dropna().drop_duplicates().reset_index(drop=True)


# The "ground truth" is simulated by aligning the synthetic IDs 
# from both datasets.
def build_ground_truth(bank_df: pd.DataFrame, register_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simulated ground truth from aligned synthetic IDs.
    This matches the assignment's "simulate some ground truth" requirement.
    """
    register_lookup = {}
    for transaction_id in register_df["transaction_id"]:
        suffix = extract_id_suffix(transaction_id)
        if suffix is not None and suffix not in register_lookup:
            register_lookup[suffix] = transaction_id

    rows = []
    for transaction_id in bank_df["transaction_id"]:
        suffix = extract_id_suffix(transaction_id)
        # If the suffix matches one in the register, 
        # we treat it as a "true match" for evaluation purposes.
        if suffix is not None and suffix in register_lookup:
            rows.append({"bank_id": transaction_id, "reg_id": register_lookup[suffix]})

    return pd.DataFrame(rows)

# The evaluation metrics are based on set comparisons of predicted vs. true matches.
def compute_classification_metrics(report_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute precision, recall, and F1 from predicted and expected match pairs.

    The function treats reconciliation as a pair-classification problem:
    each `(bank_id, reg_id)` proposed by the system is compared against the
    expected pair set, and set overlap determines the standard metrics.
    """
    predicted_pairs = {
        (row.bank_id, row.reg_id)
        for row in report_df[report_df["reg_id"].notna()][["bank_id", "reg_id"]].itertuples(index=False)
    }
    true_pairs = {
        (row.bank_id, row.reg_id)
        for row in ground_truth_df[["bank_id", "reg_id"]].itertuples(index=False)
    }

    correctly_matched = predicted_pairs & true_pairs
    precision = len(correctly_matched) / len(predicted_pairs) if predicted_pairs else 0.0
    recall = len(correctly_matched) / len(true_pairs) if true_pairs else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "correct_matches": float(len(correctly_matched)),
        "predicted_matches": float(len(predicted_pairs)),
        "true_matches": float(len(true_pairs)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
