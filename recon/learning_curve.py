from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd

from recon.ml_module import FinancialReconciler
from recon.preprocessing import get_attributes
from recon.similarity import choose_best_candidate


def _normalize_type(value: str) -> str:
    text = str(value).lower()
    return "dr" if "debit" in text or "dr" in text else "cr"


def evaluate_learning_curve(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    unique_matches_df: pd.DataFrame,
    train_fractions: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    holdout_size: int = 60,
    n_components: int = 15,
    date_tolerance_days: int = 5,
) -> pd.DataFrame:
    """
    Evaluate how the ML matcher improves as more validated pairs are available.

    The evaluation holds out a fixed set of high-confidence unique matches and
    measures whether the semantic matcher retrieves the correct register row from
    a realistic candidate pool. This gives a clearer learning signal than the
    full reconciliation metrics, which are already near-saturated on this dataset.
    """
    if unique_matches_df.empty:
        return pd.DataFrame()

    unique_matches_df = unique_matches_df.reset_index(drop=True)
    effective_holdout = min(holdout_size, max(1, len(unique_matches_df) // 5))
    validation_matches = unique_matches_df.tail(effective_holdout)
    training_matches = unique_matches_df.iloc[:-effective_holdout]

    if training_matches.empty or validation_matches.empty:
        return pd.DataFrame()

    bank_lookup = bank_df.set_index("transaction_id")
    register_lookup = register_df.set_index("transaction_id")
    rows = []

    for fraction in train_fractions:
        model = FinancialReconciler(
            n_components=n_components,
            date_tolerance_days=date_tolerance_days,
        )
        keep_count = max(1, int(len(training_matches) * fraction))
        subset = training_matches.head(keep_count)

        for match in subset.itertuples(index=False):
            bank_row = bank_lookup.loc[match.bank_id]
            register_row = register_lookup.loc[match.reg_id]
            lag = int((bank_row["date"] - register_row["date"]).days)
            model.parallel_corpus.append((get_attributes(bank_row), get_attributes(register_row), lag))

        with redirect_stdout(io.StringIO()):
            model.train_ml_model()

        correct = 0
        total = 0

        for match in validation_matches.itertuples(index=False):
            bank_row = bank_lookup.loc[match.bank_id]
            true_register_id = match.reg_id

            candidate_pool = register_df[
                (register_df["type"].map(_normalize_type) == _normalize_type(bank_row["type"]))
                & (abs((register_df["date"] - bank_row["date"]).dt.days) <= date_tolerance_days)
            ]

            best_match, _ = choose_best_candidate(
                bank_row,
                candidate_pool,
                model.attribute_columns,
                model.u_matrix,
                model.date_tolerance_days,
            )
            total += 1
            if best_match is not None and best_match["transaction_id"] == true_register_id:
                correct += 1

        rows.append(
            {
                "training_fraction": fraction,
                "training_pairs": keep_count,
                "validation_pairs": total,
                "correct_predictions": correct,
                "validation_accuracy": round(correct / total, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows)
