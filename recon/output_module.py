from __future__ import annotations

from pathlib import Path

import pandas as pd

from recon.evaluation import build_ground_truth, compute_classification_metrics


def create_final_report(matches_df: pd.DataFrame, bank_df: pd.DataFrame, register_df: pd.DataFrame) -> pd.DataFrame:
    """Create the final merged report without changing the original column layout."""
    print("\n" + "=" * 70)
    print("CREATING FINAL REPORT")
    print("=" * 70)

    report_source = matches_df.rename(columns={"notes": "match_notes"})

    report = report_source.merge(
        bank_df,
        left_on="bank_id",
        right_on="transaction_id",
        how="left",
        suffixes=("", "_DROP"),
    )

    report = report.merge(
        register_df,
        left_on="reg_id",
        right_on="transaction_id",
        how="left",
        suffixes=("_bank", "_reg"),
    )

    report = report[[column for column in report.columns if not column.endswith("_DROP")]]

    column_order = [
        "bank_id",
        "reg_id",
        "confidence",
        "match_method",
        "date_lag_days",
        "match_notes",
        "date_bank",
        "description_bank",
        "amount_bank",
        "type_bank",
        "balance",
        "date_reg",
        "description_reg",
        "amount_reg",
        "type_reg",
        "category",
    ]

    final_columns = [column for column in column_order if column in report.columns]
    final_columns += [column for column in report.columns if column not in final_columns]
    final_columns = [column for column in final_columns if column != "notes_reg"]
    report = report[final_columns]

    print(f"✓ Generated report with {len(report)} rows and {len(report.columns)} columns")
    return report


def calculate_metrics(report_df: pd.DataFrame, date_tolerance_days: int, ground_truth_df: pd.DataFrame | None = None) -> None:
    """Print matching metrics without altering the report format."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    matched = report_df[report_df["reg_id"].notna()]
    total_matches = len(matched)
    total_bank = len(report_df)

    print("\nMatching Statistics:")
    print(f"  Total bank transactions: {total_bank}")
    print(f"  Matched: {total_matches} ({total_matches / total_bank * 100:.1f}%)")
    print(f"  Unmatched: {total_bank - total_matches} ({(total_bank - total_matches) / total_bank * 100:.1f}%)")

    if ground_truth_df is not None:
        metrics = compute_classification_metrics(report_df, ground_truth_df)
        print("\nEvaluation Against Ground Truth:")
        print(f"  Correctly matched: {int(metrics['correct_matches'])}")
        print(f"  System matches: {int(metrics['predicted_matches'])}")
        print(f"  Transactions that should match: {int(metrics['true_matches'])}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    print("\nConfidence Distribution:")
    print(f"  High confidence (≥0.9): {len(matched[matched['confidence'] >= 0.9])}")
    print(f"  Medium confidence (0.7-0.9): {len(matched[(matched['confidence'] >= 0.7) & (matched['confidence'] < 0.9)])}")
    print(f"  Low confidence (<0.7): {len(matched[matched['confidence'] < 0.7])}")

    if "date_lag_days" in matched.columns:
        lag_data = matched[matched["date_lag_days"].notna()]["date_lag_days"]
        if len(lag_data) > 0:
            print("\nDate Lag Analysis:")
            print(f"  Mean: {lag_data.mean():.1f} days")
            print(f"  Median: {lag_data.median():.1f} days")
            print(f"  Range: {lag_data.min():.0f} to {lag_data.max():.0f} days")
            print(f"  Outside tolerance (>{date_tolerance_days} days): {len(lag_data[abs(lag_data) > date_tolerance_days])}")


def save_output(report_df: pd.DataFrame, output_file: str) -> Path:
    """Save the final report while preserving the same CSV format."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=True)
    return output_path


def save_review_queue(review_df: pd.DataFrame, output_file: str) -> Path:
    """Save review candidates for the review/improve phase."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_path, index=False)
    return output_path


def build_default_ground_truth(bank_df: pd.DataFrame, register_df: pd.DataFrame) -> pd.DataFrame:
    """Construct simulated ground truth when a labeled file is not provided."""
    return build_ground_truth(bank_df, register_df)
