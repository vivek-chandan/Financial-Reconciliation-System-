from __future__ import annotations

from pathlib import Path

import pandas as pd

from recon.evaluation import compute_classification_metrics


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
        "issue_flags",
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

    if "issue_flags" in report_df.columns:
        flagged = report_df[report_df["issue_flags"] != "OK"]
        print("\nIssue Flag Summary:")
        print(f"  Flagged rows: {len(flagged)}")
        if len(flagged) > 0:
            print(f"  Date differences: {report_df['issue_flags'].fillna('').str.contains('DATE_DIFFERENCE').sum()}")
            print(f"  Rounding differences: {report_df['issue_flags'].fillna('').str.contains('ROUNDING_DIFFERENCE').sum()}")
            print(f"  Low confidence: {report_df['issue_flags'].fillna('').str.contains('LOW_CONFIDENCE').sum()}")
            print(f"  Medium confidence: {report_df['issue_flags'].fillna('').str.contains('MEDIUM_CONFIDENCE').sum()}")
            print(f"  Unmatched: {report_df['issue_flags'].fillna('').str.contains('UNMATCHED').sum()}")

    if "date_lag_days" in matched.columns:
        lag_data = matched[matched["date_lag_days"].notna()]["date_lag_days"]
        if len(lag_data) > 0:
            print("\nDate Lag Analysis:")
            print(f"  Mean: {lag_data.mean():.1f} days")
            print(f"  Median: {lag_data.median():.1f} days")
            print(f"  Range: {lag_data.min():.0f} to {lag_data.max():.0f} days")
            print(f"  Outside tolerance (>{date_tolerance_days} days): {len(lag_data[abs(lag_data) > date_tolerance_days])}")


def analyze_hardest_transaction_types(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank transaction categories by how difficult they are to reconcile.

    Difficulty is driven by a combination of ML reliance, lower confidence,
    and issue flags. This turns the manual "hardest transaction types" analysis
    into a reproducible artifact for the assignment report.
    """
    analysis_df = report_df.copy()
    analysis_df["category"] = analysis_df["category"].fillna("Unknown")
    analysis_df["has_known_category"] = analysis_df["category"].ne("Unknown")
    analysis_df["is_ml_match"] = analysis_df["match_method"].eq("ML (SVD+Cosine)")
    analysis_df["is_matched"] = analysis_df["reg_id"].notna()
    analysis_df["is_flagged"] = analysis_df["issue_flags"].fillna("OK").ne("OK")
    analysis_df["is_low_confidence"] = analysis_df["issue_flags"].fillna("").str.contains("LOW_CONFIDENCE")
    analysis_df["is_medium_confidence"] = analysis_df["issue_flags"].fillna("").str.contains("MEDIUM_CONFIDENCE")
    analysis_df["is_unmatched"] = analysis_df["issue_flags"].fillna("").str.contains("UNMATCHED")

    summary = (
        analysis_df.groupby("category", dropna=False)
        .agg(
            has_known_category=("has_known_category", "max"),
            total_transactions=("bank_id", "count"),
            matched_transactions=("is_matched", "sum"),
            ml_matches=("is_ml_match", "sum"),
            flagged_transactions=("is_flagged", "sum"),
            low_confidence_matches=("is_low_confidence", "sum"),
            medium_confidence_matches=("is_medium_confidence", "sum"),
            unmatched_transactions=("is_unmatched", "sum"),
            average_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    summary["ml_match_rate"] = (summary["ml_matches"] / summary["total_transactions"]).round(4)
    summary["flag_rate"] = (summary["flagged_transactions"] / summary["total_transactions"]).round(4)
    summary["unmatched_rate"] = (summary["unmatched_transactions"] / summary["total_transactions"]).round(4)
    summary["difficulty_score"] = (
        0.45 * summary["flag_rate"]
        + 0.25 * summary["ml_match_rate"]
        + 0.20 * summary["unmatched_rate"]
        + 0.10 * (1 - summary["average_confidence"].fillna(0))
    ).round(4)

    return summary.sort_values(
        by=["has_known_category", "difficulty_score", "low_confidence_matches", "flagged_transactions", "category"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


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


def save_learning_curve(learning_curve_df: pd.DataFrame, output_file: str) -> Path:
    """Save learning-curve analysis that shows model quality vs training size."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learning_curve_df.to_csv(output_path, index=False)
    return output_path


def save_transaction_type_analysis(analysis_df: pd.DataFrame, output_file: str) -> Path:
    """Save automatic analysis of the hardest transaction types."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_path, index=False)
    return output_path
