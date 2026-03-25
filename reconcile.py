from __future__ import annotations

import argparse

from recon.evaluation import build_ground_truth, load_ground_truth_csv
from recon.input_loader import load_transaction_data
from recon.learning_curve import evaluate_learning_curve
from recon.ml_module import FinancialReconciler
from recon.output_module import (
    calculate_metrics,
    create_final_report,
    save_learning_curve,
    save_output,
    save_review_queue,
)
from recon.review_cycle import build_review_queue, ingest_validated_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run financial reconciliation with review/improve cycle.")
    parser.add_argument("--bank-file", default="bank_statements.csv", help="Bank statements CSV file.")
    parser.add_argument("--register-file", default="check_register.csv", help="Check register CSV file.")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for auto-validating ML review candidates.",
    )
    parser.add_argument(
        "--ground-truth-file",
        default=None,
        help="Optional labeled ground-truth CSV with bank_id/reg_id columns.",
    )
    parser.add_argument(
        "--use-simulated-ground-truth",
        action="store_true",
        help="Use ID-suffix-based simulated ground truth when no labeled file is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading data files...")
    bank_df, register_df = load_transaction_data(args.bank_file, args.register_file)

    reconciler = FinancialReconciler(n_components=15, date_tolerance_days=5)
    matches_df, bank_clean, register_clean = reconciler.reconcile_all(bank_df, register_df)
    unique_matches_df = matches_df[matches_df["match_method"] == "Unique Amount"].copy()
    learning_curve_df = evaluate_learning_curve(
        bank_clean,
        register_clean,
        unique_matches_df,
        n_components=reconciler.n_components,
        date_tolerance_days=reconciler.date_tolerance_days,
    )
    review_queue = build_review_queue(matches_df, args.review_threshold)
    reviewed_count = ingest_validated_matches(reconciler, review_queue, bank_clean, register_clean)
    if reviewed_count > 0:
        matches_df = reconciler.improve_with_review(bank_clean, register_clean, reviewed_count)

    final_report = create_final_report(matches_df, bank_clean, register_clean)
    ground_truth_df = None
    if args.ground_truth_file:
        ground_truth_df = load_ground_truth_csv(args.ground_truth_file)
        print(f"Using labeled ground truth from: {args.ground_truth_file}")
    elif args.use_simulated_ground_truth:
        ground_truth_df = build_ground_truth(bank_clean, register_clean)
        print("Using simulated ground truth derived from transaction ID suffixes.")
    else:
        print("No ground truth file provided; precision/recall/F1 will be skipped.")
    calculate_metrics(final_report, reconciler.date_tolerance_days, ground_truth_df)
    reconciler.print_summary()

    output_path = save_output(final_report, "output/reconciliation_results.csv")
    review_output = save_review_queue(review_queue, "output/reconciliation_review_queue.csv")
    learning_curve_output = save_learning_curve(learning_curve_df, "output/learning_curve.csv")
    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Review queue saved to: {review_output}")
    print(f"✓ Learning curve saved to: {learning_curve_output}")
    if not learning_curve_df.empty:
        print("Learning curve (validation accuracy by training size):")
        for row in learning_curve_df.itertuples(index=False):
            print(
                f"  {int(row.training_pairs):>3} pairs -> accuracy {row.validation_accuracy:.4f}"
            )
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
