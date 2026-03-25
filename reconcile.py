from __future__ import annotations

import argparse

from recon.input_loader import load_transaction_data
from recon.ml_module import FinancialReconciler
from recon.output_module import (
    build_default_ground_truth,
    calculate_metrics,
    create_final_report,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading data files...")
    bank_df, register_df = load_transaction_data(args.bank_file, args.register_file)

    reconciler = FinancialReconciler(n_components=15, date_tolerance_days=5)
    matches_df, bank_clean, register_clean = reconciler.reconcile_all(bank_df, register_df)
    review_queue = build_review_queue(matches_df, args.review_threshold)
    reviewed_count = ingest_validated_matches(reconciler, review_queue, bank_clean, register_clean)
    if reviewed_count > 0:
        matches_df = reconciler.improve_with_review(bank_clean, register_clean, reviewed_count)

    final_report = create_final_report(matches_df, bank_clean, register_clean)
    ground_truth_df = build_default_ground_truth(bank_clean, register_clean)
    calculate_metrics(final_report, reconciler.date_tolerance_days, ground_truth_df)
    reconciler.print_summary()

    output_path = save_output(final_report, "output/reconciliation_results.csv")
    review_output = save_review_queue(review_queue, "output/reconciliation_review_queue.csv")
    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Review queue saved to: {review_output}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
