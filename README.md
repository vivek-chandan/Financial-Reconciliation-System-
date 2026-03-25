# Financial-Reconciliation-System-
 It is a system that automatically matches transactions between two independent financial data sources (e.g., bank statements and internal check registers) using unsupervised machine learning techniques

### How to Run the Project

From the project root, run:

```bash
python3 reconcile.py
```

This will:
- load the bank and check register CSV files
- perform unique-amount matching
- train the SVD-based ML matcher
- generate a review queue
- retrain using validated review matches
- save the final reconciliation results


### How to Run Tests

Run the unit test suite with:

```bash
python3 -m unittest discover -s tests -v
```

This executes tests for:
- preprocessing/token normalization
- review queue generation
- evaluation metrics and ground-truth mapping

### Expected Outputs

After running the project, the following files are generated in the `output/` folder:

- `output/reconciliation_results.csv`
  Final reconciliation report containing:
  - matched bank and register transaction IDs
  - confidence scores
  - match method
  - date lag
  - transaction details from both sources

- `output/reconciliation_review_queue.csv`
  Review queue for ML-based matches, including:
  - confidence score
  - review status (`validated` or `needs_review`)
  - review round

Typical console output includes:
- number of unique-amount matches
- number of ML matches
- unmatched transactions
- precision, recall, and F1 score
- date lag summary
- final reconciliation summary
