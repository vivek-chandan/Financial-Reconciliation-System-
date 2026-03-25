# Financial Reconciliation System

This project matches transactions between two financial data sources:

- bank statements
- check register records

It uses a hybrid reconciliation workflow:

1. unique-amount matching for high-confidence exact matches
2. SVD-based semantic matching for the remaining ambiguous transactions
3. a review queue for ML matches
4. a review-driven retraining step
5. evaluation and analysis outputs for reporting

The implementation is modular, command-line based, and designed to demonstrate the full cycle:

`match -> review -> improve`

## Project Structure

```text
reconcile.py                  # CLI entrypoint
recon/input_loader.py         # input file loading
recon/preprocessing.py        # text normalization and duplicate handling
recon/vectorization.py        # feature matrix + SVD training
recon/similarity.py           # cosine similarity scoring
recon/ml_module.py            # core reconciliation pipeline
recon/review_cycle.py         # review queue + feedback loop
recon/evaluation.py           # precision / recall / F1 support
recon/learning_curve.py       # training-size improvement analysis
recon/output_module.py        # report and analysis artifact generation
tests/                        # unit tests
data/                         # input CSV files
output/                       # generated outputs
```

## Requirements

- Python 3
- `pandas`
- `numpy`
- `scikit-learn`

## How to Run

From the project root:

```bash
python3 reconcile.py
```

This runs the full pipeline and:

- loads the bank and register files
- performs unique amount matching
- trains the SVD-based ML matcher
- matches remaining transactions
- builds a review queue
- retrains using validated review matches
- generates the final reconciliation report
- generates learning-curve and hardest-category analysis files

## CLI Options

Use the default dataset:

```bash
python3 reconcile.py
```

Use explicit input file names:

```bash
python3 reconcile.py --bank-file bank_statements.csv --register-file check_register.csv
```

Use a real labeled ground-truth file:

```bash
python3 reconcile.py --ground-truth-file ground_truth.csv
```

Use simulated ground truth derived from synthetic transaction ID suffixes:

```bash
python3 reconcile.py --use-simulated-ground-truth
```

Change the ML review threshold:

```bash
python3 reconcile.py --review-threshold 0.90
```

## Ground Truth Support

The project supports two evaluation modes:

1. labeled ground truth via `--ground-truth-file`
2. simulated ground truth via `--use-simulated-ground-truth`

If no ground-truth option is provided, the reconciliation still runs normally, but precision/recall/F1 are skipped.

Accepted labeled ground-truth formats:

- `bank_id, reg_id`
- `bank_transaction_id, register_transaction_id`

## How to Run Tests

Run the full unit test suite with:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- preprocessing and token normalization
- review queue generation
- issue flag generation
- ground-truth loading
- precision / recall / F1 calculation
- learning-curve output shape
- hardest transaction category analysis

## Output Files

Running the project generates these files in `output/`:

### `output/reconciliation_results.csv`

Final reconciliation report containing:

- bank and register transaction IDs
- confidence score
- match method
- date lag
- issue flags
- bank-side transaction details
- register-side transaction details

### `output/reconciliation_review_queue.csv`

Review-stage artifact containing:

- ML match candidates
- confidence score
- issue flags
- review status
- review round

### `output/learning_curve.csv`

Learning-curve analysis showing how held-out ML validation accuracy changes as more validated training pairs are available.

Columns include:

- `training_fraction`
- `training_pairs`
- `validation_pairs`
- `correct_predictions`
- `validation_accuracy`

### `output/hardest_transaction_types.csv`

Automatic category-difficulty analysis used to identify the hardest transaction types to reconcile.

Columns include:

- total transactions
- ML match count
- flagged transaction count
- low/medium-confidence counts
- unmatched count
- average confidence
- difficulty score

## Console Output

A typical run prints:

- number of unique amount matches
- number of ML matches
- unmatched transactions
- precision / recall / F1 when ground truth is supplied
- confidence distribution
- issue flag summary
- date lag summary
- hardest transaction categories
- learning-curve accuracy by training size

## Current Workflow Summary

The current pipeline works like this:

1. preprocess both datasets and remove duplicate transaction IDs
2. match transactions with amounts that are unique in both datasets
3. use those validated pairs as training data
4. train an SVD-based latent representation
5. score remaining candidates with cosine similarity plus date-aware filtering
6. produce a review queue for ML-based matches
7. feed validated review matches back into training
8. rerun the matcher and generate outputs

## Unit Testing

The project includes unit tests for critical logic. A successful test run looks like this:

```text
Ran 12 tests in ...s

OK
```

## Notes

- The implementation is paper-inspired rather than a strict reproduction of a full mutual-information translation model.
- The review/improve loop is implemented in the actual workflow, not just described in documentation.
- The project includes explicit issue flags for review-sensitive cases such as rounding differences, low-confidence matches, and unmatched rows.
