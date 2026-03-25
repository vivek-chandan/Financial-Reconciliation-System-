# Analysis & Documentation

## Performance Analysis

The reconciliation pipeline processed `308` bank transactions and matched `307` of them. Unique amount matching resolved `286` pairs directly, while the SVD-based ML stage resolved `21` additional pairs.

### Evaluation Metrics

- Precision: `1.0000`
- Recall: `0.9968`
- F1 Score: `0.9984`

The system uses standard pairwise reconciliation metrics when ground truth is available. If a labeled ground-truth file is not supplied, those metrics are intentionally omitted rather than inferred silently.

### Hardest Transaction Types

The automatically generated category-difficulty analysis identified the following categories as the hardest to reconcile:

- `Utility`: difficulty `0.1545`, flags `3/14`, avg confidence `0.9555`
- `Subscription`: difficulty `0.0877`, flags `2/16`, avg confidence `0.9982`
- `Check Payment`: difficulty `0.0737`, flags `2/20`, avg confidence `0.9630`
- `Grocery`: difficulty `0.0628`, flags `5/61`, avg confidence `0.9867`
- `Transfer`: difficulty `0.0290`, flags `1/25`, avg confidence `0.9900`

These categories are difficult because they rely more heavily on ML matching, accumulate more issue flags, or contain weaker textual overlap between bank and register descriptions.

### Low-Confidence Match Examples

- `B0074 -> R0074` | confidence `0.5405` | category `Check Payment` | flags `ROUNDING_DIFFERENCE|LOW_CONFIDENCE`
- `B0242 -> R0242` | confidence `0.5954` | category `Online Purchase` | flags `ROUNDING_DIFFERENCE|LOW_CONFIDENCE`
- `B0118 -> R0118` | confidence `0.6691` | category `Utility` | flags `ROUNDING_DIFFERENCE|LOW_CONFIDENCE`
- `B0153 -> R0153` | confidence `0.6987` | category `Restaurant` | flags `ROUNDING_DIFFERENCE|LOW_CONFIDENCE`
- `B0286 -> R0286` | confidence `0.7200` | category `Check Payment` | flags `ROUNDING_DIFFERENCE|MEDIUM_CONFIDENCE`

### Unmatched Examples

- `B0127` | `EXXON #2678` | amount `60.73`

### Improvement with More Training Data

To make the review/improve loop measurable, the project generates a held-out learning-curve benchmark for the ML matcher. The current run produced:

- `22` training pairs -> validation accuracy `0.4912`
- `45` training pairs -> validation accuracy `0.6140`
- `91` training pairs -> validation accuracy `0.6316`
- `137` training pairs -> validation accuracy `0.6842`
- `183` training pairs -> validation accuracy `0.6842`
- `229` training pairs -> validation accuracy `0.7018`

This shows that the semantic matcher improves as more validated transaction pairs are available for training, even when the final end-to-end reconciliation F1 is already close to saturation on the assignment dataset.

## Design Decisions

The implementation uses a hybrid reconciliation strategy. Deterministic unique-amount matching is used first because it provides high-confidence seed pairs. Those seed pairs become training examples for an unsupervised semantic model built with vectorized transaction attributes, Truncated SVD, and cosine similarity.

This design was chosen because it is practical, explainable, and robust for financial reconciliation. It captures the spirit of the paper's translation-based formulation while remaining simple enough to implement and debug in a command-line project.

### Departures from the Paper

- The implementation is SVD-inspired rather than a full reproduction of a mutual-information term-alignment translation model.
- It uses practical preprocessing and business-rule filters such as amount tolerance and date windows before ML scoring.
- It adds explicit issue flags and review artifacts to make the reconciliation process auditable.

These departures were intentional trade-offs in favor of reliability, transparency, and development speed.

## Limitations & Future Improvements

The current implementation still has several limitations:

- Some ML matches remain low-confidence even after preprocessing improvements.
- The semantic model is based on normalized token attributes rather than richer learned text embeddings.
- The final reconciliation metrics can saturate on this dataset, so learning-curve evaluation is needed to show improvement from additional validated training pairs.
- Edge cases involving generic descriptions, repeated amounts, or subtle bill-payment differences remain challenging.

With more time, the next improvements would be:

- add a configurable rejection threshold so weak ML matches remain unmatched instead of being forced
- support richer descriptor alignment or embedding-based semantic matching
- expand merchant and biller normalization dictionaries
- add an interactive review UI rather than threshold-only simulated review
- support multiple review rounds and compare performance round-by-round

## Summary

The project fulfills the main assignment workflow: `match -> review -> improve`. It produces reconciliation outputs, review artifacts, evaluation metrics when ground truth is available, issue flags for risky matches, a learning curve for model improvement evidence, and automatic category-level difficulty analysis.
