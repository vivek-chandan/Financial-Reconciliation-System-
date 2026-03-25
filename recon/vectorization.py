from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_training_matrix(parallel_corpus: list[tuple], attribute_columns: list[str]) -> np.ndarray:
    """Convert aligned transaction pairs into a matrix for SVD training."""
    matrix = []
    for bank_attrs, register_attrs, lag in parallel_corpus:
        combined = {**bank_attrs, **register_attrs, f"lag_{lag}": 1}
        vector = np.array([combined.get(column, 0) for column in attribute_columns])
        matrix.append(vector)
    return np.array(matrix)


def train_svd_model(parallel_corpus: list[tuple], n_components: int) -> tuple[np.ndarray | None, list[str] | None, float]:
    """Train the SVD projection matrix from the aligned corpus."""
    if not parallel_corpus:
        return None, None, 0.0

    all_attributes = set()
    for bank_attrs, register_attrs, lag in parallel_corpus:
        all_attributes.update(bank_attrs.keys())
        all_attributes.update(register_attrs.keys())
        all_attributes.add(f"lag_{lag}")

    attribute_columns = list(all_attributes)
    matrix = build_training_matrix(parallel_corpus, attribute_columns)

    if len(matrix) <= 1:
        return None, attribute_columns, 0.0

    component_count = min(n_components, len(matrix) - 1)
    svd = TruncatedSVD(n_components=component_count, random_state=42)
    projection_matrix = svd.fit(matrix).components_.T
    explained_variance = float(svd.explained_variance_ratio_.sum())
    return projection_matrix, attribute_columns, explained_variance


def vectorize_attributes(attributes, attribute_columns: list[str]) -> np.ndarray:
    """Project a sparse attribute counter into a dense vector."""
    return np.array([attributes.get(column, 0) for column in attribute_columns])

