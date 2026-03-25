from __future__ import annotations

from pathlib import Path

import pandas as pd

# The input loader is designed to be flexible in locating CSV files,
#  allowing users to specify paths relative to either the project root or a data directory.
def resolve_input_path(preferred_path: str) -> Path:
    """Resolve CSV paths from either project root or the data directory."""
    path = Path(preferred_path)
    if path.exists():
        return path

    data_path = Path("data") / preferred_path
    if data_path.exists():
        return data_path

    raise FileNotFoundError(f"Could not find input file: {preferred_path}")

# The main loading function reads both bank and register CSVs,
#  ensuring that date columns are parsed correctly.
def load_transaction_data(bank_file: str, register_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bank and register CSV files with parsed dates."""
    bank_path = resolve_input_path(bank_file)
    register_path = resolve_input_path(register_file)
    bank_df = pd.read_csv(bank_path, parse_dates=["date"])
    register_df = pd.read_csv(register_path, parse_dates=["date"])
    return bank_df, register_df

