from __future__ import annotations

import re
from collections import Counter

import pandas as pd
# The preprocessing module focuses on cleaning and normalizing transaction descriptions to facilitate better matching.
TOKEN_PATTERN = re.compile(r"[a-z]+")
STOP_WORDS = {
    "ach",
    "by",
    "co",
    "com",
    "dep",
    "direct",
    "llc",
    "online",
    "payment",
    "pmt",
    "purchase",
    "the",
    "to",
    "transfer",
}
# The TOKEN_NORMALIZATION dictionary maps common lexical variants to a standard form, which helps align differently worded descriptions that refer to the same category (e.g., "atm" and "cash" both map to "cash"). This normalization step is crucial for improving the recall of matches across sources with varying description styles.
TOKEN_NORMALIZATION = {
    "atm": "cash",
    "auto": "insurance",
    "bistro": "restaurant",
    "bp": "gas",
    "cafe": "restaurant",
    "charge": "fee",
    "chevron": "gas",
    "diner": "restaurant",
    "elec": "electric",
    "electricity": "electric",
    "exxon": "gas",
    "fill": "gas",
    "food": "grocery",
    "fuel": "gas",
    "gas": "gas",
    "groceries": "grocery",
    "gym": "subscription",
    "health": "insurance",
    "ins": "insurance",
    "joes": "grocery",
    "kroger": "grocery",
    "membership": "subscription",
    "monthly": "subscription",
    "netflix": "subscription",
    "restaurant": "restaurant",
    "safeway": "grocery",
    "salary": "payroll",
    "shell": "gas",
    "shopping": "grocery",
    "station": "gas",
    "store": "grocery",
    "trader": "grocery",
    "utilities": "utility",
    "utility": "utility",
    "water": "utility",
    "whole": "grocery",
}

# The _normalize_token function applies the normalization rules to each token extracted from the description.
#  It also filters out stop words and very short tokens, which are unlikely to be helpful for matching and
#  can add noise to the process. By standardizing common variants and removing weak tokens,
#  this function helps create a more consistent set of attributes for each transaction, 
# improving the chances of successful matches across differently worded descriptions.

def _normalize_token(token: str) -> str | None:
    """Normalize lexical variants and drop weak/noisy tokens."""
    if token in STOP_WORDS or len(token) <= 1:
        return None
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    token = TOKEN_NORMALIZATION.get(token, token)
    if token in STOP_WORDS or len(token) <= 1:
        return None
    return token


def tokenize_description(description: str) -> list[str]:
    """
    Convert free-form bank/register descriptions into normalized lexical tokens.

    The matcher relies on these tokens as a lightweight semantic bridge between
    differently worded descriptions such as `BP GAS #5199` and `Gas station`.
    Numeric references and weak boilerplate words are dropped because they add
    noise without helping cross-source alignment.
    """
    description_text = str(description).lower()
    tokens = []
    for raw_token in TOKEN_PATTERN.findall(description_text):
        normalized = _normalize_token(raw_token)
        if normalized is not None:
            tokens.append(normalized)

    # Utility bill descriptions are often terse on the bank side. Add a shared
    # utility token so `ONLINE PMT GAS CO` and `Electric bill` stay comparable.
    if "bill" in description_text or "water" in description_text or "elec" in description_text or "electric" in description_text:
        tokens.append("utility")
    if "gas co" in description_text or "gas company" in description_text:
        tokens.append("utility")
    return tokens


def get_attributes(row: pd.Series) -> Counter:
    """Extract descriptor tokens and normalized transaction type."""
    desc_tokens = tokenize_description(row["description"])
    tx_type = str(row["type"]).lower()
    normalized_type = "dr" if "debit" in tx_type or "dr" in tx_type else "cr"
    return Counter(desc_tokens + [normalized_type])


def preprocess_data(df: pd.DataFrame, df_name: str, stats: dict) -> pd.DataFrame:
    """Remove duplicate transaction IDs and track duplicate statistics."""
    original_count = len(df)
    duplicates = df[df.duplicated(subset=["transaction_id"], keep="first")]

    if len(duplicates) > 0:
        print(f"\n  WARNING: Found {len(duplicates)} duplicate transaction_ids in {df_name}:")
        for transaction_id in duplicates["transaction_id"].unique()[:10]:
            count = len(df[df["transaction_id"] == transaction_id])
            print(f"   - {transaction_id}: appears {count} times")
        if len(duplicates) > 10:
            print(f"   ... and {len(duplicates) - 10} more")
        stats["duplicates_removed"] += len(duplicates)

    cleaned = df.drop_duplicates(subset=["transaction_id"], keep="first").copy()
    print(f"✓ {df_name}: {len(cleaned)} unique transactions (removed {original_count - len(cleaned)} duplicates)")
    return cleaned
