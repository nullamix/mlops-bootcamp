import hashlib
import pandas as pd

DIRECT_PII_COLUMNS: list = ["host_name", "host_id"]


def pseudonymize_value(value, salt: str = "qbc12") -> str:
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def handle_pii(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["host_key"] = clean["host_id"].apply(pseudonymize_value)
    clean = clean.drop(columns=[col for col in DIRECT_PII_COLUMNS if col in clean.columns])
    return clean
