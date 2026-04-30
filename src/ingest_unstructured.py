from __future__ import annotations

import re
import pandas as pd


CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
QUANTITY_PATTERN = re.compile(r"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|L|mL|tonnes?|t)\b", re.I)
PURITY_PATTERN = re.compile(r"(?:purity|assay)\s*[:=]?\s*(?P<purity>\d+(?:\.\d+)?)\s*%", re.I)


def parse_chemical_record(text: str) -> dict:
    """Extract structured fields from messy chemical inventory text.

    This is a small demonstration parser. In production this would be extended
    with dictionaries, OCR/PDF parsing, LLM-assisted extraction, validation
    against regulatory databases, and unit normalization.
    """
    cas = CAS_PATTERN.search(text)
    qty = QUANTITY_PATTERN.search(text)
    purity = PURITY_PATTERN.search(text)

    lower = text.lower()
    hazard = "unknown"
    for key in ["flammable", "corrosive", "toxic", "oxidizer", "irritant"]:
        if key in lower:
            hazard = key
            break

    storage = "ambient"
    if "cold" in lower or "refrigerated" in lower or "2-8" in lower:
        storage = "cold"
    elif "dry" in lower:
        storage = "dry"

    # Simple name heuristic: text before CAS or first comma.
    name_part = text
    if cas:
        name_part = text[: cas.start()]
    name = re.split(r"[,;|]", name_part)[0].strip(" -:")

    return {
        "raw_text": text,
        "chemical_name": name if name else None,
        "cas": cas.group(0) if cas else None,
        "quantity": float(qty.group("quantity")) if qty else None,
        "unit": qty.group("unit") if qty else None,
        "purity_percent": float(purity.group("purity")) if purity else None,
        "hazard_class": hazard,
        "storage": storage,
    }


def parse_records(texts: list[str]) -> pd.DataFrame:
    """Parse many unstructured chemical records into a dataframe."""
    return pd.DataFrame([parse_chemical_record(t) for t in texts])
