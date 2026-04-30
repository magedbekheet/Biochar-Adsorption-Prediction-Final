from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def chemberta_embeddings(
    smiles: list[str],
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 16,
) -> np.ndarray:
    """Generate mean-pooled ChemBERTa embeddings from SMILES strings.

    This function requires `transformers` and `torch`. It is intentionally kept
    separate from the Streamlit app so the deployed app can remain lightweight.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(pooled)

    return np.vstack(all_embeddings)


def load_or_create_chemberta_cache(
    smiles: pd.Series,
    cache_path: str | Path,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
) -> np.ndarray:
    """Load cached embeddings or create them if missing."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb = chemberta_embeddings(smiles.astype(str).tolist(), model_name=model_name)
    np.save(cache_path, emb)
    return emb
