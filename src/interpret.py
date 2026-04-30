from __future__ import annotations

import pandas as pd


def get_feature_importance(model_pipeline, max_features: int = 25) -> pd.DataFrame:
    """Return model feature importances when available."""
    pre = model_pipeline.named_steps["preprocess"]
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )
