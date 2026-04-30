from __future__ import annotations

import pandas as pd


def classify_molecular_behavior(row: pd.Series) -> dict:
    """Scientifically motivated qualitative classes from molecular descriptors."""
    logp = row.get("logp")
    tpsa = row.get("tpsa")
    aromatic = row.get("aromatic_rings")
    charge = row.get("formal_charge")

    polarity = "unknown"
    if pd.notna(tpsa):
        if tpsa >= 80:
            polarity = "highly polar"
        elif tpsa >= 40:
            polarity = "moderately polar"
        else:
            polarity = "low polarity"

    hydrophobicity = "unknown"
    if pd.notna(logp):
        if logp >= 3:
            hydrophobicity = "hydrophobic"
        elif logp <= 0:
            hydrophobicity = "hydrophilic"
        else:
            hydrophobicity = "moderately hydrophobic"

    aromaticity = "aromatic" if pd.notna(aromatic) and aromatic > 0 else "non-aromatic/unknown"
    charge_class = "neutral/unknown"
    if pd.notna(charge):
        charge_class = "cationic" if charge > 0 else "anionic" if charge < 0 else "neutral"

    return {
        "polarity_class": polarity,
        "hydrophobicity_class": hydrophobicity,
        "aromaticity_class": aromaticity,
        "charge_class": charge_class,
    }


def recommend_adsorbent(row: pd.Series) -> dict:
    """Rule-based adsorbent and condition recommendation grounded in adsorption chemistry."""
    cls = classify_molecular_behavior(row)
    mechanisms = []
    adsorbents = []
    conditions = []

    logp = row.get("logp")
    tpsa = row.get("tpsa")
    aromatic = row.get("aromatic_rings")
    solution_pH = row.get("Solution pH")
    surface_area = row.get("Surface area")
    pore_volume = row.get("Pore volume")

    if pd.notna(logp) and logp >= 2:
        mechanisms.append("hydrophobic partitioning")
        adsorbents.append("hydrophobic biochar or activated carbon")
    if pd.notna(aromatic) and aromatic > 0:
        mechanisms.append("π–π interaction with graphitic/aromatic carbon domains")
        adsorbents.append("high-temperature biochar / activated carbon with aromatic carbon domains")
    if pd.notna(tpsa) and tpsa >= 60:
        mechanisms.append("hydrogen bonding / polar interactions")
        adsorbents.append("oxygen/nitrogen-functionalized biochar, oxidized carbon, or polar mineral surface")
    if cls["charge_class"] == "cationic":
        mechanisms.append("electrostatic attraction to negatively charged surfaces")
        adsorbents.append("negatively charged biochar, oxidized carbon, clay, or cation-exchange-like adsorbent")
        conditions.append("operate at pH above adsorbent pHpzc when chemically stable")
    elif cls["charge_class"] == "anionic":
        mechanisms.append("electrostatic attraction to positively charged surfaces")
        adsorbents.append("positively modified biochar, metal oxide/hydroxide surface, or anion-exchange-like adsorbent")
        conditions.append("operate at pH below adsorbent pHpzc when chemically stable")

    if pd.notna(surface_area) and surface_area >= 500:
        mechanisms.append("high surface-area site availability")
    if pd.notna(pore_volume) and pore_volume >= 0.3:
        mechanisms.append("pore filling / intraparticle uptake")
    if pd.notna(solution_pH):
        conditions.append(f"current solution pH = {solution_pH:g}; verify contaminant ionization and adsorbent pHpzc")

    if not mechanisms:
        mechanisms.append("general surface adsorption; needs pKa/pHpzc and isotherm validation")
    if not adsorbents:
        adsorbents.append("screen high-surface-area biochar / activated carbon first")
    if not conditions:
        conditions.append("measure pHpzc, pKa, and adsorption isotherm before final process selection")

    return {
        **cls,
        "likely_mechanisms": "; ".join(dict.fromkeys(mechanisms)),
        "recommended_adsorbent": "; ".join(dict.fromkeys(adsorbents)),
        "condition_notes": "; ".join(dict.fromkeys(conditions)),
    }


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    recs = df.apply(recommend_adsorbent, axis=1).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), recs.reset_index(drop=True)], axis=1)
