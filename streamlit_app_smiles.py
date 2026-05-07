from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
except Exception:
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


# ============================================================
# App configuration
# ============================================================
st.set_page_config(
    page_title="Adsorption Capacity Predictor",
    layout="wide",
)

st.title("Adsorption Capacity Predictor for Organic Contaminants")
st.write(
    "A data-driven screening tool for estimating adsorption capacity from "
    "adsorbent properties, process conditions, wastewater matrix, and pollutant descriptors."
)


# ============================================================
# Paths
# Hybrid model predicts Langmuir qmax and K.
# ============================================================
APP_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    APP_DIR / "models" / "hybrid_qmax_k_model.joblib",
    Path("models/hybrid_qmax_k_model.joblib"),
]

FEATURE_PATHS = [
    APP_DIR / "models" / "hybrid_feature_columns.joblib",
    APP_DIR / "models" / "feature_columns.joblib",
    Path("models/hybrid_feature_columns.joblib"),
    Path("models/feature_columns.joblib"),
]

RANGE_MIN_PATHS = [
    APP_DIR / "models" / "X_min.joblib",
    Path("models/X_min.joblib"),
]

RANGE_MAX_PATHS = [
    APP_DIR / "models" / "X_max.joblib",
    Path("models/X_max.joblib"),
]

WASTEWATER_TYPES = ["Synthetic", "Lake water", "Secondary effluent", "Ground water"]
ADSORPTION_TYPES = ["Single", "Competative"]  # Keep dataset spelling.


POLLUTANT_LIBRARY: Dict[str, Dict[str, float | str]] = {
    # Pharmaceuticals already used in the original app
    "IBU / IBF - Ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "mol_weight": 206.28,
        "logp": 3.50,
        "tpsa": 37.30,
        "hbd": 1.0,
        "hba": 2.0,
        "num_rings": 1.0,
    },
    "CBZ - Carbamazepine": {
        "smiles": "NC(=O)N1c2ccccc2C=Cc3ccccc31",
        "mol_weight": 236.27,
        "logp": 2.45,
        "tpsa": 46.30,
        "hbd": 1.0,
        "hba": 2.0,
        "num_rings": 3.0,
    },
    "DCF - Diclofenac": {
        "smiles": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",
        "mol_weight": 296.15,
        "logp": 4.50,
        "tpsa": 49.30,
        "hbd": 2.0,
        "hba": 3.0,
        "num_rings": 2.0,
    },
    "EE2 - 17α-ethinylestradiol": {
        "smiles": "C#CC1(O)CCC2C3CCC4=CC(=O)CCC4C3CCC21C",
        "mol_weight": 296.41,
        "logp": 4.15,
        "tpsa": 40.50,
        "hbd": 2.0,
        "hba": 2.0,
        "num_rings": 4.0,
    },
    "NPX / NXP - Naproxen": {
        "smiles": "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",
        "mol_weight": 230.26,
        "logp": 3.18,
        "tpsa": 46.50,
        "hbd": 1.0,
        "hba": 3.0,
        "num_rings": 2.0,
    },
    "ATE - Atenolol": {
        "smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "mol_weight": 266.34,
        "logp": 0.16,
        "tpsa": 84.60,
        "hbd": 4.0,
        "hba": 4.0,
        "num_rings": 1.0,
    },

    # Additional literature pollutants / dyes for external validation
    "MB - Methylene Blue": {
        "smiles": "CN(C)c1ccc2nc3ccc(N(C)C)cc3[s+]c2c1.[Cl-]",
        "mol_weight": 319.85,
        "logp": 3.14,
        "tpsa": 43.90,
        "hbd": 0.0,
        "hba": 3.0,
        "num_rings": 3.0,
    },
    "MO - Methyl Orange": {
        "smiles": "CN(C)c1ccc(N=Nc2ccc(S(=O)(=O)[O-])cc2)cc1.[Na+]",
        "mol_weight": 327.33,
        "logp": 3.46,
        "tpsa": 89.40,
        "hbd": 0.0,
        "hba": 6.0,
        "num_rings": 2.0,
    },
    "RhB - Rhodamine B": {
        "smiles": "CCN(CC)c1ccc2c(c1)C(=O)OC3=C2C=CC(=[N+](CC)CC)C=C3.[Cl-]",
        "mol_weight": 479.02,
        "logp": 3.20,
        "tpsa": 52.90,
        "hbd": 0.0,
        "hba": 5.0,
        "num_rings": 4.0,
    },
    "CR - Congo Red": {
        "smiles": "Nc1ccccc1N=Nc2ccc(cc2)C(=C3C=CC(=CC3)N=Nc4ccccc4N)S(=O)(=O)[O-].[Na+]",
        "mol_weight": 696.66,
        "logp": 5.10,
        "tpsa": 177.00,
        "hbd": 4.0,
        "hba": 10.0,
        "num_rings": 6.0,
    },
    "Acetaminophen / Paracetamol": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "mol_weight": 151.16,
        "logp": 0.46,
        "tpsa": 49.30,
        "hbd": 2.0,
        "hba": 2.0,
        "num_rings": 1.0,
    },
    "BPA - Bisphenol A": {
        "smiles": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
        "mol_weight": 228.29,
        "logp": 3.64,
        "tpsa": 40.46,
        "hbd": 2.0,
        "hba": 2.0,
        "num_rings": 2.0,
    },
    "SMX - Sulfamethoxazole": {
        "smiles": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
        "mol_weight": 253.28,
        "logp": 0.89,
        "tpsa": 98.22,
        "hbd": 2.0,
        "hba": 6.0,
        "num_rings": 2.0,
    },
    "CIP - Ciprofloxacin": {
        "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "mol_weight": 331.35,
        "logp": 0.28,
        "tpsa": 74.57,
        "hbd": 2.0,
        "hba": 6.0,
        "num_rings": 4.0,
    },
    "Manual molecular descriptors": {
        "smiles": "",
        "mol_weight": 206.28,
        "logp": 3.50,
        "tpsa": 37.30,
        "hbd": 1.0,
        "hba": 2.0,
        "num_rings": 1.0,
    },
}



# ============================================================
# Helpers
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, List[str], Path, Path, pd.Series, pd.Series]:
    model_path = next((p for p in MODEL_PATHS if p.exists()), None)
    feature_path = next((p for p in FEATURE_PATHS if p.exists()), None)
    min_path = next((p for p in RANGE_MIN_PATHS if p.exists()), None)
    max_path = next((p for p in RANGE_MAX_PATHS if p.exists()), None)

    if model_path is None:
        raise FileNotFoundError(
            "Could not find hybrid_qmax_k_model.joblib in ./models/ or the project root."
        )
    if feature_path is None:
        raise FileNotFoundError(
            "Could not find hybrid_feature_columns.joblib or feature_columns.joblib in ./models/ or the project root."
        )
    if min_path is None or max_path is None:
        raise FileNotFoundError(
            "Could not find X_min.joblib and X_max.joblib in ./models/ or the project root. "
            "These files are needed to restrict inputs to the training range."
        )

    model_obj = joblib.load(model_path)
    cols = [str(c).strip() for c in joblib.load(feature_path)]
    x_min = joblib.load(min_path)
    x_max = joblib.load(max_path)

    return model_obj, cols, model_path, feature_path, x_min, x_max


def rng(col: str, fallback_min: float, fallback_max: float) -> Tuple[float, float]:
    if col in X_min.index and col in X_max.index:
        return float(X_min[col]), float(X_max[col])
    return float(fallback_min), float(fallback_max)


def mid(col: str, fallback_min: float, fallback_max: float) -> float:
    a, b = rng(col, fallback_min, fallback_max)
    return (a + b) / 2.0


def clipped(value: float, col: str, fallback_min: float, fallback_max: float) -> float:
    a, b = rng(col, fallback_min, fallback_max)
    return min(max(float(value), a), b)


def descriptors_from_smiles(smiles: str) -> Dict[str, float]:
    """Calculate app molecular descriptors from a SMILES string using RDKit."""
    if Chem is None:
        raise RuntimeError(
            "RDKit is not installed, so descriptors cannot be calculated from SMILES. "
            "Install RDKit or use preset/manual descriptors."
        )

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError("Invalid SMILES string. Please check the chemical structure input.")

    return {
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
    }


def descriptor_source_note(source: str, smiles: str | None = None) -> None:
    if source == "preset":
        st.sidebar.caption("Descriptors are calculated from the preset SMILES when RDKit is available; stored fallback values are used otherwise.")
    elif source == "smiles":
        st.sidebar.caption("Descriptors are calculated directly from the SMILES string using RDKit.")
    else:
        st.sidebar.caption("Manual mode lets you test pollutants not available as presets or SMILES.")


def add_wastewater_dummies(input_data: Dict[str, float], wastewater_type: str) -> Dict[str, float]:
    for wt in WASTEWATER_TYPES:
        input_data[f"Wastewater type_{wt}"] = 1.0 if wastewater_type == wt else 0.0
    return input_data


def add_engineered_features(input_data: Dict[str, float]) -> Dict[str, float]:
    """Add the same physics-inspired features used in the notebook training step."""
    eps = 1e-9

    initial_concentration = float(input_data["Initial concentration"])
    adsorbent_dosage = float(input_data["Adsorbent dosage"])
    surface_area = float(input_data["Surface area"])
    adsorption_time = float(input_data["Adsorption time"])

    input_data["C_over_dosage"] = initial_concentration / (adsorbent_dosage + eps)
    input_data["log_initial_concentration"] = np.log1p(max(0.0, initial_concentration))
    input_data["log_surface_area"] = np.log1p(max(0.0, surface_area))
    input_data["log_adsorption_time"] = np.log1p(max(0.0, adsorption_time))
    input_data["time_over_dosage"] = adsorption_time / (adsorbent_dosage + eps)

    return input_data


def build_input_frame(input_data: Dict[str, float], feature_columns: List[str]) -> pd.DataFrame:
    """Build a one-row feature matrix with exactly the training feature schema."""
    x_new = pd.DataFrame([input_data])
    x_new.columns = x_new.columns.astype(str).str.strip()

    missing = [c for c in feature_columns if c not in x_new.columns]
    if missing:
        raise ValueError("The app is missing model input columns: " + ", ".join(missing))

    x_new = x_new[feature_columns]
    x_new = x_new.apply(pd.to_numeric, errors="coerce")

    if x_new.isna().any().any():
        bad_cols = x_new.columns[x_new.isna().any()].tolist()
        raise ValueError("Non-numeric or missing values found in: " + ", ".join(bad_cols))

    return x_new


def hybrid_langmuir_predict(param_model: object, x_new: pd.DataFrame) -> Tuple[float, float, float]:
    """Predict capacity using ML-estimated qmax and K in the Langmuir equation."""
    param_pred = param_model.predict(x_new)

    qmax_pred = float(param_pred[0, 0])
    k_pred = float(np.expm1(param_pred[0, 1]))

    qmax_pred = max(0.0, qmax_pred)
    k_pred = min(max(k_pred, 1e-6), 10.0)

    concentration = float(x_new["Initial concentration"].iloc[0])
    pred = (qmax_pred * k_pred * concentration) / (1.0 + k_pred * concentration)
    pred = max(0.0, float(pred))

    return pred, qmax_pred, k_pred


def validate_inputs(
    adsorption_time: float,
    initial_concentration: float,
    adsorbent_dosage: float,
    carbon: float,
) -> List[str]:
    errors: List[str] = []
    if adsorption_time <= 0:
        errors.append("Adsorption time must be > 0. At t = 0, adsorption capacity should be approximately zero.")
    if initial_concentration <= 0:
        errors.append("Initial concentration must be > 0.")
    if adsorbent_dosage <= 0:
        errors.append("Adsorbent dosage must be > 0.")
    if carbon <= 0:
        errors.append("C (%) must be > 0 to calculate H/C, O/C, N/C, and (O+N)/C ratios.")
    return errors


def recommendation_text(input_data: Dict[str, float], pred: float) -> List[str]:
    recs: List[str] = []

    c_over_dosage = input_data.get("C_over_dosage", None)

    if pred >= 100:
        recs.append("High predicted adsorption capacity under the selected conditions.")
    elif pred >= 30:
        recs.append("Moderate predicted adsorption capacity under the selected conditions.")
    else:
        recs.append("Low predicted adsorption capacity under the selected conditions.")

    if c_over_dosage is not None:
        recs.append(
            "The concentration-to-dosage ratio is an important driver of capacity; "
            "higher pollutant loading per adsorbent mass can increase apparent mg/g capacity."
        )

    if input_data["Adsorbent dosage"] > 0:
        recs.append(
            "Increasing adsorbent dosage can improve removal efficiency, but may reduce capacity expressed as mg/g."
        )

    if input_data["Surface area"] >= 400:
        recs.append(
            "High surface area may support adsorption-site availability, although surface chemistry and pore accessibility also matter."
        )
    else:
        recs.append(
            "Lower surface area may limit site availability; capacity may improve with a more developed porous structure."
        )

    if input_data["N"] >= 2 or input_data["N/C"] >= 0.04:
        recs.append(
            "N-containing surface groups may contribute to polar interactions, hydrogen bonding, or electron donor–acceptor interactions."
        )

    if input_data["O/C"] >= 0.05 or input_data["(O+N)/C"] >= 0.08:
        recs.append(
            "Higher heteroatom content suggests more polar surface functionality, which may affect adsorption depending on pollutant polarity and pH."
        )

    if input_data["logp"] >= 2:
        recs.append(
            "The pollutant is relatively hydrophobic, so hydrophobic interactions with carbon-rich adsorbents may contribute to adsorption."
        )
    elif input_data["tpsa"] >= 70:
        recs.append(
            "The pollutant is relatively polar; electrostatic effects, hydrogen bonding, and surface functionality may be important."
        )

    if input_data["num_rings"] >= 1:
        recs.append(
            "Aromatic rings may support π–π interactions with graphitic or aromatic carbon domains."
        )

    if input_data["competitive"] == 1:
        recs.append(
            "Competitive adsorption can reduce effective capacity because multiple species may compete for available adsorption sites."
        )

    recs.append(
        "These explanations are qualitative chemistry-based interpretations; SHAP analysis should be used for model-specific feature attribution."
    )

    return recs


# ============================================================
# Load artifacts
# ============================================================
try:
    model, feature_columns, model_path, feature_path, X_min, X_max = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.info(
    "Hybrid model loaded: machine learning estimates Langmuir qmax and K, "
    "then the Langmuir equation computes adsorption capacity."
)


# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.header("Pollutant")
descriptor_mode = st.sidebar.radio(
    "How do you want to define the pollutant?",
    ["Preset pollutant", "Enter SMILES", "Manual descriptors"],
    horizontal=False,
)

if descriptor_mode == "Preset pollutant":
    preset_names = [name for name in POLLUTANT_LIBRARY if name != "Manual molecular descriptors"]
    pollutant = st.sidebar.selectbox("Pollutant / dye preset", preset_names)
    preset = POLLUTANT_LIBRARY[pollutant]
    preset_smiles = str(preset.get("smiles", ""))

    if preset_smiles and Chem is not None:
        try:
            desc = POLLUTANT_LIBRARY[pollutant].copy()
        except Exception:
            desc = {k: float(preset[k]) for k in ["mol_weight", "logp", "tpsa", "hbd", "hba", "num_rings"]}
            st.sidebar.warning("Could not calculate descriptors from preset SMILES; using stored fallback values.")
    else:
        desc = {k: float(preset[k]) for k in ["mol_weight", "logp", "tpsa", "hbd", "hba", "num_rings"]}

    descriptor_source_note("preset", preset_smiles)
    if preset_smiles:
        with st.sidebar.expander("Show preset SMILES"):
            st.code(preset_smiles, language="text")
    manual = False

elif descriptor_mode == "Enter SMILES":
    pollutant = "Custom SMILES pollutant"
    custom_smiles = st.sidebar.text_input(
        "SMILES",
        value="CC(=O)Nc1ccc(O)cc1",
        help="Enter a valid SMILES string. RDKit will calculate molecular descriptors automatically.",
    )
    descriptor_source_note("smiles", custom_smiles)

    try:
        desc = descriptors_from_smiles(custom_smiles)
        st.sidebar.success(
            "Descriptors calculated from SMILES using RDKit."
        )

        st.sidebar.caption(
            "Calculated descriptor values may differ slightly from "
            "stored or literature values depending on molecular form "
            "(e.g., salt/protonation state) and descriptor method."
        )
    except Exception as exc:
        st.sidebar.error(str(exc))
        desc = {k: float(POLLUTANT_LIBRARY["Manual molecular descriptors"][k]) for k in ["mol_weight", "logp", "tpsa", "hbd", "hba", "num_rings"]}

    manual = False

else:
    pollutant = "Manual molecular descriptors"
    desc = {k: float(POLLUTANT_LIBRARY[pollutant][k]) for k in ["mol_weight", "logp", "tpsa", "hbd", "hba", "num_rings"]}
    descriptor_source_note("manual")
    manual = True

st.sidebar.header("Input conditions")

initial_concentration = st.sidebar.number_input(
    "Initial concentration (mg/L)",
    min_value=rng("Initial concentration", 0.5, 100.0)[0],
    max_value=rng("Initial concentration", 0.5, 100.0)[1],
    value=clipped(50.0, "Initial concentration", 0.5, 100.0),
    step=5.0,
)

solution_pH = st.sidebar.number_input(
    "Solution pH",
    min_value=rng("Solution pH", 1.0, 14.0)[0],
    max_value=rng("Solution pH", 1.0, 14.0)[1],
    value=mid("Solution pH", 1.0, 14.0),
    step=0.1,
)

adsorption_time = st.sidebar.number_input(
    "Adsorption time (min)",
    min_value=rng("Adsorption time", 0.1, 500.0)[0],
    max_value=rng("Adsorption time", 0.1, 500.0)[1],
    value=clipped(120.0, "Adsorption time", 0.1, 500.0),
    step=10.0,
)

adsorbent_dosage = st.sidebar.number_input(
    "Adsorbent dosage (g/L)",
    min_value=rng("Adsorbent dosage", 0.02, 0.16)[0],
    max_value=rng("Adsorbent dosage", 0.02, 0.16)[1],
    value=clipped(0.08, "Adsorbent dosage", 0.02, 0.16),
    step=0.01,
)

temperature = st.sidebar.number_input(
    "Adsorption temperature (°C)",
    min_value=rng("Adsorption temperature", 20.0, 40.0)[0],
    max_value=rng("Adsorption temperature", 20.0, 40.0)[1],
    value=clipped(25.0, "Adsorption temperature", 20.0, 40.0),
    step=1.0,
)

ion_concentration = st.sidebar.number_input(
    "Ion concentration",
    min_value=rng("Ion concentration", 0.0, 1.0)[0],
    max_value=rng("Ion concentration", 0.0, 1.0)[1],
    value=mid("Ion concentration", 0.0, 1.0),
    step=0.1,
)

humic_acid = st.sidebar.number_input(
    "Humic acid",
    min_value=rng("Humic acid", 0.0, 1.0)[0],
    max_value=rng("Humic acid", 0.0, 1.0)[1],
    value=mid("Humic acid", 0.0, 1.0),
    step=0.1,
)

st.sidebar.header("Adsorbent properties")

surface_area = st.sidebar.number_input(
    "Surface area (m²/g)",
    min_value=rng("Surface area", 0.0, 1000.0)[0],
    max_value=rng("Surface area", 0.0, 1000.0)[1],
    value=clipped(400.0, "Surface area", 0.0, 1000.0),
    step=50.0,
)

pore_volume = st.sidebar.number_input(
    "Pore volume (cm³/g)",
    min_value=rng("Pore volume", 0.0, 2.0)[0],
    max_value=rng("Pore volume", 0.0, 2.0)[1],
    value=clipped(0.5, "Pore volume", 0.0, 2.0),
    step=0.05,
)

average_pore_size = st.sidebar.number_input(
    "Average pore size (nm)",
    min_value=rng("Average pore size", 0.0, 20.0)[0],
    max_value=rng("Average pore size", 0.0, 20.0)[1],
    value=clipped(5.0, "Average pore size", 0.0, 20.0),
    step=0.5,
)

carbon = st.sidebar.number_input(
    "C (%)",
    min_value=rng("C", 0.0, 100.0)[0],
    max_value=rng("C", 0.0, 100.0)[1],
    value=clipped(79.0, "C", 0.0, 100.0),
    step=1.0,
)
hydrogen = st.sidebar.number_input("H (%)", min_value=rng("H", 0.3, 2.25)[0], max_value=rng("H", 0.3, 2.25)[1], value=mid("H", 0.3, 2.25), step=0.1)
oxygen = st.sidebar.number_input("O (%)", min_value=rng("O", 1.8, 5.93)[0], max_value=rng("O", 1.8, 5.93)[1], value=mid("O", 1.8, 5.93), step=0.1)
nitrogen = st.sidebar.number_input("N (%)", min_value=rng("N", 0.0, 5.0)[0], max_value=rng("N", 0.0, 5.0)[1], value=mid("N", 0.0, 5.0), step=0.1)
ash = st.sidebar.number_input("Ash (%)", min_value=rng("Ash", 5.6, 24.9)[0], max_value=rng("Ash", 5.6, 24.9)[1], value=mid("Ash", 5.6, 24.9), step=0.5)

st.sidebar.header("Pollutant molecular descriptors")
mol_weight = st.sidebar.number_input(
    "Molecular weight (g/mol)",
    min_value=0.0,
    max_value=1000.0,
    value=clipped(desc["mol_weight"], "mol_weight", 0.0, 1000.0),
    step=1.0,
    disabled=not manual,
)
if mol_weight > rng("mol_weight", 0.0, 1000.0)[1]:
    st.warning(
        f"Molecular weight is outside the training range "
        f"[{rng('mol_weight', 0.0, 1000.0)[0]:.1f}, {rng('mol_weight', 0.0, 1000.0)[1]:.1f}]. "
        "Prediction may be less reliable."
    )

logp = st.sidebar.number_input(
    "logP",
    min_value=rng("logp", -5.0, 5.0)[0],
    max_value=rng("logp", -5.0, 5.0)[1],
    value=clipped(desc["logp"], "logp", -5.0, 5.0),
    step=0.1,
    disabled=not manual,
)

tpsa = st.sidebar.number_input(
    "TPSA",
    min_value=rng("tpsa", 0.0, 200.0)[0],
    max_value=rng("tpsa", 0.0, 200.0)[1],
    value=clipped(desc["tpsa"], "tpsa", 0.0, 200.0),
    step=1.0,
    disabled=not manual,
)

hbd = st.sidebar.number_input(
    "H-bond donors",
    min_value=rng("hbd", 0.0, 10.0)[0],
    max_value=rng("hbd", 0.0, 10.0)[1],
    value=clipped(desc["hbd"], "hbd", 0.0, 10.0),
    step=1.0,
    disabled=not manual,
)

hba = st.sidebar.number_input(
    "H-bond acceptors",
    min_value=rng("hba", 0.0, 10.0)[0],
    max_value=rng("hba", 0.0, 10.0)[1],
    value=clipped(desc["hba"], "hba", 0.0, 10.0),
    step=1.0,
    disabled=not manual,
)

num_rings = st.sidebar.number_input(
    "Number of rings",
    min_value=rng("num_rings", 0.0, 10.0)[0],
    max_value=rng("num_rings", 0.0, 10.0)[1],
    value=clipped(desc["num_rings"], "num_rings", 0.0, 10.0),
    step=1.0,
    disabled=not manual,
)

wastewater_type = st.sidebar.selectbox("Wastewater type", WASTEWATER_TYPES)
adsorption_type = st.sidebar.selectbox("Adsorption type", ADSORPTION_TYPES)
competitive = 0.0 if adsorption_type == "Single" else 1.0


# ============================================================
# Build model input dictionary
# ============================================================
if carbon > 0:
    h_c = hydrogen / carbon
    o_c = oxygen / carbon
    n_c = nitrogen / carbon
    on_c = (oxygen + nitrogen) / carbon
else:
    h_c = o_c = n_c = on_c = 0.0

input_data: Dict[str, float] = {
    "C": carbon,
    "H": hydrogen,
    "O": oxygen,
    "N": nitrogen,
    "(O+N)/C": on_c,
    "H/C": h_c,
    "O/C": o_c,
    "N/C": n_c,
    "Ash": ash,
    "Surface area": surface_area,
    "Pore volume": pore_volume,
    "Average pore size": average_pore_size,
    "Adsorption time": adsorption_time,
    "Initial concentration": initial_concentration,
    "Solution pH": solution_pH,
    "Adsorbent dosage": adsorbent_dosage,
    "Adsorption temperature": temperature,
    "Ion concentration": ion_concentration,
    "Humic acid": humic_acid,
    "competitive": competitive,
    "logp": logp,
    "tpsa": tpsa,
    "mol_weight": mol_weight,
    "hbd": hbd,
    "hba": hba,
    "num_rings": num_rings,
}

input_data = add_engineered_features(input_data)
input_data = add_wastewater_dummies(input_data, wastewater_type)


# ============================================================
# Main app layout
# ============================================================
st.subheader("Prediction")
st.caption(
    "Click the button after changing inputs. The app restricts inputs to the saved training range "
    "and uses a hybrid ML + Langmuir model for physically consistent capacity estimates. "
    "Pollutant descriptors can come from presets, SMILES, or manual input."
)

if st.button("Get prediction", type="primary"):
    validation_errors = validate_inputs(
        adsorption_time=adsorption_time,
        initial_concentration=initial_concentration,
        adsorbent_dosage=adsorbent_dosage,
        carbon=carbon,
    )
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)
        st.stop()

    try:
        x_new = build_input_frame(input_data, feature_columns)
        pred, qmax_pred, k_pred = hybrid_langmuir_predict(model, x_new)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted adsorption capacity", f"{pred:.2f} mg/g")
        p1, p2 = st.columns(2)
        p1.metric("Estimated qmax", f"{qmax_pred:.2f} mg/g")
        p2.metric("Estimated K", f"{k_pred:.4f} L/mg")

        if pred >= 100:
            st.success("High predicted adsorption capacity")
        elif pred >= 30:
            st.warning("Moderate predicted adsorption capacity")
        else:
            st.error("Low predicted adsorption capacity")

    with col2:
        st.markdown("#### Adsorption rationale")
        for rec in recommendation_text(input_data, pred):
            st.write(f"- {rec}")

        with st.expander("Hybrid Langmuir equation used"):
            st.latex(r"q_e = \frac{q_{max} K C_e}{1 + K C_e}")
            st.write(
                "The model predicts qmax and K from the selected descriptors; "
                "the Langmuir equation then computes the final capacity."
            )
else:
    st.info("Set the input values in the sidebar, then click **Get prediction**.")
