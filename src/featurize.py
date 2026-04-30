from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit molecule, returning None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_fingerprint(mol):
    """Create a Morgan/ECFP-like fingerprint using RDKit's current generator API."""
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def fingerprint_to_array(fp) -> np.ndarray:
    """Convert an RDKit ExplicitBitVect to a numpy array."""
    if fp is None:
        return np.array([])
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def molecular_descriptors(mol) -> dict:
    """Calculate interpretable molecular descriptors relevant to separations."""
    if mol is None:
        return {
            "mol_weight": np.nan,
            "logp": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "tpsa": np.nan,
            "num_rings": np.nan,
            "aromatic_rings": np.nan,
            "rotatable_bonds": np.nan,
            "formal_charge": np.nan,
            "heavy_atoms": np.nan,
        }

    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Add RDKit molecule objects, fingerprints, validity flag, and descriptors."""
    out = df.copy()
    out["mol"] = out[smiles_col].apply(smiles_to_mol)
    out["valid_smiles"] = out["mol"].notna()
    out = out[out["valid_smiles"]].reset_index(drop=True)

    out["fingerprint"] = out["mol"].apply(morgan_fingerprint)
    descriptors = out["mol"].apply(molecular_descriptors).apply(pd.Series)
    out = pd.concat([out, descriptors], axis=1)
    return out
