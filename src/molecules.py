from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
except Exception:  # allows app to run even if RDKit is not installed yet
    Chem = None

# Small manually curated examples for common pollutants in the EC dataset and demo app.
# Extend this with PubChem lookup later.
POLLUTANT_SMILES = {
    "IBF": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
    "DCF": "O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl",  # diclofenac
    "NAP": "COc1ccc2ccccc2c1C(=O)O",          # naproxen, approximate
    "CBZ": "NC(=O)N1c2ccccc2C=Cc3ccccc31",    # carbamazepine
    "SMX": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", # sulfamethoxazole
    "TC": "CN(C)[C@H]1C(=O)/C(=C(/O)NC(N)=O)C(=O)[C@@]2(O)C(=O)C3=C(O)c4c(O)cccc4[C@H](C)[C@H]3C[C@H]12", # tetracycline
    "CIP": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", # ciprofloxacin
    "CAF": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",      # caffeine
    "BPA": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",   # bisphenol A
    "ATZ": "CCNc1nc(Cl)nc(NC(C)C)n1",         # atrazine
    "CAR": "NC(=O)N1c2ccccc2C=Cc3ccccc31",    # often carbamazepine in adsorption literature
}


def smiles_to_mol(smiles: str):
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def descriptors_from_smiles(smiles: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "mol_weight": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "aromatic_rings": np.nan,
            "rotatable_bonds": np.nan,
            "formal_charge": np.nan,
        }
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
    }


def add_molecular_descriptors(df: pd.DataFrame, smiles_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Add RDKit descriptors when a pollutant-to-SMILES mapping is available."""
    smiles_map = smiles_map or POLLUTANT_SMILES
    out = df.copy()
    if "Pollutant" not in out.columns:
        return out
    out["smiles"] = out["Pollutant"].map(smiles_map)
    desc = out["smiles"].apply(descriptors_from_smiles).apply(pd.Series)
    return pd.concat([out, desc], axis=1)
