import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------------
# STEP 1: Load dataset
# -------------------------
# Example: your CSV should have columns like ["SMILES", "Odor"]
df = pd.read_csv("Multi-Labelled_Smiles_Odors_dataset.csv")

print("Original dataset shape:", df.shape)

# -------------------------
# STEP 2: Clean & canonicalize SMILES
# -------------------------
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)
df = df.dropna(subset=["SMILES"])  # remove invalid molecules

# -------------------------
# STEP 3: Normalize descriptors
# -------------------------
# Example: descriptors are stored as "floral; fruity; sweet"
df["Odor"] = df["Odor"].str.lower().str.replace(",", ";").str.replace(";", ";")

# Split into lists
df["Odor_list"] = df["Odor"].apply(lambda x: [d.strip() for d in str(x).split(";") if d.strip()])

# Build descriptor vocabulary
all_descriptors = sorted(set([desc for sublist in df["Odor_list"] for desc in sublist]))
print("Total unique descriptors:", len(all_descriptors))
print(all_descriptors[:20])  # first 20 for preview

# -------------------------
# STEP 4: Multi-hot encode descriptors
# -------------------------
for desc in all_descriptors:
    df[desc] = df["Odor_list"].apply(lambda x: 1 if desc in x else 0)

# -------------------------
# STEP 5: Generate molecular fingerprints
# -------------------------
def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        return list(fp)
    else:
        return [0] * nBits

df["fingerprint"] = df["SMILES"].apply(smiles_to_fingerprint)

# -------------------------
# STEP 6: Save ML-ready dataset
# -------------------------
# Features (X): fingerprints
# Labels (Y): multi-hot odor descriptors
X = list(df["fingerprint"])
Y = df[all_descriptors].values

print("Final X shape:", len(X), "fingerprint length:", len(X[0]))
print("Final Y shape:", Y.shape)

# Save
df.to_pickle("odor_dataset_processed.pkl")  # keeps fingerprints & labels
df.to_csv("odor_dataset_processed.csv", index=False)

print("✅ Preprocessing complete! Data saved.")
