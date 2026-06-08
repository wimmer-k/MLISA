from __future__ import annotations
import numpy as np
import pandas as pd
import awkward as ak
import uproot
def filter_layers(df: pd.DataFrame, layer_min=0, layer_max=4):
    return df[(df["reaction_layer"] >= layer_min) & (df["reaction_layer"] <= layer_max)].copy()

def load_from_root(root_path: str, tree_name: str | None = None, max_layer: int = 5) -> pd.DataFrame:
    """
    Read ROOT TTree and produce a DataFrame with columns:
    b_in, b_out, reaction_layer, reaction_depth, dE_1..dE_max_layer
    """

    f = uproot.open(root_path)

    # Auto-pick a TTree if not given
    if tree_name is None:
        candidates = [k.split(";")[0] for k, v in f.items() if hasattr(v, "arrays")]
        if not candidates:
            raise ValueError("No TTree found in ROOT file.")
        tree_name = candidates[0]
        if (tree_name == "htree"):
            tree_name = "gtree"

    print(f"Reading tree '{tree_name}' from {root_path}...")
    tree = f[tree_name]
    
    needed = []
    if(tree_name == "gtree"):
        needed = ["Beta_before_layer_LISA", "Beta_after_layer_LISA", "n_reaction_layer_LISA", "pos_reaction_LISA", "e_layer_LISA"] 
    elif(tree_name == "str"):
        needed = ["finbetas", "fbetas", "fReactionLayer", "freacpos", "fEdet"]
    
    # Only read the needed branches
    arrs = tree.arrays(needed, library="ak")
    # --- Scalars we need ---
    # finbetas[0] -> b_in
    if tree_name == "gtree":
        finbetas_3d = ak.pad_none(arrs[needed[0]], 1, axis=1)
        # take first vector
        beta0 = finbetas_3d[:, 0]
        bx = beta0["fX"]
        by = beta0["fY"]
        bz = beta0["fZ"]
        b_in_ak = np.sqrt(bx*bx + by*by + bz*bz)
    elif tree_name == "str":
        finbetas = ak.pad_none(arrs[needed[0]], 5, axis=1)
        b_in_ak = finbetas[:, 0]
    b_in = ak.to_numpy(ak.fill_none(b_in_ak, np.nan))
    # fbetas[4] -> b_out
    if tree_name == "gtree":
        foutbetas_3d = ak.pad_none(arrs[needed[1]], 5, axis=1)
        # take first vector
        beta0 = foutbetas_3d[:, 4]
        bx = beta0["fX"]
        by = beta0["fY"]
        bz = beta0["fZ"]
        b_out_ak = np.sqrt(bx*bx + by*by + bz*bz)
    elif tree_name == "str":
        foutbetas = ak.pad_none(arrs[needed[1]], 5, axis=1)
        b_out_ak = foutbetas[:, 4]

    b_out = ak.to_numpy(ak.fill_none(b_out_ak, np.nan))
    # fReactionLayer -> reaction_layer (ints)
    if tree_name == "gtree":
        reaction_layer = ak.to_numpy(ak.values_astype(arrs[needed[2]], np.int64))
    elif tree_name == "str":
        reaction_layer = ak.to_numpy(ak.values_astype(arrs[needed[2]], np.int64))
    # freacpos.Z() -> reaction_depth (simple fallback to .fZ)
    pos = arrs[needed[3]]
    z = None
    fields = set(ak.fields(pos)) if ak.fields(pos) is not None else set()
    if "Z" in fields:
        z = pos["Z"]
    elif "fZ" in fields:
        z = pos["fZ"]

    # If neither Z nor fZ exists, just leave as NaN
    reaction_depth = (
        ak.to_numpy(ak.values_astype(ak.fill_none(z, np.nan), np.float64))
        if z is not None else
        np.full(len(reaction_layer), np.nan, dtype=float)
    )

    # fdEdx[0..max_layer-1] -> dE_1..dE_max_layer
    fdedx = ak.pad_none(arrs[needed[4]], max_layer, axis=1)
    dE_cols = {f"dE_{i+1}": ak.to_numpy(ak.fill_none(fdedx[:, i], np.nan))
               for i in range(max_layer)}

    # Assemble DataFrame and drop rows missing required features
    df = pd.DataFrame({
        "b_in": b_in,
        "b_out": b_out,
        "reaction_layer": reaction_layer,
        "reaction_depth": reaction_depth,
        **dE_cols,
    })

    feature_cols = ["b_in", "b_out", "reaction_layer"] + [f"dE_{i+1}" for i in range(max_layer)]
    df = df.dropna(subset=feature_cols)
    return df

def load_from_csv(csv_path: str, max_layer: int = 5) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep rows even if reaction_depth is NaN (no reaction)
    feature_cols = ["b_in", "b_out"] + [f"dE_{i+1}" for i in range(max_layer)] + ["reaction_layer"]
    df = df.dropna(subset=feature_cols)
    return df
