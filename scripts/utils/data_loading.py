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
        candidates = [k for k, v in f.items() if hasattr(v, "arrays")]
        if not candidates:
            raise ValueError("No TTree found in ROOT file.")
        tree_name = candidates[0]

    tree = f[tree_name]
    
    # Only read the needed branches
    needed = ["finbetas", "fbetas", "fReactionLayer", "freacpos", "fEdet"]
    arrs = tree.arrays(needed, library="ak")

    # --- Scalars we need ---
    # finbetas[0] -> b_in
    finbetas = ak.pad_none(arrs["finbetas"], 1, axis=1)
    b_in = ak.to_numpy(ak.fill_none(finbetas[:, 0], np.nan))

    # fbetas[4] -> b_out
    fbetas = ak.pad_none(arrs["fbetas"], 5, axis=1)
    b_out = ak.to_numpy(ak.fill_none(fbetas[:, 4], np.nan))

    # fReactionLayer -> reaction_layer (ints)
    reaction_layer = ak.to_numpy(ak.values_astype(arrs["fReactionLayer"], np.int64))

    # freacpos.Z() -> reaction_depth (simple fallback to .fZ)
    pos = arrs["freacpos"]
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
    fdedx = ak.pad_none(arrs["fEdet"], max_layer, axis=1)
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
