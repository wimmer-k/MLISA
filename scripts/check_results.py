import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data_loading import load_from_root, load_from_csv, filter_layers

# ------------------------------------
# Helpers (match benchmark conventions)
# ------------------------------------
def tag_from_config_path(config_path: str) -> str:
    return Path(config_path).stem

def infer_paths(config_path: str, model: str):
    tag = tag_from_config_path(config_path)
    events_csv = Path("results") / tag / f"{model}_events.csv"
    source_csv = Path("data") / tag / "smeared.csv"
    return tag, events_csv, source_csv

def load_source_df(config_path: str) -> pd.DataFrame:
    """
    Mirror benchmark_models.py:
      - If config.data.root_file present -> use ROOT loader (minimal columns)
      - Else fallback to CSV: data/<tag>/smeared.csv
      - Apply layer filter (analysis.layer_min/max; default 0..4)
      - Drop NaNs (as in benchmark)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    data_cfg = config.get("data", {}) or {}
    root_file = data_cfg.get("root_file")
    tree_name = data_cfg.get("tree", None)
    max_layer = int(data_cfg.get("max_layer", 5))

    layer_min = int(config.get("analysis", {}).get("layer_min", 1))
    layer_max = int(config.get("analysis", {}).get("layer_max", 5))

    if root_file:
        print(f"[load] ROOT: {root_file} (tree={tree_name or 'auto'})")
        df = load_from_root(root_file, tree_name=tree_name, max_layer=max_layer)
    else:
        tag = tag_from_config_path(config_path)
        csv_path = Path("data") / tag / "smeared.csv"
        print(f"[load] CSV (default): {csv_path}")
        df = pd.read_csv(csv_path)

    # Standard preprocessing to match benchmark
    if "reaction_layer" in df.columns:
        df = filter_layers(df, layer_min, layer_max)

    df = df.dropna()
    return df
# ---------------------------------------------------
# Heatmap analogue of visualize_data.py --by-reaction
# ---------------------------------------------------
def plot_heatmaps_by_reaction_correct(config_path: str,
                                      model: str,
                                      outdir: str = "plots",
                                      bins: int = 50,
                                      dE_min: float | None = None,
                                      dE_max: float | None = None):
    tag, events_csv, source_csv = infer_paths(config_path, model)
    if not events_csv.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_csv} "
                                f"(run benchmark with --save --save-events).")
    # Load predictions/events
    events = pd.read_csv(events_csv)

    pred_col = f"pred_layer_{model}"
    need_ev = {"event_id", "true_layer", pred_col}
    if not need_ev.issubset(events.columns):
        raise ValueError(f"Events CSV missing columns: {need_ev - set(events.columns)}")

    # Load source features the same way as benchmark (ROOT or CSV + filters)
    src = load_source_df(config_path)

    need_src = {"reaction_layer", "dE_1", "dE_2", "dE_3", "dE_4", "dE_5"}
    if not need_src.issubset(src.columns):
        missing = need_src - set(src.columns)
        raise ValueError(
            "Source data missing required columns. "
            f"Missing: {missing}. Did you run with max_layer>=5?"
        )

    # Merge dE_* and reaction_layer via event_id
    src_idx = src.reset_index().rename(columns={"index": "event_id"})
    delta = events[pred_col].to_numpy() - events["true_layer"].to_numpy()
    ev = (
        events.assign(_delta=delta)
        .merge(
            src_idx[["event_id", "reaction_layer", "dE_1", "dE_2", "dE_3", "dE_4", "dE_5"]],
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    )

    # Energy binning range
    all_dE = ev[["dE_1", "dE_2", "dE_3", "dE_4", "dE_5"]].to_numpy(float)
    if dE_min is None:
        dE_min = float(np.nanmin(all_dE))
    if dE_max is None:
        dE_max = float(np.nanmax(all_dE))
    e_bins = np.linspace(dE_min, dE_max, bins + 1)

    # Reaction layers (columns)
    r_layers = list(range(0, 5))  # enforce 1..5 columns
    cols = len(r_layers)
 
    # Rows = delta categories
    delta_rows = [0, +1, -1]
    row_labels = {0: "delta=0 (correct)", 1: "delta=+1", -1: "delta=-1"}
    rows = len(delta_rows)
 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.6), squeeze=False)


    # Matrix is [layer_index 1..5, energy bins]
    dE_cols = ["dE_1", "dE_2", "dE_3", "dE_4", "dE_5"]
    y_ticks = [1, 2, 3, 4, 5]

    # Precompute global vmax for consistent color scale
    global_vmax = 0
    mats = {}
    for r_i, d in enumerate(delta_rows):
        for c_i, rl in enumerate(r_layers):
            sub = ev[(ev["_delta"] == d) & (ev["reaction_layer"] == rl)]
            H = np.zeros((5, bins), dtype=int)
            if not sub.empty:
                for j, col in enumerate(dE_cols):
                    hist, _ = np.histogram(sub[col].to_numpy(float), bins=e_bins)
                    H[j, :] = hist
                global_vmax = max(global_vmax, int(H.max()))
            mats[(r_i, c_i)] = H

    # Draw with shared colorbar scale
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")  # masked (NaN) = white
    for r_i, d in enumerate(delta_rows):
        for c_i, rl in enumerate(r_layers):
            ax = axes[r_i][c_i]
            H = mats[(r_i, c_i)]
            Hm = np.ma.masked_equal(H, 0)
            im = ax.imshow(
                Hm, origin="lower", aspect="auto",
                extent=[e_bins[0], e_bins[-1], 0.5, 5.5],
                vmin=1, vmax=max(global_vmax, 1),  # counts start at 1
                cmap=cmap, interpolation="nearest"
            )
            if r_i == 0:
                ax.set_title(f"reaction_{rl}")
            if c_i == 0:
                ax.set_ylabel(f"{row_labels[d]}\nlayer index")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Energy loss dE")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"dE_{k}" for k in y_ticks])

    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    #cbar.set_label("Counts")
    fig.suptitle(f"dE_k vs layer (heatmap) by reaction layer - model={model}", y=1.02)
    fig.tight_layout()

    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    out = outdir_p / f"heatmap_by_reaction_correct_{model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Heatmap analogue of visualize_data.py --plot-type hist3d --by-reaction, using only correctly identified events.")
    ap.add_argument("--config", required=True, help="YAML used by benchmark (determines tag).")
    ap.add_argument("--model", required=True, help="Model key used in benchmark (logreg, rf, gb, knn).")
    ap.add_argument("--outdir", default="plots", help="Where to save the figure.")
    ap.add_argument("--bins", type=int, default=50, help="Number of energy-loss bins.")
    ap.add_argument("--dE-min", type=float, default=None, help="Lower bound for dE bins (auto if None).")
    ap.add_argument("--dE-max", type=float, default=None, help="Upper bound for dE bins (auto if None).")
    args = ap.parse_args()

    plot_heatmaps_by_reaction_correct(
        config_path=args.config,
        model=args.model,
        outdir=args.outdir,
        bins=args.bins,
        dE_min=args.dE_min,
        dE_max=args.dE_max,
    )
