import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# ---------------------------------------------------
# Heatmap by reaction (incl. "no reaction" as column 0)
# Rows = delta categories (0, +1, -1)
# Also adds "no reaction" truth/pred columns to augmented CSV
# ---------------------------------------------------
def plot_heatmaps_by_reaction_correct(config_path: str,
                                      model: str,
                                      outdir: str = "plots",
                                      bins: int = 50,
                                      dE_min: float | None = None,
                                      dE_max: float | None = None,
                                      save_augmented: bool = True):
    tag, events_csv, source_csv = infer_paths(config_path, model)
    if not events_csv.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_csv} "
                                f"(run benchmark with --save --save-events).")
    if not source_csv.exists():
        raise FileNotFoundError(f"smeared.csv not found: {source_csv}")

    # Load (mirror benchmark's dropna before split)
    events = pd.read_csv(events_csv)
    src = pd.read_csv(source_csv).dropna()

    pred_col = f"pred_layer_{model}"
    need_ev = {"event_id", "true_layer", pred_col}
    if not need_ev.issubset(events.columns):
        raise ValueError(f"Events CSV missing columns: {need_ev - set(events.columns)}")

    need_src = {"reaction_layer", "dE_1", "dE_2", "dE_3", "dE_4", "dE_5"}
    if not need_src.issubset(src.columns):
        raise ValueError(f"smeared.csv missing columns: {need_src - set(src.columns)}")

    # Compute delta = pred - true
    delta = events[pred_col].to_numpy() - events["true_layer"].to_numpy()
    events = events.assign(_delta=delta)

    # Add "no reaction" truth/pred columns (layer id == 0 means no reaction)
    events["no_reaction_true"] = (events["true_layer"] == 0).astype(int)
    events[f"no_reaction_pred_{model}"] = (events[pred_col] == 0).astype(int)

    # Merge dE_* and source reaction_layer via event_id (for plotting)
    src_idx = src.reset_index().rename(columns={"index": "event_id"})
    ev = events.merge(
        src_idx[["event_id", "reaction_layer", "dE_1", "dE_2", "dE_3", "dE_4", "dE_5"]],
        on="event_id", how="left", validate="one_to_one"
    )

    # Optionally save augmented events CSV
    if save_augmented:
        out_aug = Path("results") / tag / f"{model}_events_plus.csv"
        events.to_csv(out_aug, index=False)
        print(f"[saved] augmented events with no-reaction columns -> {out_aug}")

    # Energy binning range
    all_dE = ev[["dE_1", "dE_2", "dE_3", "dE_4", "dE_5"]].to_numpy(float)
    if dE_min is None:
        dE_min = float(np.nanmin(all_dE))
    if dE_max is None:
        dE_max = float(np.nanmax(all_dE))
    e_bins = np.linspace(dE_min, dE_max, bins + 1)

    # Reaction-layer columns: include 0 (no reaction) + 1..5
    r_layers = [0, 1, 2, 3, 4, 5]
    col_titles = ["no_reaction"] + [f"reaction_{k}" for k in range(1, 6)]
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

    # Draw with shared colorbar scale; empty bins -> white
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")  # masked zeros = white
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
            # Titles: first column is "no_reaction", others reaction_1..5
            if r_i == 0:
                ax.set_title(col_titles[c_i])
            if c_i == 0:
                ax.set_ylabel(f"{row_labels[d]}\nlayer index")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Energy loss dE")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"dE_{k}" for k in y_ticks])

    fig.suptitle(f"dE_k vs layer (heatmap) incl. no_reaction - model={model}", y=1.02)
    fig.tight_layout()

    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    out = outdir_p / f"heatmap_by_reaction_correct_{model}_with_noreac.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Heatmap analogue of visualize_data.py --plot-type hist3d --by-reaction "
            "with an extra column for 'no reaction' (layer 0). Rows are delta=0,+1,-1."
        )
    )
    ap.add_argument("--config", required=True, help="YAML used by benchmark (determines tag).")
    ap.add_argument("--model", required=True, help="Model key used in benchmark (logreg, rf, gb, knn, svc, lgbm).")
    ap.add_argument("--outdir", default="plots", help="Where to save the figure.")
    ap.add_argument("--bins", type=int, default=50, help="Number of energy-loss bins.")
    ap.add_argument("--dE-min", type=float, default=None, help="Lower bound for dE bins (auto if None).")
    ap.add_argument("--dE-max", type=float, default=None, help="Upper bound for dE bins (auto if None).")
    ap.add_argument("--no-save-augmented", action="store_true",
                    help="Do not save the augmented events CSV with no-reaction columns.")
    args = ap.parse_args()

    plot_heatmaps_by_reaction_correct(
        config_path=args.config,
        model=args.model,
        outdir=args.outdir,
        bins=args.bins,
        dE_min=args.dE_min,
        dE_max=args.dE_max,
        save_augmented=not args.no_save_augmented,
    )
