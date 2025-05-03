import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_energy_distribution_3d(df, outdir, bins=50, by_reaction=False, max_layer=5, show_plot=True):
    layers = [f'dE_{i+1}' for i in range(max_layer)]
    if by_reaction:
        reaction_layers = sorted(df['reaction_layer'].unique())
        n = len(reaction_layers)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
    else:
        reaction_layers = [None]

    for i, rl in enumerate(reaction_layers):
        if by_reaction:
            ax = axes[i]
        else:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')

        data = df if rl is None else df[df['reaction_layer'] == rl]
        label = "all" if rl is None else f"reaction_{rl}"

        for j, col in enumerate(layers):
            counts, edges = np.histogram(data[col], bins=bins)
            xpos = (edges[:-1] + edges[1:]) / 2
            ypos = np.full_like(xpos, j + 1)
            zpos = np.zeros_like(xpos)
            dx = (edges[1] - edges[0]) * 0.9
            dy = 0.6
            dz = counts

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.7)

        ax.set_xlabel("Energy Loss [MeV]")
        ax.set_ylabel("Layer")
        ax.set_zlabel("Count")
        ax.set_title(f"Energy Loss ({label})")

        if not by_reaction:
            plt.tight_layout()
            if outdir:
                plt.savefig(Path(outdir) / f"3d_energy_{label}.png")
            if show_plot:
                plt.show()
            else:
                plt.close()

    if by_reaction:
        for k in range(i + 1, len(axes)):
            fig.delaxes(axes[k])
        plt.tight_layout()
        if outdir:
            plt.savefig(Path(outdir) / "3d_energy_by_reaction.png")
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_scatter_vs_b_in(df, outdir=None, show_plot=True, by_reaction=False):
    cols = [col for col in df.columns if col.startswith("dE_")]
    if by_reaction:
        reaction_layers = sorted(df['reaction_layer'].unique())
        n = len(reaction_layers)
        cols_per_fig = 3
        rows = (n + cols_per_fig - 1) // cols_per_fig
        fig, axes = plt.subplots(rows, cols_per_fig, figsize=(cols_per_fig * 6, rows * 4))
        axes = axes.flatten()

        for i, rl in enumerate(reaction_layers):
            ax = axes[i]
            sub = df[df['reaction_layer'] == rl]
            for col in cols:
                ax.scatter(sub["b_in"], sub[col], alpha=0.3, s=10, label=col)
            ax.set_title(f"reaction_layer = {rl}")
            ax.set_xlabel("b_in")
            ax.set_ylabel("dE")
            ax.legend()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if outdir:
            plt.savefig(Path(outdir) / "scatter_b_in_vs_dE_by_reaction.png")
        if show_plot:
            plt.show()
        else:
            plt.close()
    else:
        fig, axes = plt.subplots(len(cols), 1, figsize=(6, 4 * len(cols)), sharex=True)
        for i, col in enumerate(cols):
            ax = axes[i]
            ax.scatter(df["b_in"], df[col], alpha=0.3, s=10)
            ax.set_ylabel(col)
            ax.set_title(f"{col} vs b_in")
        axes[-1].set_xlabel("b_in")
        plt.tight_layout()
        if outdir:
            plt.savefig(Path(outdir) / "scatter_b_in_vs_dE.png")
        if show_plot:
            plt.show()
        else:
            plt.close()
            
def plot_reaction_depth_distribution(df, outdir=None, show_plot=True):
    if "reaction_depth" not in df.columns or "reaction_layer" not in df.columns:
        print("Missing columns in data.")
        return

    df_reacted = df[df["reaction_layer"] > 0]
    if df_reacted.empty:
        print("No reaction events found.")
        return

    layers = sorted(df_reacted["reaction_layer"].unique())
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(7, 4 * n_layers), sharex=False)

    if n_layers == 1:
        axes = [axes]  # ensure it's iterable

    for i, layer in enumerate(layers):
        ax = axes[i]
        subset = df_reacted[df_reacted["reaction_layer"] == layer]
        ax.scatter(subset["reaction_depth"], subset[f"dE_{layer}"], alpha=0.3, s=10)
        ax.set_title(f"Layer {layer}: Reaction Depth vs Energy Deposit")
        ax.set_xlabel("Reaction Depth [arb. units]")
        ax.set_ylabel(f"dE_{layer} [MeV]")

    plt.tight_layout()
    if outdir:
        plt.savefig(Path(outdir) / "depth_vs_energy_per_layer.png")
    if show_plot:
        plt.show()
    else:
        plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D plot of energy loss histograms")
    parser.add_argument("--data", type=str, required=True, help="Path to smeared.csv")
    parser.add_argument("--outdir", type=str, default=None, help="Folder to save plots")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots interactively")
    parser.add_argument("--by-reaction", action="store_true", help="Split plots by reaction_layer")
    parser.add_argument("--plot-type", type=str, default="hist3d", choices=["hist3d", "scatter", "depth"], help="Which plot to generate")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna()

    if args.plot_type == "hist3d":
        plot_energy_distribution_3d(
            df, outdir=args.outdir, show_plot=not args.no_show, by_reaction=args.by_reaction
        )
    elif args.plot_type == "scatter":
        plot_scatter_vs_b_in(
            df, outdir=args.outdir, show_plot=not args.no_show, by_reaction=args.by_reaction
        )
    elif args.plot_type == "depth":
        plot_reaction_depth_distribution(
            df, outdir=args.outdir, show_plot=not args.no_show
        )
