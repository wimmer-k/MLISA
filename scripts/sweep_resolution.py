import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import argparse
import shutil


def modify_config_resolution(config_path, new_config_path, resolution):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["detector"]["energy_resolution"] = [float(resolution)] * config["simulation"]["layers"]

    with open(new_config_path, "w") as f:
        yaml.dump(config, f)


def extract_accuracy_by_layer(report_path, n_layers):
    with open(report_path, "r") as f:
        lines = f.readlines()

    accuracies = [0.0] * (n_layers + 1)  # layers 0 to N
    for line in lines:
        try:
            if line.strip() and line.strip()[0].isdigit():
                parts = line.strip().split()
                label = int(parts[0])
                recall = float(parts[2])
                accuracies[label] = recall
        except:
            continue
    return accuracies


def run_benchmark_for_resolution(base_config, resolution, tmp_config_path, results_dir):
    raw_file = Path("data") / tmp_config_path.stem / "raw.csv"
    if not raw_file.exists():
        subprocess.run([
            "python3", "scripts/generate_data.py",
            "--config", str(tmp_config_path),
            "--raw-only"
        ])
    modify_config_resolution(base_config, tmp_config_path, resolution)
    subprocess.run([
        "python3", "scripts/generate_data.py",
        "--config", str(tmp_config_path),
        "--smear-only"
    ])

    subprocess.run([
        "python3", "scripts/benchmark_models.py",
        "--config", str(tmp_config_path),
        "--save",
        "--no-show"
    ])

    tag = Path(tmp_config_path).stem
    result_path = Path("results") / tag
    reports = list(result_path.glob("*_report.txt"))

    all_acc = {}
    for rep in reports:
        model = rep.stem.replace("_report", "")
        accs = extract_accuracy_by_layer(rep, n_layers=5)
        all_acc[model] = accs
    return all_acc


def plot_accuracy_vs_resolution(results, resolutions, outdir):
    # Save to CSV
    rows = []
    for i, res in enumerate(resolutions):
        for model, recalls in results[i].items():
            for layer, recall in enumerate(recalls):
                rows.append({
                    "resolution": float(res),
                    "model": model,
                    "layer": layer,
                    "recall": float(recall)
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(Path(outdir) / "recall_by_layer.csv", index=False)
    models = results[0].keys()
    layers = range(len(next(iter(results[0].values()))))

    fig, axes = plt.subplots(len(models), 1, figsize=(8, 5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        for l in layers:
            accs = [r[model][l] for r in results]
            ax.plot(resolutions, accs, label=f"Layer {l}")
        ax.set_title(f"{model.upper()} - Accuracy vs Resolution")
        ax.set_xlabel("Resolution (sigma)")
        ax.set_ylabel("Recall (per layer)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(outdir) / "accuracy_vs_resolution.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results/res_sweep")
    parser.add_argument("--res-range", nargs=3, type=float, default=[0.005, 0.025, 5],
                        help="start stop num_points")
    args = parser.parse_args()

    base_config = Path(args.config)
    outdir = Path(args.outdir)

    resolutions = np.linspace(args.res_range[0], args.res_range[1], int(args.res_range[2]))
    tmp_cfg_path = base_config.parent / (base_config.stem + "_tmp.yaml")

    all_results = []
    for r in resolutions:
        print(f"Running for resolution: {r:.4f}")
        res = run_benchmark_for_resolution(base_config, r, tmp_cfg_path, outdir)
        all_results.append(res)

    plot_accuracy_vs_resolution(all_results, resolutions, outdir)

    # Cleanup temp config
    if tmp_cfg_path.exists():
        tmp_cfg_path.unlink()
