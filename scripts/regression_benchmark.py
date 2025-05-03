import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS = {
    "rf": RandomForestRegressor(n_estimators=100, random_state=42),
    "gb": GradientBoostingRegressor(random_state=42),
    "svr": SVR(),
    "linreg": LinearRegression()
}

def compute_total_depth(row, thicknesses):
    if row["reaction_layer"] == 0:
        return np.nan
    idx = int(row["reaction_layer"]) - 1
    return sum(thicknesses[:idx]) + row["reaction_depth"]

def run_regression(config_path, show_plots, save_outputs):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tag = Path(config_path).stem
    df = pd.read_csv(Path("data") / tag / "smeared.csv")
    df = df.dropna(subset=[f"dE_{i+1}" for i in range(config["simulation"]["layers"])]).copy()

    thicknesses = config["simulation"]["layer_thickness"]
    df["reaction_occurred"] = (df["reaction_layer"] > 0).astype(int)
    df["reaction_total_depth"] = df.apply(lambda row: compute_total_depth(row, thicknesses), axis=1)

    # Only use events with a reaction
    reacted_df = df[df["reaction_occurred"] == 1].copy()

    features = config["analysis"]["features"]
    X = reacted_df[features]
    y = reacted_df["reaction_total_depth"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["analysis"].get("test_size", 0.25), random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    out_dir = Path("results") / tag
    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    for key, model in tqdm(MODELS.items(), desc="Models"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n=== {key.upper()} ===")
        print(f"MAE:  {mae:.4f} mm")
        print(f"RMSE: {rmse:.4f} mm")

        if save_outputs:
            pred_df = pd.DataFrame({"true": y_test, "pred": y_pred})
            pred_df.to_csv(out_dir / f"{key}_predictions.csv", index=False)

        layer_colors = reacted_df.loc[y_test.index, "reaction_layer"]
        unique_layers = sorted(layer_colors.unique())
        cmap = plt.cm.get_cmap('viridis', len(unique_layers))
        import matplotlib.colors as mcolors
        norm = mcolors.BoundaryNorm(boundaries=np.arange(min(unique_layers)-0.5, max(unique_layers)+1.5), ncolors=len(unique_layers))

        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(y_test, y_pred, c=layer_colors, cmap=cmap, norm=norm, alpha=0.6, s=10)
        cbar = plt.colorbar(scatter, ticks=unique_layers)
        cbar.set_label("Reaction Layer")
        
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("True Reaction Depth [mm]")
        plt.ylabel("Predicted Depth [mm]")
        plt.title(f"{key.upper()} Reaction Depth Prediction")
        plt.tight_layout()

        if save_outputs:
            plt.savefig(out_dir / f"{key}_prediction_plot.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run_regression(args.config, show_plots=not args.no_show, save_outputs=args.save)
