import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml
from utils.data_loading import load_from_root, load_from_csv, filter_layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
    LGBMClassifier = None
    
# ------------------------------
# Model Definitions
# ------------------------------
MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gb": GradientBoostingClassifier(random_state=42),
    "svc": SVC(kernel="rbf", probability=True, random_state=42),
}
if _HAS_LGBM:
    MODELS["lgbm"] = LGBMClassifier(random_state=42)
    
SCALE_THESE = {"logreg", "svc", "knn"}

def _train_depth_regressors_per_layer(X_train, y_layer_train, y_depth_train, layers_unique):
    """
    Train one regressor per layer. Returns dict: layer -> regressor
    """
    regs = {}
    for lyr in layers_unique:
        idx = (y_layer_train == lyr)
        if idx.sum() == 0:
            continue
        # simple, robust default
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        reg.fit(X_train[idx], y_depth_train[idx])
        regs[lyr] = reg
    return regs

def _predict_depth_with_layer_models(regs_by_layer, X, layer_preds):
    """
    For each sample, pick the regressor matching its predicted layer.
    If missing, fall back to any available regressor (first one).
    """
    if not regs_by_layer:
        return np.full(len(X), np.nan)

    fallback = next(iter(regs_by_layer.values()))
    y_hat = np.empty(len(X), dtype=float)
    for i, lyr in enumerate(layer_preds):
        reg = regs_by_layer.get(lyr, fallback)
        y_hat[i] = reg.predict(X[i:i+1])[0]
    return y_hat

def plot_confusion_matrix(cm, classes, title, save_path=None, show_plot=True, save_csv_path=None):
    """
     Plot Confusion Matrix
    """    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if save_csv_path:
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        df_cm.round(3).to_csv(save_csv_path)
    if show_plot:
        plt.show()
    else:
        print(title)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        print(df_cm.round(4))
        plt.close()

def run_benchmark(config_path, save_outputs, show_plots, save_events=False):
    """
    Run the benchmark and save the output if needed
    """    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    tag = Path(config_path).stem
    # --- Flexible data loading with safe fallback to old behavior ---
    data_cfg   = config.get("data", {}) or {}
    root_file  = data_cfg.get("root_file")    # e.g. "data/wiktor/50Mn_49V_for_Kathrin.root"
    tree_name  = data_cfg.get("tree", None)
    max_layer  = int(data_cfg.get("max_layer", 5))

    if root_file:
        # uses the shared ROOT loader (b_in, b_out, reaction_layer, reaction_depth, dE_1..)
        from utils.data_loading import load_from_root  # keep this import local if you want zero impact when not using ROOT
        print(f"[load] ROOT: {root_file} (tree={tree_name or 'auto'})")
        df = load_from_root(root_file, tree_name=tree_name, max_layer=max_layer)
    else:
        # === old behavior (unchanged) ===
        data_path = Path("data") / tag / "smeared.csv"
        df = pd.read_csv(data_path)

    layer_min = int(config.get("analysis", {}).get("layer_min", 1))
    layer_max = int(config.get("analysis", {}).get("layer_max", 5))
    if "reaction_layer" in df.columns:
        df = df[(df["reaction_layer"] >= layer_min) & (df["reaction_layer"] <= layer_max)]
    
    df = df.dropna()  
    features = config["analysis"].get("features")
    target = config["analysis"].get("target", "reaction_layer")
    test_size = config["analysis"].get("test_size", 0.25)
    model_keys = config["analysis"].get("models", list(MODELS.keys()))

    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    results_dir = Path("results") / tag
    if save_outputs:
        results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running models: {model_keys}\n")
    for key in tqdm(model_keys):
        model = MODELS[key]
        print(f"\n=== {key.upper()} ===")
        if key in SCALE_THESE:
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc  = scaler.transform(X_test)
        else:
            X_train_proc, X_test_proc = X_train, X_test

        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
  
        # Classification report
        report = classification_report(y_test, y_pred, zero_division=0, digits =4)
        print(report)
        if save_outputs:
            with open(results_dir / f"{key}_report.txt", "w") as f:
                f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true') * 100
        cm_title = f"{key.upper()} - Normalized Confusion Matrix"
        save_img_path = results_dir / f"{key}_confusion.png" if save_outputs else None
        save_csv_path = results_dir / f"{key}_confusion.csv" if save_outputs else None
        plot_confusion_matrix(
            cm, classes=np.unique(y), title=cm_title,
            save_path=save_img_path, show_plot=show_plots, save_csv_path=save_csv_path
        )
        # Event-level export
        if save_outputs and save_events:
            test_index = X_test.index
            events_df = pd.DataFrame({
                "event_id": test_index,
                "true_layer": y_test.values,
                f"pred_layer_{key}": y_pred
            })
            # Save
            events_path = results_dir / f"{key}_events.csv"
            events_df.to_csv(events_path, index=False)
            print(f"[saved] {events_path}")
            
        if args.save_importance:
            # Feature importance or permutation importance
            print("\nFeature Importances:")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_[0]  # logreg, first class
            else:
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                importances = result.importances_mean

            for name, imp in zip(features, importances):
                print(f"  {name}: {imp:.3f}")

            if save_outputs:
                imp_df = pd.DataFrame({"feature": features, "importance": importances})
                imp_df.to_csv(results_dir / f"{key}_importance.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ML models on smeared simulation data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--save", action="store_true", help="Save reports and plots")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots")
    parser.add_argument("--save-importance", action="store_true", help="Also save feature importances")
    parser.add_argument("--save-events", action="store_true", help="Save per-event predictions CSVs for each model")
    args = parser.parse_args()

    run_benchmark(args.config, args.save, show_plots=not args.no_show, save_events=args.save_events)
