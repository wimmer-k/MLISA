import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# ------------------------------
# Model Definitions
# ------------------------------
MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "gb": GradientBoostingClassifier(random_state=42)
}

def plot_confusion_matrix(cm, classes, title, save_path=None):
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
    plt.show()

def run_benchmark(config_path, save_outputs):
    """
    Run the benchmark and save the output if needed
    """    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tag = Path(config_path).stem
    data_path = Path("data") / tag / "smeared.csv"
    df = pd.read_csv(data_path)
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
        #print("Class distribution in y_train:")
        #print(y_train.value_counts())
        #print("Overall class distribution in y:")
        #print(y.value_counts())
        if key == "logreg":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        import numpy as np
        unique_preds = np.unique(y_pred, return_counts=True)

        # Classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        if save_outputs:
            with open(results_dir / f"{key}_report.txt", "w") as f:
                f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize='true') * 100
        cm_title = f"{key.upper()} - Normalized Confusion Matrix"
        save_path = results_dir / f"{key}_confusion.png" if save_outputs else None
        plot_confusion_matrix(cm, classes=np.unique(y), title=cm_title, save_path=save_path)

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
    args = parser.parse_args()

    run_benchmark(args.config, args.save)
