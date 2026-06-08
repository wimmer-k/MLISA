import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml
import time
import onnx
import onnxruntime as ort
from utils.data_loading import load_from_root, load_from_csv, filter_layers

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def plot_confusion_matrix(cm, classes, title):
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
   
    print(title)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm.round(4))
    plt.show()

def run_predict(config_path,save_outputs=False,show_plots=True,
                    save_importance=False,save_events=False,save_model=False):
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
    trained_model = data_cfg.get("trained_model", None)  # e.g. "results/Agata/mlp_model.onnx"
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

    ################
    session = ort.InferenceSession(str(trained_model))
    x = np.array([[0.562300, 0.520406, 387.079272, 405.309043, 427.669856, 440.046838, 429.610327]], dtype=np.float32)  # Example input (adjust shape as needed)
    label, probs = session.run(None, {session.get_inputs()[0].name: x})
    print("Predicted label:", label)
    print("Predicted probabilities:", probs)

    ################

    layer_min = int(config.get("analysis", {}).get("layer_min", 1))
    layer_max = int(config.get("analysis", {}).get("layer_max", 5))
    if "reaction_layer" in df.columns:
        df = df[(df["reaction_layer"] >= layer_min) & (df["reaction_layer"] <= layer_max)]
    
    df = df.dropna()  
    features = config["analysis"].get("features")
    target = config["analysis"].get("target", "reaction_layer")
    test_size = config["analysis"].get("test_size", 0.25)

    print(df.head())

    X = df[features]
    y = df[target]
    
    print(X.head())
    print(y.head())


    results_dir = Path("results") / tag
    if save_outputs:
        results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running model: {trained_model}\n")
    
    model = onnx.load(trained_model)
    onnx.checker.check_model(model)

    print("Model loaded successfully")
    print("IR version:", model.ir_version)
    print("Producer:", model.producer_name)
    print("Opset imports:", model.opset_import)

    # Create ONNX Runtime session
    session = ort.InferenceSession(str(trained_model), providers=["CPUExecutionProvider"])

    # Inspect model input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Input name:", input_name)
    print("Output name:", output_name)
    print("Expected input shape:", session.get_inputs()[0].shape)
    print("Expected input type:", session.get_inputs()[0].type)

    # Convert pandas DataFrame to numpy float32
    X_np = X.to_numpy(dtype=np.float32)

    # Run prediction
    y_pred = session.run([output_name], {input_name: X_np})[0]

    print("Prediction shape:", y_pred.shape)
    print(y_pred[:10])

    # Classification report
    report = classification_report(y, y_pred, zero_division=0, digits =4)
    print("Classification Report:")
    print(report)
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, normalize='true') * 100
    cm_title = f"{trained_model.split('/')[-1]} - Normalized Confusion Matrix"
    plot_confusion_matrix(
        cm, classes=np.unique(y), title=cm_title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ML models on smeared simulation data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--save", action="store_true", help="Save reports and plots")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots")
    parser.add_argument("--save-importance", action="store_true", help="Also save feature importances")
    parser.add_argument("--save-events", action="store_true", help="Save per-event predictions CSVs for each model")
    parser.add_argument("--save-model", action="store_true", help="Save trained models in ONNX format for ROOT usage")
    args = parser.parse_args()

    run_predict(args.config,args.save,show_plots=not args.no_show,save_importance=args.save_importance,
        save_events=args.save_events,save_model=args.save_model)