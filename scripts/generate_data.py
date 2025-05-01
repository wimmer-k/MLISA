import argparse
import pandas as pd
import numpy as np
import yaml
import subprocess
import datetime
from pathlib import Path
import sys
from tqdm import tqdm

# Setup path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.paths import PROJECT_ROOT, load_config

u = 931.5 # atomic mass unit

def kinetic_energy(A, beta):
    gamma = 1 / np.sqrt(1 - beta**2)
    return (gamma - 1) * A * u

def beta(A, E_kin):
    gamma = E_kin / (A * u) + 1
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta

def simulate_event(Z, A, b_min, b_max, layers, reaction_prob, layer_thickness, eloss_scaling = 1.0):
    """
    Simulate one particle through a layered detector with potential reaction.
    """
    b_in = np.random.uniform(b_min, b_max)
    b = b_in
    current_Z = Z
    E_losses = []
    reacted = False
    reaction_layer = 0  # 0 means no reaction
    reaction_depth = None
    
    for i in range(layers):
        thickness = layer_thickness[i]
        E_kin = kinetic_energy(A,b)
        # Simple stopping power model: dE/dx ~ Z^2 / b^2
        stopping_power = (current_Z ** 2) / (b ** 2) * eloss_scaling
        delta_E = stopping_power * thickness
        E_losses.append(delta_E)
        # print(current_Z,b,E_kin/A,delta_E)

        # Placeholder for future depth-resolved dE:
        # If reaction occurs here, to be adjusted later.
        # Reaction chance
        reaction_chance = reaction_prob * (thickness / np.mean(layer_thickness))
        if not reacted and np.random.rand() < reaction_chance:
            reacted = True
            reaction_layer = i + 1
            current_Z = max(current_Z - 1, 1)
            reaction_depth = np.random.uniform(0, thickness)

            # Future upgrade:
            # - split delta_E into dE_before + dE_after using reaction_depth
            # - update beta after first partial step
            
        # Update velocity
        E_kin = max(E_kin - delta_E, 1e-6)  # Prevent negative kinetic energy
        b = beta(A, E_kin)

    return {
        'b_in': b_in,
        'b_out': b,
        'reaction_layer': reaction_layer,
        'reaction_depth': reaction_depth,
        **{f'dE_{i+1}': E_losses[i] for i in range(layers)}
    }

def generate_dataset(n, **kwargs):
    return [simulate_event(**kwargs) for _ in tqdm(range(n), desc="Generating events")]

def apply_energy_smearing(df, resolutions):
    dE_cols = [col for col in df.columns if col.startswith("dE_")]
    if isinstance(resolutions, (float, int)):
        resolutions = [resolutions] * len(dE_cols)
    assert len(resolutions) == len(dE_cols), "Resolution list must match number of dE columns"

    df_smeared = df.copy()
    for i, col in enumerate(dE_cols):
        sigma = df[col] * resolutions[i]
        noise = np.random.normal(0, sigma)
        df_smeared[col] += noise
    return df_smeared

def save_metadata(config_path, n_events, output_path, args_dict):
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_events": n_events,
        "config_used": str(config_path),
        "args": args_dict,
    }

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        meta["git_commit"] = git_commit
    except Exception:
        meta["git_commit"] = "unknown"

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    meta["config_snapshot"] = config_data

    meta_path = Path(output_path).with_suffix(".meta.yaml")
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)
    print(f"Saved metadata to {meta_path}")

def main(config_path, generate=True, smear=True, n_override=None, args_dict=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sim_cfg = config["simulation"]
    det_cfg = config.get("detector", {})
    paths = config["paths"]

    raw_path = PROJECT_ROOT / paths["output_data_raw"]
    smeared_path = PROJECT_ROOT / paths["output_data_smeared"]

    n_events = n_override if n_override else sim_cfg["n_events"]

    # Load or generate
    if generate:
        print(f"Generating {n_events} events...")
        df = pd.DataFrame(generate_dataset(
            n=n_events,
            Z=sim_cfg["Z"],
            A=sim_cfg["A"],
            b_min=sim_cfg["b_min"],
            b_max=sim_cfg["b_max"],
            layers=sim_cfg["layers"],
            reaction_prob=sim_cfg["reaction_prob"],
            layer_thickness=sim_cfg["layer_thickness"],
            eloss_scaling=sim_cfg["eloss_scaling"]
        ))
        df.to_csv(raw_path, index=False)
        print(f"Saved raw dataset to {raw_path}")
        save_metadata(config_path, len(df), raw_path, args_dict)
    else:
        print(f"Loading raw dataset from {raw_path}")
        df = pd.read_csv(raw_path)

    # Smear if requested
    if smear:
        print("Applying smearing...")
        resolutions = det_cfg.get("energy_resolution", 0.05)
        df_smeared = apply_energy_smearing(df, resolutions)
        df_smeared.to_csv(smeared_path, index=False)
        print(f"Saved smeared dataset to {smeared_path}")
        save_metadata(config_path, len(df), smeared_path, args_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or smear simulation data")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--smeared", action="store_true", help="Save smeared data as well")
    parser.add_argument("--raw-only", action="store_true", help="Only generate raw data, no smearing")
    parser.add_argument("--smear-only", action="store_true", help="Only smear existing raw data")
    parser.add_argument("-n", "--n-events", type=int, help="Override number of events")

    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config

    if args.smear_only:
        main(config_path, generate=False, smear=True, args_dict=vars(args))
    elif args.raw_only:
        main(config_path, generate=True, smear=False, n_override=args.n_events, args_dict=vars(args))
    else:
        main(config_path, generate=True, smear=True, n_override=args.n_events, args_dict=vars(args))
