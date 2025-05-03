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
    """
    Calculate kinetic energy from relativistic beta
    """
    gamma = 1 / np.sqrt(1 - beta**2)
    return (gamma - 1) * A * u

def beta(A, E_kin):
    """
    Calculate relativistic beta from kinetic energy
    """
    gamma = E_kin / (A * u) + 1
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta

def simulate_event(Z, A, b_in, layers, reaction_prob, layer_thickness, delta_Z=1, eloss_scaling = 1.0):
    """
    Simulate one particle through a layered detector with potential reaction.
    """
    b = b_in
    current_Z = Z
    current_A = A
    E_losses = []
    reacted = False
    reaction_layer = 0  # 0 means no reaction
    reaction_depth = None
    
    for i in range(layers):
        thickness = layer_thickness[i]
        reaction_chance = reaction_prob * (thickness / np.mean(layer_thickness))
        if not reacted and np.random.rand() < reaction_chance:
            reacted = True
            reaction_layer = i + 1
            reaction_depth = np.random.uniform(0, thickness)
            
            # Pre-reaction energy loss
            sp_pre = (current_Z ** 2) / (b ** 2) * eloss_scaling
            dE_pre = sp_pre * reaction_depth
            
            E_kin_pre = kinetic_energy(current_A, b)
            E_kin_post = max(E_kin_pre - dE_pre, 1e-6)
            b_post = beta(current_A, E_kin_post)
            
            # Apply reaction (Z and A change, velocity remains same for this layer)
            current_Z = max(current_Z - delta_Z, 1)
            current_A = max(current_A - delta_Z, 1)

            # Post-reaction energy loss
            d2 = thickness - reaction_depth
            sp_post = (current_Z ** 2) / (b_post ** 2) * eloss_scaling
            dE_post = sp_post * d2
            b = beta(current_A, max(E_kin_post - dE_post, 1e-6))

            delta_E = dE_pre + dE_post
            
        else:
            sp = (current_Z ** 2) / (b ** 2) * eloss_scaling
            delta_E = sp * thickness
            
            E_kin = kinetic_energy(current_A, b)
            E_kin = max(E_kin - delta_E, 1e-6)
            b = beta(current_A, E_kin)

        E_losses.append(delta_E)

        # Update kinetic energy and velocity after the whole layer
        E_kin = kinetic_energy(current_A, b)
        E_kin = max(E_kin - delta_E, 1e-6)
        b = beta(current_A, E_kin)

    return {
        'b_in': b_in,
        'b_out': b,
        'reaction_layer': reaction_layer,
        'reaction_depth': reaction_depth,
        **{f'dE_{i+1}': E_losses[i] for i in range(layers)}
    }

def generate_dataset(n, velocity_distribution="uniform", b_min=None, b_max=None, b_mean=None, b_sigma=None, **kwargs):
    """
    Generate a dataset of events using simulate_event.
    Handles velocity distribution setup.
    """
    if velocity_distribution == "gaussian":
        assert b_mean is not None and b_sigma is not None, "b_mean and b_sigma must be set for gaussian distribution"
        betas = np.random.normal(loc=b_mean, scale=b_sigma, size=n)
        betas = np.clip(betas, 0, 1)
    else:
        assert b_min is not None and b_max is not None, "b_min and b_max must be set for uniform distribution"
        betas = np.random.uniform(low=b_min, high=b_max, size=n)

    return pd.DataFrame([
        simulate_event(b_in=b, **kwargs) for b in tqdm(betas, desc="Simulating events")
    ])

def apply_energy_smearing(df, resolutions):
    """
    Add optional detector resolution.
    """
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
    """
    Save the meta data for reproducibility
    """
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
    meta["output_tag"] = output_path.parent.name
    meta_path = Path(output_path).with_suffix(".meta.yaml")
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)
    print(f"Saved metadata to {meta_path}")

def main(config_path, generate=True, smear=True, n_override=None, args_dict=None):
    """
    Run simulation based on configuration file and commandline input, save data and meta data to file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sim_cfg = config["simulation"]
    det_cfg = config.get("detector", {})

    # Use config filename (without extension) as output tag
    config_tag = Path(config_path).stem
    output_dir = PROJECT_ROOT / "data" / config_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw.csv"
    smeared_path = output_dir / "smeared.csv"

    n_events = n_override if n_override else sim_cfg["n_events"]

    # Load or generate
    if generate:
        print(f"Generating {n_events} events...")
        df = generate_dataset(
            n=sim_cfg["n_events"],
            velocity_distribution=sim_cfg.get("velocity_distribution", "uniform"),
            b_min=sim_cfg.get("b_min"),
            b_max=sim_cfg.get("b_max"),
            b_mean=sim_cfg.get("b_mean"),
            b_sigma=sim_cfg.get("b_sigma"),
            Z=sim_cfg["Z"],
            A=sim_cfg["A"],
            delta_Z=sim_cfg["delta_Z"],
            layers=sim_cfg["layers"],
            eloss_scaling=sim_cfg["eloss_scaling"],
            reaction_prob=sim_cfg["reaction_prob"],
            layer_thickness=sim_cfg["layer_thickness"]
        )
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
