from pathlib import Path
import yaml
import sys

def get_project_root():
    # If running in script: use __file__
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    # If running in Jupyter: use current working directory (typically in notebooks/)
    else:
        return Path.cwd().parents[0]  # from notebooks/ -> project root

PROJECT_ROOT = get_project_root()

def load_config():
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)
