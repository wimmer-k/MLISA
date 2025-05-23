# MLISA - Machine Learning for LISA

**MLISA** is a simulation and analysis toolkit developed for the **LISA** project (*LIfetime measurements with Solid Active targets*). It simulates beam particles passing through stacked active target layers, optionally undergoing nuclear reactions and depositing energy. The data is structured for direct use in machine learning models to analyze where (and later what) reactions occur.

This framework simulates particle interactions in layered solid targets, models detector resolution effects, and produces clean, structured outputs for analysis or ML training.

MLISA consists of modular Python scripts for data generation, simulation, visualization, and machine learning analysis.


##  Features

- Physics-driven simulation (relativistic energy loss)
- Configurable particle type, beam velocity distribution, and layer stack
- Layer-specific energy smearing (Gaussian resolution)
- CLI tools with YAML config support
- Rich visualization options for quality control
- Benchmarking and evaluation of ML models
- Structured `.csv` and `.yaml` outputs with metadata


## Simulation and data generation

MLISA simulates a beam of particles (defined by mass number `A` and charge `Z`) impinging on a multi-layered active target. Each particle is assigned a velocity (`b_in`) drawn from either a uniform or Gaussian distribution. As the particle traverses the target layers:

1. **Energy loss is calculated** in each layer using a relativistic Bethe-type formula, scaled by layer thickness and a tunable energy loss factor.
2. With a configurable probability, the particle may undergo a **nuclear reaction** in any given layer.
3. If a reaction occurs:
   - The energy loss behavior may change due to reduced charge (`Z`) of the product.
   - The particle continues with modified properties through the remaining layers.
4. The simulation records:
   - Initial and final velocities (`b_in`, `b_out`)
   - Energy loss per layer (`dE_1` to `dE_N`)
   - The layer in which the reaction occurred (or 0 if no reaction)

This process mimics what might happen in a real solid active target setup, providing realistic data for machine learning model development.

### Usage:

Simulate and smear (default config)

```bash
python3 scripts/generate_data.py
```

Use a custom config file

```bash
python3 scripts/generate_data.py --config config/sim1.yaml
```

Just generate raw data

```bash
python3 scripts/generate_data.py --raw-only
```

Just smear existing raw data

```bash
python3 scripts/generate_data.py --smear-only
```

Control number of events

```bash
python3 scripts/generate_data.py -n 50000
```

## Visualizing Energy Loss Distributions

The `visualize_data.py` script creates useful plots of the simulated data.

### Usage:

3D histogram of energy loss per layer

```bash
python3 scripts/visualize_data.py --data data/sim1/smeared.csv --plot-type hist3d
```

Add `--by-reaction` to show separate subplots for each reaction layer.

Scatter plots of energy loss vs. b_in

```bash
python3 scripts/visualize_data.py --data data/sim1/smeared.csv --plot-type scatter
```

You can combine with `--by-reaction` to generate separate scatter plots by reaction layer.

All plots can be saved using `--outdir results/sim1/` and disabled from displaying with `--no-show`.

## Model benchmarking

The `benchmark_models.py` script trains and evaluates multiple ML models for predicting the reaction layer.

### Usage:

Run ML training and evaluation

```bash
python3 scripts/benchmark_models.py --config config/sim1.yaml
```

Save all reports, plots, and feature importances

```bash
python3 scripts/benchmark_models.py --config config/sim1.yaml --save
```

Results are saved to a `results/<config_name>/` folder, including:

- Normalized confusion matrices (`.png` and `.csv`)
- Classification reports (`.txt`)
- Feature importances (`.csv`)

## Output Structure

Each simulation run produces structured outputs in a `data/<config_name>/` folder:

```bash
data/<config_name>/
|-- raw.csv                # Truth-level simulation data (no smearing)
|-- raw.meta.yaml          # Simulation parameters snapshot
|-- smeared.csv            # Includes Gaussian detector resolution effects
|-- smeared.meta.yaml      # Smearing parameters snapshot
```

Additional results from ML evaluation and visualizations are saved in:

```bash
results/<config_name>/
|-- *.png                  # Plots (3D histograms, scatter plots, confusion matrices)
|-- *.csv                  # Feature importance, confusion matrix tables
|-- *.txt                  # Model evaluation summaries
```

## Configuration Example (config/sim1.yaml)

```yaml
simulation:
  Z: 50
  A: 132
  n_events: 100000
  velocity_distribution: uniform
  b_min: 0.6
  b_max: 0.7
  reaction_prob: 0.1
  layers: 5
  eloss_scaling: 0.1
  layer_thickness: [0.4997, 0.5123, 0.5211, 0.5263, 0.5337]

detector:
  energy_resolution: [0.012, 0.012, 0.012, 0.012, 0.012]  # relative resolution, sigma

analysis:
  features: ['b_in', 'b_out', 'dE_1', 'dE_2', 'dE_3', 'dE_4', 'dE_5']
  target: reaction_layer
  test_size: 0.25
  models: ['logreg', 'rf', 'knn', 'gb']
```

## Coming Soon

- Intra-layer reaction depth modeling
- Support for different reaction types (Z change)
- Regression-based position prediction
- Integration of GEANT-based data

