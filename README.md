# MLISA - Machine Learning for LISA

**MLISA** is a simulation and analysis toolkit developed for the **LISA** project (*LIfetime measurements with Solid Active targets*). It is designed to generate and prepare data from active target detectors for use in machine learning workflows, particularly for identifying reaction layers, estimating energy loss, and supporting lifetime analysis.

This framework simulates particle interactions in layered solid targets, models detector resolution effects, and produces clean, structured outputs for analysis or ML training.

---

##  Features

- Physics-driven simulation (relativistic energy loss)
- Configurable particle types, layer thicknesses, and beam properties
- Layer-specific energy smearing (Gaussian)
- YAML-configurable CLI for data generation
- Metadata logging and automatic output structuring
- Structured `.csv` output, ready for ML training or analysis
- Model benchmarking via a separate CLI tool

---

## Simulation and data generation

###  Simulate and smear (default config)

```bash
python3 scripts/generate_data.py
```

###  Use a custom config file

```bash
python3 scripts/generate_data.py --config configs/sim1.yaml
```

###   Just generate raw data

```bash
python3 scripts/generate_data.py --raw-only
```

###   Just smear existing raw data

```bash
python3 scripts/generate_data.py --smear-only
```

###   Control number of events

```bash
python3 scripts/generate_data.py -n 50000
```

## Model benchmarking

###   Train and evaluate multiple ML models on your generated data:

```bash
python3 scripts/benchmark_models.py --config configs/sim1.yaml
```

###   To also save confusion matrices, reports, and feature importances:

```bash
python3 scripts/benchmark_models.py --config configs/sim1.yaml --save
```
Results are saved to: results/sim1/

## Output Structure

Each run creates a folder based on the config filename (e.g. sim1.yaml -> data/sim1/):

```bash
data/sim1/
|-- raw.csv                # Truth-level simulation (no smearing)
|-- raw.meta.yaml          # Metadata snapshot
|-- smeared.csv            # With detector-like resolution applied
|-- smeared.meta.yaml      # Metadata snapshot
```

## Configuration Example (configs/sim1.yaml)

```yaml
simulation:
  Z: 50
  A: 132
  n_events: 100000
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
