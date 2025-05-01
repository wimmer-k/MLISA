# LISA: LIfetime measurements with Solid Active targets

**LISA** is a simulation and analysis toolkit developed for studying particles interacting with layered **solid active targets**. The project supports configurable beam parameters, in-target reactions, and detector response modeling - with outputs ready for machine learning-based lifetime and reaction position analysis.

---

##  Features

- Physics-driven simulation (relativistic energy loss)
- Configurable particle types, layer thicknesses, and beam properties
- Layer-specific energy smearing (Gaussian)
- YAML-configurable CLI for data generation
- Metadata logging and automatic output structuring
- Ready for ML: structured `.csv` outputs

---

## Quick Start

###  Simulate and smear (default config)

```bash
python scripts/generate_data.py
```

###  Use a custom config file

```bash
python scripts/generate_data.py --config configs/sim1.yaml
```

###   Just generate raw data

```bash
python scripts/generate_data.py --raw-only
```

###   Just smear existing raw data

```bash
python scripts/generate_data.py --smear-only
```

###   Control number of events

```bash
python scripts/generate_data.py -n 50000
```

## Output Structure

Each run creates a folder based on the config filename (e.g. sim1.yaml -> data/sim1/):

```bash
data/sim1/
|-- raw.csv                 # Truth-level simulation (no smearing)
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

paths:
  output_data_raw: unused  # paths are set automatically per config
  output_data_smeared: unused
```
