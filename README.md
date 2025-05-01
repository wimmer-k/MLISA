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