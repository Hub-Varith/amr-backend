# genome2mic

Predict laboratory minimum inhibitory concentrations (MICs) from an assembled
bacterial genome, report uncertainty, and rank likely-active antibiotics.

**These are predictions of in-vitro susceptibility, not prescribing advice.**
Dose, route, and final drug choice depend on patient factors and remain with the
clinician. Confirm with standard antimicrobial susceptibility testing (AST).

## Model implementation

The training and prediction implementation is on **`Axion747/gk5s3`**. See
[MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md) for setup, required inputs,
training commands, model behavior, bundle files, and limitations.

To check out this branch on another machine:

```bash
git clone --branch Axion747/gk5s3 --single-branch \
  https://github.com/Hub-Varith/amr-backend.git
cd amr-backend
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest tests -q
```

The repository contains the model source, configs, tests, and curated demos.
Real training data and fitted model weights are separate, gitignored artifacts.
Training requires the stage outputs specified in
[DATA_CONTRACT.md](DATA_CONTRACT.md). Use `train --cv-only` during development to
fit the models without scoring the frozen test split.

The five supported species are *Escherichia coli*, *Klebsiella pneumoniae*,
*Staphylococcus aureus*, *Pseudomonas aeruginosa*, and *Acinetobacter baumannii*.
Each supported species–drug pair has its own model. The real-data release uses
known-AMR features; the broader pipeline also supports training-only unitig
features.

## Further documentation

- [Model implementation and training](MODEL_IMPLEMENTATION.md)
- [Data contract and schemas](DATA_CONTRACT.md)
- [Raw-data pipeline and VM setup](docs/VM_RUNBOOK.md)
- [Breakpoint verification status](docs/BREAKPOINT_VERIFICATION.md)
- [Synthetic demo](reports/synthetic_demo/README.md)
- [Real-genome pipeline demo](reports/hackathon_demo/README.md)

Synthetic results do not establish performance on real isolates. The real-genome
demo checks the prediction pipeline and does not validate susceptibility accuracy.
