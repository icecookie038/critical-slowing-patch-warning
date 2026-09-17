# Patch-based early warning of critical slowing down in spatial SEIR systems

This repository is a public snapshot of the spatial SEIR research project. It contains simulation and feature-extraction code, training and evaluation scripts, historical result summaries, and manuscript drafts. The project studies patch-based early-warning signals for critical slowing down in a spatial spreading model; it does not include the separate 2026 modeling Problem B work.

## Layout

- `src/`: spatial SEIR simulation and feature-extraction modules.
- `scripts/`: training, evaluation, result aggregation, and table generation.
- `results_summary/`: historical CSV summaries from the existing research snapshot.
- `docs/`: experiment plans, methods and discussion drafts, and manuscript tables.
- `archive/v0_legacy/`: historical code snapshots.
- `requirements.txt`: the original, unpinned dependency list.

## Run from the repository root

```bash
python -m pip install -r requirements.txt
python scripts/train_deep_model.py --help
python scripts/train_patch_baselines.py --help
python scripts/make_final_figures.py --help
```

The scripts can generate datasets, model weights, figures, and intermediate results. Generated files are excluded by `.gitignore`; review script options before starting a full experiment.

## Results and provenance

The committed `results_summary/` and `docs/paper_tables/` are historical outputs. The earlier project documentation reported approximate AUC 0.86–0.91, AUPRC 0.82–0.89, F1 0.73–0.81, and successful warning lead times around 13–14 steps. These historical claims have not been independently rerun for this public snapshot. Exact environment reconstruction requires an original environment record or dependency lockfile.

Three historical CSV summaries contained machine-specific absolute `Run_Dir` paths. This public snapshot replaces those prefixes with `scripts/results/<run_name>` while preserving each run identifier and every metric; the original CSV files remain in the private archive.
