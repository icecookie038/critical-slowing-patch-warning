# QIR Event research direction

The new direction is quality-aware event-time modelling from irregular raster
observations. Start with [the implementation and fixed pilot protocol](docs/qir_event_phase1.md)
and [the source register](docs/qir_sources.md).

Historical SEIR/CA experiments remain available as controlled simulations. The
recovered v3.5-v3.7 work concerns real-data recovery and spatial-proxy auditing.
Those results must not be replaced by simulation accuracy or by this small prototype.

Phase one implements event compilation, a masked multiscale CNN with causal GRU,
cloglog likelihood, ablations, source-block calibration and a support diagnostic.
It includes a runnable synthetic smoke test and a small retrospective marsh pilot.
It does not yet establish transferable learned spatial information or calibrated
event-time uncertainty, and it has not been tested on a second ecosystem.

```bash
python -m pip install -r requirements-qir.txt
python -m pytest tests/test_qir_event.py -q
python -m src.qir_event.experiment
```

Every run writes source/input hashes, parameters, split IDs, metrics, predictions,
episode ledgers and checkpoints. Raw datasets and checkpoints are ignored by Git.

