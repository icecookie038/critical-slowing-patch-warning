# Tropical-cyclogenesis dynamics (`tcg-dynamics`)

This repository develops a reproducible, physically controlled study of whether
satellite-observed convective organization adds information about western North
Pacific tropical cyclogenesis beyond the large-scale environment.

The project is deliberately staged. It does **not** currently claim a robust
genesis predictor, a discovered governing equation, or critical slowing down.

## Evidence completed

### v0.1: real Himawari feasibility pilot

Five 2019 western North Pacific cases, each sampled every three hours during the
72 hours before the first JTWC tropical-storm classification / 34-kt report. The
workflow established the IBTrACS/Himawari download, calibration, storm-following
crop, cloud-feature, and grouped-validation chain.

```bash
python run_pilot.py manifest
python run_pilot.py download
python run_pilot.py process
python run_pilot.py analyze
```

### v0.2: small developed/nondeveloped case-control pilot

Five IBTrACS disturbances that remained below 34 kt for their tracked lifetimes
were added as provisional negative cases. The workflow uses 72-hour trajectories,
leave-one-matched-pair-out validation, and exact within-pair permutation tests.
The sample remains too small and geographically mismatched for publishable
performance claims.

```bash
python run_case_control.py manifest
python run_case_control.py download
python run_case_control.py process
python run_case_control.py analyze
python run_case_control.py package
```

### v0.3: sparse ERA5 environment baseline (complete; full cadence pending)

The next decision gate adds three prespecified environmental variables:

- 850-hPa area-mean relative vorticity;
- 600/700/850-hPa mean relative humidity;
- 200--850-hPa deep-layer vertical wind shear.

The analysis compares metadata-only, environment-only, cloud-area-only,
environment-plus-cloud, and environment-plus-cloud-plus-organization models under
leave-one-matched-pair-out validation. Generate the exact CDS request plan with:

```bash
python run_environment.py plan
```

After accepting the ERA5 pressure-level dataset terms and configuring a personal
CDS API token:

```bash
python run_environment.py download
python run_environment.py process
python run_environment.py analyze
```

No API key is stored in this repository. See `docs/research_scope.md` for the
decision gates that must be passed before representation learning, system
identification, or early-warning claims.

An anonymous sparse pilot is also available through the NSF NCAR ERA5 mirror on
the AWS Open Data program. It streams only the required remote chunks at four
prespecified lead-time anchors and does not store the global NetCDF files:

```bash
python run_environment_public.py process
python run_environment_public.py validate-vorticity
python run_environment_public.py analyze
```

This four-anchor analysis is a diagnostic fallback, not a replacement for the
complete three-hourly ERA5 trajectory planned through CDS.

Current ten-track diagnostic results are environment-only AUC 0.72,
cloud-area-only AUC 1.00, environment plus cloud AUC 0.84, and environment plus
cloud plus organization AUC 0.88. Exact paired inference is limited by five
pairs, and metadata alone reaches AUC 0.88. These values are not treated as
publishable performance.

The existing 240-frame Himawari table can also be evaluated under causal
lead-time truncation. Each cutoff uses only images available at that time or
earlier:

```bash
python run_lead_time.py analyze
```

The causal cutoff curve shows that cold-cloud area first remains above AUC 0.80
at the 24-hour cutoff; organization does not remain above that threshold until
the final 3-hour cutoff. See
`outputs/reports/pilot_v0_3_direction_assessment.md` for the current go/no-go
decision.

## Installation

```bash
python -m pip install -r requirements.txt
```

Raw Himawari and ERA5 files are excluded from Git. The Himawari processor
decompresses one frame at a time to a temporary directory.

## Main outputs

- `data/interim/pilot_manifest.csv`: v0.1 storm/time/object manifest.
- `data/processed/case_control_track_windows.csv`: portable v0.2 paired trajectory windows.
- `outputs/tables/pilot_features.csv`: v0.1 frame-level features.
- `outputs/tables/case_control_features.csv`: v0.2 frame-level features used by
  the environment and lead-time analyses.
- `outputs/tables/case_control_metrics.csv`: v0.2 grouped metrics.
- `data/interim/era5_request_index.csv`: v0.3 case-month request inventory.
- `outputs/reports/pilot_v0_3_environment_status.md`: v0.3 readiness status.

## Event definition

The pilot event time is the first IBTrACS record where JTWC status is `TS` or
`USA_WIND >= 34 kt`. This explicit operational threshold supports reproducibility;
it is not treated as the unique physical definition of genesis.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
