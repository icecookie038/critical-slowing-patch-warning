# Data policy

Raw observations and reanalysis files are intentionally excluded from Git.

- `data/raw/ibtracs/`: IBTrACS source CSV.
- `data/raw/himawari/`: Himawari-8 Band-13 HSD files.
- `data/raw/era5/`: ERA5 pressure-level NetCDF files.
- `data/interim/`: reproducible manifests and request indexes.
- `data/processed/case_control_track_windows.csv`: portable 240-time case window
  table without machine-local cache paths.

The committed processed window table identifies the cases and observation times. Himawari files
can be restored by the v0.1/v0.2 download commands. ERA5 requests are generated
by `python run_environment.py plan`; downloading them requires a personal CDS API
token and acceptance of the ERA5 dataset terms.
