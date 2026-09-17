$ErrorActionPreference = "Stop"

Write-Host "Running v2.4 integrated patch dynamics: vegetation seed42"

py scripts\analysis\analyze_integrated_patch_dynamics.py `
  --system vegetation `
  --seed 42 `
  --base-npz data\processed\v2_0_vegetation_ca_clean\vegetation_ca_h30_seed42.npz `
  --prepatch-npz data\processed\v2_0_vegetation_prepatch\seir_v1_3_prepatch_h30_seed42.npz `
  --pwsi-npz data\processed\v2_0_vegetation_pwsi\seir_v1_4_pwsi_only_h30_seed42.npz `
  --patch-direction low `
  --patch-threshold 0.35 `
  --connectivity 8 `
  --persistent-k 3 `
  --sigma 2.0
