$ErrorActionPreference = "Stop"

$seed = 2026
$threshold = 0.50
$tag = "thr0p50"

$base = "data\processed\v1_2_label_fix\seir_v1_2_h30_seed$seed.npz"
$prepatch = "data\processed\v1_3_prepatch\seir_v1_3_prepatch_h30_seed$seed.npz"
$pwsi = "data\processed\v1_4_pwsi\seir_v1_4_pwsi_only_h30_seed$seed.npz"

if (!(Test-Path $base)) {
    throw "Missing base file: $base"
}

if (!(Test-Path $prepatch)) {
    throw "Missing prepatch file: $prepatch"
}

if (!(Test-Path $pwsi)) {
    throw "Missing pwsi file: $pwsi"
}

Write-Host "Running SEIR seed${seed} with patch-threshold=$threshold"

py scripts\analysis\analyze_integrated_patch_dynamics.py `
  --system seir `
  --seed $seed `
  --base-npz $base `
  --prepatch-npz $prepatch `
  --pwsi-npz $pwsi `
  --patch-direction high `
  --patch-threshold $threshold `
  --connectivity 8 `
  --persistent-k 3 `
  --sigma 2.0 `
  --outdir results\v2_4_integrated_patch_dynamics\seir_seed${seed}_$tag

py scripts\analysis\summarize_v2_4_representative_indicators.py `
  --indir results\v2_4_integrated_patch_dynamics\seir_seed${seed}_$tag
