$ErrorActionPreference = "Stop"

$seeds = @(42, 123, 2026)
$threshold = 0.50
$tag = "thr0p50"

foreach ($seed in $seeds) {
    $base = "data\processed\v2_0_vegetation_ca_clean\vegetation_ca_h30_seed$seed.npz"
    $prepatch = "data\processed\v2_0_vegetation_prepatch\seir_v1_3_prepatch_h30_seed$seed.npz"
    $pwsi = "data\processed\v2_0_vegetation_pwsi\seir_v1_4_pwsi_only_h30_seed$seed.npz"

    if (!(Test-Path $base)) {
        Write-Host "Skip seed${seed}: missing base file $base"
        continue
    }

    if (!(Test-Path $prepatch)) {
        Write-Host "Skip seed${seed}: missing prepatch file $prepatch"
        continue
    }

    if (!(Test-Path $pwsi)) {
        Write-Host "Skip seed${seed}: missing pwsi file $pwsi"
        continue
    }

    Write-Host "Running vegetation CA seed${seed} with patch-threshold=$threshold"

    py scripts\analysis\analyze_integrated_patch_dynamics.py `
      --system vegetation `
      --seed $seed `
      --base-npz $base `
      --prepatch-npz $prepatch `
      --pwsi-npz $pwsi `
      --patch-direction low `
      --patch-threshold $threshold `
      --connectivity 8 `
      --persistent-k 3 `
      --sigma 2.0 `
      --outdir results\v2_4_integrated_patch_dynamics\vegetation_seed${seed}_$tag

    py scripts\analysis\summarize_v2_4_representative_indicators.py `
      --indir results\v2_4_integrated_patch_dynamics\vegetation_seed${seed}_$tag
}
