import sys
from pathlib import Path
import numpy as np

print("Starting v1.2 dataset generation...")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from seir_model import generate_patch_dataset

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "v1_2_label_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

(
    X_img,
    X_patch,
    y_remaining,
    y_risk,
    sim_id,
    time_idx,
    critical_time,
    infected_area,
    dominant_patch,
    reff,
) = generate_patch_dataset(
    num_sims=200,
    L=64,
    sim_steps=200,
    input_seq_len=10,
    horizon=30,
    seed=42,
    num_workers=0,
    theta_I=0.03,
    theta_A=0.03,
    persistent_k=3,
)

save_path = OUT_DIR / "seir_v1_2_h30_seed42.npz"

np.savez_compressed(
    save_path,
    X_img=X_img,
    X_patch=X_patch,
    y_remaining=y_remaining,
    y_risk=y_risk,
    sim_id=sim_id,
    time_idx=time_idx,
    critical_time=critical_time,
    infected_area=infected_area,
    dominant_patch=dominant_patch,
    reff=reff,
)

print("Saved to:", save_path)
print("X_img:", X_img.shape)
print("X_patch:", X_patch.shape)
print("y_remaining:", y_remaining.shape)
print("y_risk:", y_risk.shape)
print("sim_id:", sim_id.shape)
print("time_idx:", time_idx.shape)
print("critical_time:", critical_time.shape)
print("infected_area:", infected_area.shape)
print("dominant_patch:", dominant_patch.shape)
print("reff:", reff.shape)
print("risk ratio:", float(np.mean(y_risk)))
print("mean critical_time:", float(np.mean(critical_time)))
print("min critical_time:", int(np.min(critical_time)))
print("max critical_time:", int(np.max(critical_time)))
