import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from seir_model import generate_patch_dataset


settings = [
    {"name": "A_strict_005_005", "theta_I": 0.05, "theta_A": 0.05},
    {"name": "B_medium_003_003", "theta_I": 0.03, "theta_A": 0.03},
    {"name": "C_loose_002_003", "theta_I": 0.02, "theta_A": 0.03},
]

for cfg in settings:
    print("\n" + "=" * 60)
    print("Testing:", cfg["name"])
    print("=" * 60)

    data = generate_patch_dataset(
        num_sims=30,
        L=32,
        sim_steps=120,
        input_seq_len=10,
        horizon=30,
        seed=42,
        num_workers=0,
        theta_I=cfg["theta_I"],
        theta_A=cfg["theta_A"],
        persistent_k=3,
    )

    print("返回对象数量:", len(data))

    for i, item in enumerate(data):
        print(i, item.shape)