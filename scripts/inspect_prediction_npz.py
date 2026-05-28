from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "scripts" / "results"


def find_prediction_file():
    patterns = [
        "seed42_sims300_L64_steps200_h30_groupclassic_patch_only*/validation_predictions.npz",
        "seed42_sims300_L64_steps200_h30_grouptraditional_ews_only*/validation_predictions.npz",
        "seed42_sims300_L64_steps200_h30_groupfull*/validation_predictions.npz",
    ]

    for pattern in patterns:
        candidates = sorted(RESULTS_DIR.glob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError("No validation_predictions.npz found for seed42.")


def main():
    path = find_prediction_file()

    print("Inspecting:")
    print(path)
    print()

    data = np.load(path, allow_pickle=True)

    print("Keys:")
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    print()
    print("Preview:")
    for key in data.files:
        arr = data[key]
        flat = arr.reshape(-1)
        print(f"{key}: first values = {flat[:10]}")


if __name__ == "__main__":
    main()