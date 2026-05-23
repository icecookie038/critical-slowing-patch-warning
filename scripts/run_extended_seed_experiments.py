from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = ROOT / "scripts" / "train_deep_model.py"

SEEDS = [42, 123, 2026, 3407, 7777]
MODES = ["full", "img_only", "patch_only"]

COMMON_ARGS = {
    "sims": 300,
    "L": 64,
    "steps": 200,
    "h": 30,
    "rw": 0.0,
    "wd": 0.001,
}


def build_command(seed, mode):
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--seed", str(seed),
        "--sims", str(COMMON_ARGS["sims"]),
        "--L", str(COMMON_ARGS["L"]),
        "--steps", str(COMMON_ARGS["steps"]),
        "--h", str(COMMON_ARGS["h"]),
        "--mode", mode,
        "--rw", str(COMMON_ARGS["rw"]),
        "--wd", str(COMMON_ARGS["wd"]),
    ]


def main():
    if not TRAIN_SCRIPT.exists():
        print(f"Cannot find training script: {TRAIN_SCRIPT}")
        print("Please check whether scripts/train_deep_model.py exists.")
        return

    print("Extended seed experiment plan")
    print("=" * 60)

    commands = []

    for seed in SEEDS:
        for mode in MODES:
            cmd = build_command(seed, mode)
            commands.append(cmd)
            print(" ".join(cmd))

    print("=" * 60)
    print(f"Total experiments: {len(commands)}")
    print()
    print("This script currently prints the commands only.")
    print("After confirming the command format is correct, we will enable automatic execution.")


if __name__ == "__main__":
    main()