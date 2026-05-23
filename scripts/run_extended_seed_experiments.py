from pathlib import Path
import argparse
import subprocess
import sys
import os
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = ROOT / "scripts" / "train_deep_model.py"

SEEDS = [42, 123, 2026, 3407, 7777]
MODES = ["full", "img_only", "patch_only"]

COMMON_ARGS = {
    "num_sims": 300,
    "grid": 64,
    "sim_steps": 200,
    "horizon": 30,
    "reg_weight": 0.0,
    "weight_decay": 0.001,
}


def build_command(seed, mode):
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--seed", str(seed),
        "--num-sims", str(COMMON_ARGS["num_sims"]),
        "--grid", str(COMMON_ARGS["grid"]),
        "--sim-steps", str(COMMON_ARGS["sim_steps"]),
        "--horizon", str(COMMON_ARGS["horizon"]),
        "--input-mode", mode,
        "--reg-weight", str(COMMON_ARGS["reg_weight"]),
        "--weight-decay", str(COMMON_ARGS["weight_decay"]),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run extended robustness experiments for different seeds and input modes."
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the experiments. Without this flag, the script only prints commands.",
    )

    parser.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="Run or print commands only for one seed.",
    )

    parser.add_argument(
        "--only-mode",
        type=str,
        default=None,
        choices=MODES,
        help="Run or print commands only for one input mode.",
    )

    return parser.parse_args()


def filter_experiments(only_seed=None, only_mode=None):
    selected = []

    for seed in SEEDS:
        if only_seed is not None and seed != only_seed:
            continue

        for mode in MODES:
            if only_mode is not None and mode != only_mode:
                continue

            selected.append((seed, mode))

    return selected


def run_command(cmd):
    print("=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    start_time = datetime.now()

    env = os.environ.copy()

    extra_paths = [
        str(ROOT),
        str(ROOT / "src"),
    ]

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = ";".join(extra_paths + [old_pythonpath])
    else:
        env["PYTHONPATH"] = ";".join(extra_paths)

    result = subprocess.run(cmd, cwd=ROOT, env=env)

    end_time = datetime.now()
    elapsed = end_time - start_time

    if result.returncode != 0:
        print()
        print("Experiment failed.")
        print(f"Return code: {result.returncode}")
        print(f"Elapsed time: {elapsed}")
        sys.exit(result.returncode)

    print()
    print("Experiment finished successfully.")
    print(f"Elapsed time: {elapsed}")


def main():
    args = parse_args()

    if not PYTHON.exists():
        print(f"Cannot find Python interpreter: {PYTHON}")
        sys.exit(1)

    if not TRAIN_SCRIPT.exists():
        print(f"Cannot find training script: {TRAIN_SCRIPT}")
        print("Please check whether scripts/train_deep_model.py exists.")
        sys.exit(1)

    experiments = filter_experiments(
        only_seed=args.only_seed,
        only_mode=args.only_mode,
    )

    print("Extended robustness experiment runner")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Python: {PYTHON}")
    print(f"Training script: {TRAIN_SCRIPT}")
    print(f"Run mode: {'EXECUTE' if args.run else 'DRY RUN'}")
    print(f"Selected experiments: {len(experiments)}")
    print("=" * 80)

    commands = []

    for seed, mode in experiments:
        cmd = build_command(seed, mode)
        commands.append(cmd)
        print(" ".join(cmd))

    print("=" * 80)
    print(f"Total selected experiments: {len(commands)}")

    if not args.run:
        print()
        print("Dry run only. No experiment was executed.")
        print("Use --run to actually run the selected experiments.")
        return

    print()
    print("Execution mode enabled. Experiments will be run sequentially.")
    print()

    for index, cmd in enumerate(commands, start=1):
        print()
        print(f"Experiment {index}/{len(commands)}")
        run_command(cmd)

    print()
    print("All selected experiments finished.")


if __name__ == "__main__":
    main()
