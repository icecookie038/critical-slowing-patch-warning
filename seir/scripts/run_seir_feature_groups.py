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

FEATURE_GROUPS = [
    "image_only",
    "classic_patch_only",
    "dynamic_patch_only",
    "classic_dynamic_patch",
    "full",
]

COMMON_ARGS = {
    "num_sims": 300,
    "grid": 64,
    "sim_steps": 200,
    "horizon": 30,
    "reg_weight": 0.0,
    "weight_decay": 0.001,
}


def build_command(seed: int, feature_group: str) -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--seed",
        str(seed),
        "--num-sims",
        str(COMMON_ARGS["num_sims"]),
        "--grid",
        str(COMMON_ARGS["grid"]),
        "--sim-steps",
        str(COMMON_ARGS["sim_steps"]),
        "--horizon",
        str(COMMON_ARGS["horizon"]),
        "--reg-weight",
        str(COMMON_ARGS["reg_weight"]),
        "--weight-decay",
        str(COMMON_ARGS["weight_decay"]),
        "--feature-group",
        feature_group,
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run v0.9 SEIR feature group experiments."
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run experiments. Without this flag, only print commands.",
    )

    parser.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="Run or print commands only for one seed.",
    )

    parser.add_argument(
        "--only-group",
        type=str,
        default=None,
        choices=FEATURE_GROUPS,
        help="Run or print commands only for one feature group.",
    )

    return parser.parse_args()


def filter_experiments(only_seed=None, only_group=None):
    selected = []

    for seed in SEEDS:
        if only_seed is not None and seed != only_seed:
            continue

        for group in FEATURE_GROUPS:
            if only_group is not None and group != only_group:
                continue

            selected.append((seed, group))

    return selected


def run_command(cmd: list[str]) -> None:
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
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + [old_pythonpath])
    else:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

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
        sys.exit(1)

    experiments = filter_experiments(
        only_seed=args.only_seed,
        only_group=args.only_group,
    )

    print("v0.9 SEIR feature group experiment runner")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Run mode: {'EXECUTE' if args.run else 'DRY RUN'}")
    print(f"Selected experiments: {len(experiments)}")
    print("=" * 80)

    commands = []

    for seed, group in experiments:
        cmd = build_command(seed, group)
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
    print("Execution mode enabled. Experiments will run sequentially.")

    for index, cmd in enumerate(commands, start=1):
        print()
        print(f"Experiment {index}/{len(commands)}")
        run_command(cmd)

    print()
    print("All selected v0.9 feature group experiments finished.")


if __name__ == "__main__":
    main()