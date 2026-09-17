#!/usr/bin/env python3
"""Run the locked salt-marsh resolution, scale, and model-sensitivity audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
V35_SCRIPT = REPO_ROOT / "scripts" / "analysis" / (
    "run_v3_5_local_recovery_spatial_explanation.py"
)
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "analysis" / (
    "summarize_v3_6_salt_marsh_precision_audit.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "tidal_marsh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "v3_6_published_csd_benchmark",
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Reuse completed configuration directories and only rebuild summaries.",
    )
    return parser.parse_args()


def configuration_command(
    args: argparse.Namespace,
    *,
    output_name: str,
    patch_resolution_m: float,
    block_widths_m: tuple[float, ...],
    primary_width_m: float,
    alpha: float,
) -> list[str]:
    return [
        sys.executable,
        str(V35_SCRIPT),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_dir / output_name),
        "--block-widths-m",
        *(str(value) for value in block_widths_m),
        "--primary-width-m",
        str(primary_width_m),
        "--patch-resolution-m",
        str(patch_resolution_m),
        "--poisson-alpha",
        str(alpha),
        "--permutations",
        str(args.permutations),
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configurations = [
        {
            "output_name": f"salt_marsh_resolution_{resolution:g}m",
            "patch_resolution_m": resolution,
            "block_widths_m": (64.0,),
            "primary_width_m": 64.0,
            "alpha": 0.1,
        }
        for resolution in (0.25, 1.0, 5.0, 10.0)
    ]
    configurations.append(
        {
            "output_name": "salt_marsh_resolution_5m_multiscale",
            "patch_resolution_m": 5.0,
            "block_widths_m": (32.0, 64.0, 128.0),
            "primary_width_m": 64.0,
            "alpha": 0.1,
        }
    )
    configurations.extend(
        {
            "output_name": f"salt_marsh_5m_alpha_{alpha:g}",
            "patch_resolution_m": 5.0,
            "block_widths_m": (64.0,),
            "primary_width_m": 64.0,
            "alpha": alpha,
        }
        for alpha in (0.01, 1.0, 10.0)
    )
    configurations.extend(
        {
            "output_name": f"salt_marsh_5m_128m_alpha_{alpha:g}",
            "patch_resolution_m": 5.0,
            "block_widths_m": (128.0,),
            "primary_width_m": 128.0,
            "alpha": alpha,
        }
        for alpha in (0.01, 1.0, 10.0)
    )

    if not args.summarize_only:
        for index, configuration in enumerate(configurations, start=1):
            print(
                f"[{index}/{len(configurations)}] "
                f"{configuration['output_name']}",
                flush=True,
            )
            command = configuration_command(args, **configuration)
            subprocess.run(command, check=True, cwd=REPO_ROOT)

    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--benchmark-dir",
            str(args.output_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
