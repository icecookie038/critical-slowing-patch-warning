#!/usr/bin/env python3
"""Download and verify the published datasets used by the v3.6 benchmark.

The files come from the supplementary material of Clements & Ozgul (2016),
Dryad data for Dai et al. (2015), and Figshare data for Rindi et al. (2018).
Raw data remain under ``data/raw`` and are excluded from Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


CLEMENTS_FILES = {
    "experimental_data.xls": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fncomms10984/MediaObjects/"
        "41467_2016_BFncomms10984_MOESM1694_ESM.xls",
        "15e7e71bb7cb8a7b34f118bf5907d876",
    ),
    "bifurcation_code.txt": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fncomms10984/MediaObjects/"
        "41467_2016_BFncomms10984_MOESM1695_ESM.txt",
        "177a6e152a0acb17e61e08ff2bd317bb",
    ),
    "ews_code.txt": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fncomms10984/MediaObjects/"
        "41467_2016_BFncomms10984_MOESM1696_ESM.txt",
        "8362df038a2dc9d5644e77074ea25758",
    ),
}

DAI_FILE = (
    "data_deterioration.zip",
    "https://datadryad.org/api/v2/files/38203/download",
    "f739beab74af2f981d410c10277ec364",
)

RINDI_ARTICLE_API = "https://api.figshare.com/v2/articles/6200822"
RINDI_FILES = {
    "cys_cover_avg_1yst.txt": "d929fb705a7f5996b8415e7843ed1203",
    "cys_cover_avg_2nd.txt": "c1036005b61db6c1f217e28d623a22f5",
    "dat_cl_1yst.txt": "4549dd387a9f95ffdbf83dad868e000d",
    "dat_cl_2nd.txt": "a4e5f7dff18150bbd97cc14c0c4194af",
    "Exp_data_2014.txt": "f68a4627374dc1880bf4b649d1089323",
    "Exp_data_2015.txt": "d185a3a226ab43a4467370295b668058",
}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - published repository checksum
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "critical-slowing-patch-warning/3.6"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        return json.load(response)


def download_verified(
    url: str,
    target: Path,
    expected_md5: str,
    *,
    force: bool,
) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        observed = file_md5(target)
        if observed != expected_md5:
            raise RuntimeError(
                f"Existing file failed checksum: {target}; rerun with --force"
            )
        return "already_verified"

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "critical-slowing-patch-warning/3.6"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    observed = file_md5(partial)
    if observed != expected_md5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {target.name}: expected {expected_md5}, "
            f"observed {observed}"
        )
    partial.replace(target)
    return "downloaded_and_verified"


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/csd_benchmark"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, object]] = []

    clements_root = args.destination / "clements_2016"
    for filename, (url, expected) in CLEMENTS_FILES.items():
        target = clements_root / filename
        status = download_verified(url, target, expected, force=args.force)
        records.append(
            {
                "dataset": "clements_2016",
                "file": filename,
                "md5": expected,
                "status": status,
            }
        )

    dai_name, dai_url, dai_md5 = DAI_FILE
    dai_root = args.destination / "dai_2015"
    dai_archive = dai_root / dai_name
    status = download_verified(dai_url, dai_archive, dai_md5, force=args.force)
    extracted = dai_root / "data_deterioration"
    expected_extracted = extracted / "all_DF.txt"
    if args.force or not expected_extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        safe_extract(dai_archive, extracted)
    records.append(
        {
            "dataset": "dai_2015",
            "file": dai_name,
            "md5": dai_md5,
            "status": status,
        }
    )

    metadata = fetch_json(RINDI_ARTICLE_API)
    available = {
        str(item["name"]): item
        for item in metadata.get("files", [])  # type: ignore[union-attr]
    }
    rindi_root = args.destination / "rindi_2018"
    for filename, expected in RINDI_FILES.items():
        if filename not in available:
            raise RuntimeError(f"Figshare record is missing {filename}")
        item = available[filename]
        repository_md5 = item.get("computed_md5") or item.get("supplied_md5")
        if repository_md5 != expected:
            raise RuntimeError(
                f"Figshare checksum changed for {filename}: {repository_md5}"
            )
        target = rindi_root / filename
        status = download_verified(
            str(item["download_url"]), target, expected, force=args.force
        )
        records.append(
            {
                "dataset": "rindi_2018",
                "file": filename,
                "md5": expected,
                "status": status,
            }
        )

    for record in records:
        print(
            f"{record['dataset']}: {record['file']} "
            f"[{record['status']}; md5={record['md5']}]"
        )
    print(f"Published CSD data ready: {args.destination.resolve()}")


if __name__ == "__main__":
    main()
