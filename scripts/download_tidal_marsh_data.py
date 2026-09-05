"""Download and verify the van Belzen et al. tidal-marsh data bundle.

The archive is a CC0 mirror of the Dryad record hosted by Zenodo.  Raw data
remain under ``data/raw`` and are excluded from Git.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_URL = (
    "https://zenodo.org/api/records/4998258/files/"
    "Data%20Bundle%20NCOMMS.zip/content"
)
ARCHIVE_MD5 = "eb545ba9e8e68e41c7f2988157e447c7"
ARCHIVE_NAME = "Data_Bundle_NCOMMS.zip"
EXPECTED_RELATIVE_FILE = Path(
    "Data Bundle NCOMMS/Data/data/Hellegat/Hellegat1976bin.tif"
)


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - required for published checksum
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "critical-slowing-patch-warning/3.5"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    partial.replace(target)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/tidal_marsh"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.destination
    expected = destination / EXPECTED_RELATIVE_FILE
    archive = destination / ARCHIVE_NAME

    if expected.exists() and not args.force:
        print(f"Dataset already available: {destination.resolve()}")
        return

    if args.force or not archive.exists():
        print(f"Downloading {ARCHIVE_URL}")
        download(ARCHIVE_URL, archive)

    observed_md5 = file_md5(archive)
    if observed_md5 != ARCHIVE_MD5:
        raise RuntimeError(
            f"Checksum mismatch: expected {ARCHIVE_MD5}, got {observed_md5}"
        )

    safe_extract(archive, destination)
    if not expected.exists():
        raise RuntimeError(f"Extraction did not create {expected}")

    print(f"Verified MD5: {observed_md5}")
    print(f"Extracted dataset: {destination.resolve()}")


if __name__ == "__main__":
    main()

