"""Read the published van Belzen bundle without extracting or executing its code.

Static masks and inundation covariates are retrospective publication products.
This adapter is for a retrospective pilot, not a historical deployment replay.
"""
import argparse
import hashlib
import io
import json
import re
import warnings
import zipfile
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning

EXPECTED_MD5 = "eb545ba9e8e68e41c7f2988157e447c7"


def convert(bundle, output, stride=2):
    bundle, output = Path(bundle), Path(output)
    if stride < 1:
        raise ValueError("positive stride required")
    raw = bundle.read_bytes()
    if hashlib.md5(raw).hexdigest() != EXPECTED_MD5:
        raise ValueError("bundle checksum differs from published Zenodo file")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(source="https://zenodo.org/records/4998258", md5=EXPECTED_MD5,
                  sha256=hashlib.sha256(raw).hexdigest(), stride=stride,
                  sampling="regular pixel subsample; not area-averaged coarse graining",
                  pressure_availability="static retrospective map derived from 2005-2010 water levels",
                  mask_availability="static published study mask; not a dated operational QA product",
                  evidence="retrospective pilot only", domains=[])
    with zipfile.ZipFile(bundle) as z:
        def read_band(name):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with MemoryFile(z.read(name)) as mem, mem.open() as ds:
                    return ds.read(1)[::stride, ::stride]
        for site in ["Hellegat", "Paulina"]:
            prefix = f"Data Bundle NCOMMS/Data/data/{site}/"
            paths = sorted(n for n in z.namelist() if re.fullmatch(re.escape(prefix+site)+r"\d{4}bin\.tif", n))
            times = np.array([int(re.search(r"(\d{4})bin", n).group(1)) for n in paths], float)
            states = np.stack([read_band(p) for p in paths]).astype(np.uint8)
            mask = read_band(prefix+site+"_mask.tif")
            # These site-specific conventions follow StatisticalResilienceIndicators.m.
            valid = (mask == 0) if site == "Hellegat" else (mask > 0)
            env = np.loadtxt(io.BytesIO(z.read(prefix+site+"_IFmap.txt")), delimiter=",")[::stride, ::stride]
            if valid.shape != states.shape[1:] or env.shape != valid.shape:
                raise ValueError("published map shapes differ")
            valid &= np.isfinite(env) & (env >= 0) & (env <= 1)
            # Pixel size comes from the author's dx=0.25, not the TIFF identity transform.
            np.savez_compressed(output/f"{site}.npz", states=states,
                                valid=np.broadcast_to(valid, states.shape), times=times,
                                environment=np.broadcast_to(np.where(valid, env, 0).astype(np.float32), states.shape),
                                domain=site, pixel_size_m=.25*stride)
            report["domains"].append(dict(name=site, times=times.tolist(), shape=list(states.shape),
                                           eligible_pixels=int(valid.sum())))
    (output/"data_card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", default="data/processed/qir_marsh")
    p.add_argument("--stride", type=int, default=2)
    a = p.parse_args()
    convert(a.bundle, a.output, a.stride)

