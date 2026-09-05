"""Conservative event compilation from irregular, partially observed binary rasters."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RasterSeries:
    states: np.ndarray  # T,H,W; 1=vegetation, 0=absence
    valid: np.ndarray
    times: np.ndarray  # numeric years, strictly increasing
    environment: np.ndarray  # T,H,W, available at each timestamp
    domain: str
    pixel_size_m: float

    def validate(self):
        if self.states.ndim != 3 or self.valid.shape != self.states.shape:
            raise ValueError("states and valid must have identical T,H,W shapes")
        if self.valid.dtype != np.bool_:
            raise ValueError("valid must be an explicit boolean mask")
        if self.environment.shape != self.states.shape:
            raise ValueError("environment must be T,H,W with time-safe availability")
        if len(self.times) != len(self.states) or len(self.times) < 2:
            raise ValueError("at least two matching timestamps required")
        if not np.isfinite(self.times).all() or not (np.diff(self.times) > 0).all():
            raise ValueError("timestamps must be finite and strictly increasing")
        if not np.isin(self.states[self.valid], [0, 1]).all():
            raise ValueError("observed states must be binary")
        if not np.isfinite(self.environment[self.valid]).all():
            raise ValueError("observed environment must be finite")
        if not np.isfinite(self.pixel_size_m) or self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive")


def compile_events(series, coordinates=None, block_size_m=256.0):
    """Compile first observed recovery per loss episode, retaining right censoring.

    A loss requires adjacent valid 1->0 observations. At the first missing
    endpoint, censor at the last valid start; do not infer absence/recovery
    through a gap. A later episode requires a new observed 1->0 transition.
    Repeated loss episodes remain grouped by pixel/block. Terminal losses are
    present in the episode ledger even though they contribute no exposure.
    Target is *first observed* recovery, not unobserved sub-interval transitions.
    """
    series.validate()
    if not np.isfinite(block_size_m) or block_size_m < series.pixel_size_m:
        raise ValueError("block size must be at least one pixel")
    t, h, w = series.states.shape
    if coordinates is None:
        coordinates = np.argwhere(np.any(series.valid, axis=0))
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be N,2")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("duplicate coordinates would duplicate event exposure")
    rows, episodes = [], []
    block_px = max(1, int(round(block_size_m / series.pixel_size_m)))
    for rr, cc in coordinates:
        r, c = int(rr), int(cc)
        if r != rr or c != cc or not (0 <= r < h and 0 <= c < w):
            raise ValueError("coordinate outside raster or noninteger")
        block = f"{series.domain}:{r // block_px}:{c // block_px}"
        k = 1
        while k < t:
            loss = (series.valid[k-1, r, c] and series.valid[k, r, c]
                    and series.states[k-1, r, c] == 1 and series.states[k, r, c] == 0)
            if not loss:
                k += 1
                continue
            loss_index = k
            event_id = f"{series.domain}:{r}:{c}:{loss_index}"
            event = dict(event_id=event_id, domain=series.domain, block_id=block,
                         row=r, col=c, loss_index=k, loss_time=float(series.times[k]))
            recovered, reason = False, "end_of_observation"
            while k < t-1:
                if not series.valid[k+1, r, c]:
                    reason = "first_missing_endpoint"
                    break
                recovered = bool(series.states[k+1, r, c] == 1)
                rows.append(dict(**event, start_index=k, end_index=k+1,
                                 start_time=float(series.times[k]),
                                 end_time=float(series.times[k+1]),
                                 delta_t=float(series.times[k+1]-series.times[k]),
                                 waiting_time=float(series.times[k]-series.times[loss_index]),
                                 environment=float(series.environment[k, r, c]),
                                 event=int(recovered)))
                k += 1
                if recovered:
                    reason = "observed_recovery"
                    break
            episodes.append(dict(**event, exit_time=float(series.times[k]),
                                 recovered=recovered, exit_reason=reason))
            k += 1
    return pd.DataFrame(rows), pd.DataFrame(episodes)


def sample_history(series, row, scales_m=(32., 64., 128.), size=16):
    """Physical centered patches, nearest sampling; padded/outside pixels invalid.

    Only frames at or before start_index are materialized. Future outcome
    images/quality are deliberately absent from the returned arrays.
    """
    if size < 2 or any(s <= 0 for s in scales_m):
        raise ValueError("positive physical scales and size >=2 required")
    k, r, c = int(row.start_index), int(row.row), int(row.col)
    h, w = series.states.shape[1:]
    images, masks = [], []
    for scale in scales_m:
        offsets = ((np.arange(size)+.5)/size-.5) * scale/series.pixel_size_m
        iy = np.rint(r+offsets).astype(int)
        ix = np.rint(c+offsets).astype(int)
        inside = (iy[:, None] >= 0) & (iy[:, None] < h) & (ix[None, :] >= 0) & (ix[None, :] < w)
        iy, ix = np.clip(iy, 0, h-1), np.clip(ix, 0, w-1)
        mask = series.valid[:k+1, iy[:, None], ix[None, :]] & inside
        image = series.states[:k+1, iy[:, None], ix[None, :]]
        images.append(np.where(mask, image, 0).astype(np.float32))
        masks.append(mask)
    return np.stack(images, axis=1), np.stack(masks, axis=1), series.times[:k+1].copy()


def split_source_blocks(rows, source, target, calibration_fraction=.25, seed=2026):
    if source == target or not 0 < calibration_fraction < 1:
        raise ValueError("distinct domains and a proper calibration fraction required")
    blocks = np.array(sorted(rows.loc[rows.domain == source, "block_id"].unique()))
    if len(blocks) < 4:
        raise ValueError("need at least four source blocks")
    np.random.default_rng(seed).shuffle(blocks)
    ncal = max(1, min(len(blocks)-2, int(np.ceil(len(blocks)*calibration_fraction))))
    cal = rows.domain.eq(source) & rows.block_id.isin(blocks[:ncal])
    train = rows.domain.eq(source) & ~cal
    test = rows.domain.eq(target)
    if not test.any():
        raise ValueError("target domain is empty")
    return {"train": np.flatnonzero(train), "calibration": np.flatnonzero(cal), "test": np.flatnonzero(test)}

