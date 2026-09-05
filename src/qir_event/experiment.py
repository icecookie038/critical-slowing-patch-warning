"""Fixed-protocol phase-one experiment, synthetic smoke or explicit NPZ data."""
import argparse
import copy
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .events import RasterSeries, compile_events, sample_history, split_source_blocks
from .model import QIREventLite, event_probability, interval_nll
from .validation import block_weights, fit_calibration_offset, metrics, probability, SupportDiagnostic


def synthetic_series(seed, domain):
    """Engineering fixture only; explicitly not evidence for real transfer."""
    rng = np.random.default_rng(seed)
    times = np.array([0., 1., 2.5, 4., 7., 9., 12., 16.])
    h = w = 96
    env = np.broadcast_to(np.linspace(.1, .9, w)[None, :], (h, w))
    x = np.ones((len(times), h, w), dtype=np.uint8)
    for k, dt in enumerate(np.diff(times), start=1):
        old = x[k-1]
        recover = rng.random((h, w)) < -np.expm1(-dt*np.exp(-1.3-2*env))
        loss = rng.random((h, w)) < .10
        x[k] = np.where(old == 0, recover, ~loss)
    valid = rng.random(x.shape) > .015
    return RasterSeries(x, valid, times, np.broadcast_to(env, x.shape).copy(), domain, 8.)


def load_series(path):
    with np.load(path, allow_pickle=False) as d:
        s = RasterSeries(d["states"], d["valid"], d["times"], d["environment"],
                         str(d["domain"].item()), float(d["pixel_size_m"].item()))
    s.validate()
    return s


def buffered_coordinates(series, block_m, max_scale, per_block, seed):
    """Sample coordinates without labels; keep entire image footprints in blocks.

    Geometric separation prevents train/calibration image overlap. It does not
    prove independence; real spatial correlation range must be audited later.
    """
    block_px = int(round(block_m/series.pixel_size_m))
    margin = int(np.ceil(max_scale/2/series.pixel_size_m)) + 1
    if block_px <= 2*margin:
        raise ValueError("blocks too small for patch footprint buffer")
    rng = np.random.default_rng(seed)
    coords = []
    h, w = series.states.shape[1:]
    for r in range(0, h, block_px):
        for c in range(0, w, block_px):
            r1, c1 = min(r+block_px, h), min(c+block_px, w)
            if r1-r <= 2*margin or c1-c <= 2*margin:
                continue
            available = np.argwhere(series.valid[:, r+margin:r1-margin, c+margin:c1-margin].any(0))
            if not len(available):
                continue
            chosen = rng.choice(len(available), min(per_block, len(available)), replace=False)
            coords.extend(available[chosen] + [r+margin, c+margin])
    if not coords:
        raise ValueError("no eligible buffered pixels")
    return np.asarray(coords)


def make_tensors(rows, domains, scales, image_size):
    n = len(rows); t = max(len(s.times) for s in domains.values())
    images = np.zeros((n, t, len(scales), image_size, image_size), dtype=np.float32)
    masks = np.zeros_like(images, dtype=bool)
    times = np.zeros((n, t), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    for i, row in enumerate(rows.itertuples()):
        x, m, ts = sample_history(domains[row.domain], row, scales, image_size)
        length = len(ts)
        images[i, :length], masks[i, :length] = x, m
        times[i, :length] = ts-ts[0]
        times[i, length:] = ts[-1]-ts[0]
        lengths[i] = length
    bx = np.c_[rows.environment, np.log1p(rows.waiting_time)].astype(np.float32)
    return dict(images=torch.from_numpy(images), valid=torch.from_numpy(masks),
                times=torch.from_numpy(times), lengths=torch.from_numpy(lengths),
                baseline_x=torch.from_numpy(bx))


def predict(net, tensors, indices, batch_size=64):
    values = []
    net.eval()
    with torch.no_grad():
        for j in range(0, len(indices), batch_size):
            idx = indices[j:j+batch_size]
            values.append(net(**{k: v[idx] for k, v in tensors.items()}).numpy())
    return np.concatenate(values)


def train(net, tensors, rows, indices, epochs, lr, seed, frozen_baseline=None):
    torch.manual_seed(seed)
    if frozen_baseline is not None:
        net.baseline.load_state_dict(frozen_baseline.state_dict())
        for p in net.baseline.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    dt = torch.tensor(rows.delta_t.to_numpy(), dtype=torch.float32)
    y = torch.tensor(rows.event.to_numpy(), dtype=torch.float32)
    # Globally equal block weights. Fixed-size shuffled minibatches accumulate
    # the full weighted objective, rather than renormalizing within each batch.
    weights = torch.tensor(block_weights(rows.iloc[indices].block_id), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        net.train(); optimizer.zero_grad(); total = 0.
        order = rng.permutation(len(indices))
        for j in range(0, len(order), 64):
            local = order[j:j+64]; idx = indices[local]
            eta = net(**{k: v[idx] for k, v in tensors.items()})
            weight = weights[local]
            loss = interval_nll(eta, dt[idx], y[idx], weight)*weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step(); history.append(total)
    return history


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.input:
        datasets = [load_series(p) for p in args.input]
        provenance = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.input}
        evidence = "real_data_pilot_not_confirmatory"
    else:
        datasets = [synthetic_series(args.seed, "fixture_A"), synthetic_series(args.seed+1, "fixture_B")]
        provenance = {}; evidence = "synthetic_engineering_smoke_only"
    domains = {s.domain: s for s in datasets}
    if len(domains) != 2:
        raise ValueError("phase one requires exactly two distinct named domains")
    frames, ledgers = [], []
    for s in datasets:
        coords = buffered_coordinates(s, config["block_size_m"], max(config["scales_m"]),
                                      config["pixels_per_block"], args.seed)
        frame, ledger = compile_events(s, coords, config["block_size_m"])
        if frame.empty:
            raise ValueError(f"no observed risk intervals: {s.domain}")
        frames.append(frame); ledgers.append(ledger)
    rows = pd.concat(frames, ignore_index=True)
    raw_tensors = make_tensors(rows, domains, config["scales_m"], config["image_size"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    outputs = []; diagnostics = []; losses = {}; predictions = []
    names = list(domains)
    for source, target in [names, names[::-1]]:
        split = split_source_blocks(rows, source, target, seed=args.seed)
        tr, cal, te = (split[k] for k in ["train", "calibration", "test"])
        tensors = dict(raw_tensors)
        bx = raw_tensors["baseline_x"]
        mu, sd = bx[tr].mean(0), bx[tr].std(0).clamp_min(1e-6)
        tensors["baseline_x"] = (bx-mu)/sd
        baseline = None
        # Supports use only prediction-time covariates, never target outcomes.
        quality = raw_tensors["valid"].float().mean((-1, -2)).numpy()
        last = raw_tensors["lengths"].numpy()-1
        support_x = np.c_[bx.numpy(), quality[np.arange(len(rows)), last]]
        support = SupportDiagnostic().fit(support_x[tr], support_x[cal])
        accept = support.accepted(support_x[te])
        diagnostics.append(dict(source=source, target=target,
                                split_blocks={k: sorted(rows.iloc[v].block_id.unique()) for k,v in split.items()},
                                train_intervals=len(tr), calibration_intervals=len(cal), test_intervals=len(te),
                                support_threshold=support.threshold, accepted_fraction=float(accept.mean())))
        for mode in ["baseline", "quality", "spatial", "qir"]:
            torch.manual_seed(args.seed)
            net = QIREventLite(len(config["scales_m"]), config["hidden"], mode)
            history = train(net, tensors, rows, tr, config["epochs"], config["learning_rate"], args.seed, baseline)
            if mode == "baseline":
                baseline = copy.deepcopy(net.baseline)
            eta_cal = predict(net, tensors, cal)
            offset = fit_calibration_offset(eta_cal, rows.delta_t.to_numpy()[cal], rows.event.to_numpy()[cal], rows.block_id.to_numpy()[cal])
            eta = predict(net, tensors, te)
            y = rows.event.to_numpy()[te]; dt = rows.delta_t.to_numpy()[te]; blocks = rows.block_id.to_numpy()[te]
            for calibrated in [False, True]:
                prob = probability(eta+(offset if calibrated else 0), dt)
                outputs.append(dict(source=source, target=target, mode=mode, calibrated=calibrated,
                                    calibration_offset=offset if calibrated else 0., **metrics(y, prob, blocks)))
            calibrated_prob = probability(eta+offset, dt)
            for subset, selection in [("supported", accept), ("unsupported", ~accept)]:
                if selection.any():
                    outputs.append(dict(source=source, target=target, mode=mode, calibrated=True,
                                        subset=subset, calibration_offset=offset,
                                        **metrics(y[selection], calibrated_prob[selection], blocks[selection])))
            prediction = rows.iloc[te][["event_id", "block_id", "start_time", "end_time", "event"]].copy()
            prediction["source"] = source; prediction["target"] = target; prediction["mode"] = mode
            prediction["probability"] = calibrated_prob
            prediction["support_distance"] = support.distance(support_x[te])
            prediction["supported"] = accept
            predictions.append(prediction)
            torch.save(dict(state_dict=net.state_dict(), mode=mode, baseline_mean=mu, baseline_scale=sd,
                            calibration_offset=offset, config=config), out/f"{source}_to_{target}_{mode}.pt")
            # Fixed random spatial masking: retain the outcome truth, change only history.
            degraded = dict(tensors)
            generator = torch.Generator().manual_seed(args.seed)
            keep = torch.rand(tensors["valid"].shape, generator=generator) >= .25
            degraded["valid"] = tensors["valid"] & keep
            p_degraded = probability(predict(net, degraded, te)+offset, dt)
            outputs.append(dict(source=source, target=target, mode=mode, calibrated=True,
                                degradation="25_percent_history_mask", calibration_offset=offset,
                                **metrics(y, p_degraded, blocks)))
            losses[f"{source}->{target}:{mode}"] = {"first": history[0], "last": history[-1], "epochs": len(history)}
            print(f"{source}->{target} {mode}: loss {history[0]:.4f}->{history[-1]:.4f}", flush=True)
    summary = pd.DataFrame(outputs)
    summary.to_csv(out/"metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/"predictions.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(out/"episodes.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path("src/qir_event").glob("*.py"))}
    source_hashes[args.config] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    metadata = dict(evidence=evidence, seed=args.seed, config=config, input_sha256=provenance,
                    code_sha256=source_hashes,
                    code_commit=commit, dirty_worktree=dirty, python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, training_losses=losses,
                    n_intervals=len(rows), n_episodes=sum(map(len, ledgers)), splits=diagnostics,
                    limitations=["fixed small pilot; not an external efficacy claim", "no survival coverage guarantees",
                                 "source block separation does not establish spatial independence",
                                 "fixed epochs; target metrics never select hyperparameters",
                                 "M0 is a small neural baseline, not the frozen v3.7 spline baseline",
                                 "uncertainty ensembles, temporal dropout and second system not implemented"])
    (out/"run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qir_phase1.json")
    parser.add_argument("--input", nargs=2, type=Path)
    parser.add_argument("--output", default="results/qir_smoke")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
