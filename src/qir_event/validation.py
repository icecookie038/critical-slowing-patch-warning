"""Empirical calibration and support diagnostics, without coverage guarantees."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def block_weights(blocks):
    _, inverse, counts = np.unique(np.asarray(blocks), return_inverse=True, return_counts=True)
    w = 1. / counts[inverse]
    return w / w.sum()


def probability(eta, dt):
    return -np.expm1(-np.exp(np.clip(np.asarray(eta)+np.log(dt), -30., 20.)))


def metrics(y, p, blocks):
    w = block_weights(blocks)
    p = np.clip(p, 1e-12, 1-1e-12)
    return dict(n_intervals=len(y), n_blocks=len(np.unique(blocks)),
                brier=float(np.sum(w*(p-y)**2)),
                log_loss=float(-np.sum(w*(y*np.log(p)+(1-y)*np.log1p(-p)))),
                calibration_bias=float(np.sum(w*(p-y))),
                auc=float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None)


def fit_calibration_offset(eta, dt, y, blocks):
    """Fit one hazard intercept on source calibration blocks only."""
    w = block_weights(blocks)
    def objective(offset):
        p = np.clip(probability(eta+offset, dt), 1e-12, 1-1e-12)
        return -np.sum(w*(y*np.log(p)+(1-y)*np.log1p(-p)))
    result = minimize_scalar(objective, bounds=(-5., 5.), method="bounded")
    if not result.success:
        raise RuntimeError("source calibration did not converge")
    return float(result.x)


class SupportDiagnostic:
    """Training-only standardization and kNN; calibration-only distance cutoff.

    This is an empirical covariate extrapolation flag, NOT a calibrated
    probability of correctness or a solution to arbitrary concept shift.
    """
    def fit(self, train_x, calibration_x, quantile=.95):
        self.mean = train_x.mean(0)
        std = train_x.std(0)
        # A constant mask fraction should not inflate every distance by 1e6.
        # Retain its natural units (fractions in [0,1]) for this diagnostic.
        self.scale = np.where(std > 1e-6, std, 1.)
        self.knn = NearestNeighbors(n_neighbors=min(5, len(train_x))).fit((train_x-self.mean)/self.scale)
        self.threshold = float(np.quantile(self.distance(calibration_x), quantile))
        return self

    def distance(self, x):
        return self.knn.kneighbors((x-self.mean)/self.scale)[0].mean(1)

    def accepted(self, x):
        return self.distance(x) <= self.threshold
