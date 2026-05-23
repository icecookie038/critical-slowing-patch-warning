# seir_model.py
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.ndimage import label, generate_binary_structure, gaussian_filter, gaussian_filter1d
from multiprocessing import Pool
from tqdm import tqdm


# =========================
# 全局默认参数
# =========================
DEFAULT_SEED = 42
MIN_PATCH_SIZE = 5

BASE_GRID_POPULATION = 100
MIN_DENSITY_RATIO = 0.2
MAX_DENSITY_RATIO = 8.0
DENSITY_SMOOTH = 2.5

INITIAL_INFECTION_DENSITY = 0.0015

MOVE_PROB = 0.003
SEASONAL_AMPLITUDE = 0.25
SEASONAL_PERIOD = 365

AIR_TRAVEL_STRENGTH = 0.002
LONG_DISTANCE_EVENT_PROB = 0.20
SUPER_SPREADER_PROB = 0.001
SUPER_SPREADER_STRENGTH = 0.12


# =========================
# 工具函数
# =========================
def compute_csd_series(density_history, window=20):
    density_history = np.asarray(density_history, dtype=np.float64)
    n = len(density_history)

    ac1_vals = np.zeros(n)
    var_vals = np.zeros(n)
    trend_vals = np.zeros(n)

    for t in range(window, n):
        w = density_history[t - window:t]

        ac1_vals[t] = safe_corrcoef(w[:-1], w[1:])
        var_vals[t] = np.var(w)

        x = np.arange(len(w))
        if np.std(w) > 1e-12:
            trend_vals[t] = np.polyfit(x, w, 1)[0]

    ac1_smooth = gaussian_filter1d(ac1_vals, sigma=2.0)
    var_smooth = gaussian_filter1d(var_vals, sigma=2.0)
    trend_smooth = gaussian_filter1d(trend_vals, sigma=2.0)

    return ac1_smooth, var_smooth, trend_smooth
def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    c = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(c):
        return 0.0

    return float(c)


def jensen_shannon_divergence(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    eps = 1e-12
    a = a + eps
    b = b + eps

    a = a / np.sum(a)
    b = b / np.sum(b)

    m = 0.5 * (a + b)

    kl_a = np.sum(a * np.log(a / m))
    kl_b = np.sum(b * np.log(b / m))

    jsd = 0.5 * (kl_a + kl_b)

    if not np.isfinite(jsd):
        return 0.0

    return float(jsd)


def detect_critical_point(
    density_history,
    window=20,
    threshold_ac1=0.80,
    min_peak_density=1e-4,
    min_rel_density=0.08,
):
    """
    更严格的临界点检测函数。

    返回：
        int: 临界点位置
        None: 该模拟没有可靠临界点，应跳过

    设计逻辑：
    1. 如果整条轨迹感染峰值太低，说明没有形成有效爆发，跳过；
    2. 候选点处感染密度不能接近 0；
    3. 候选点处 Trend 必须为正；
    4. 候选点处 Variance 必须达到一定水平；
    5. 优先选择 AC1 高、Variance 高、Trend 正的最早时间点。
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    density_history = np.asarray(density_history, dtype=np.float64)
    n = len(density_history)

    if n < 2 * window + 5:
        return None

    max_density = float(np.max(density_history))

    # 没有真正爆发，跳过
    if max_density < min_peak_density:
        return None

    ac1_vals = np.zeros(n, dtype=np.float64)
    var_vals = np.zeros(n, dtype=np.float64)
    trend_vals = np.zeros(n, dtype=np.float64)

    for t in range(window, n):
        w = density_history[t - window:t]

        if len(w) < 3:
            continue

        # AC1
        if np.std(w[:-1]) > 1e-12 and np.std(w[1:]) > 1e-12:
            c = np.corrcoef(w[:-1], w[1:])[0, 1]
            ac1_vals[t] = 0.0 if not np.isfinite(c) else c
        else:
            ac1_vals[t] = 0.0

        # Variance
        var_vals[t] = np.var(w)

        # Trend
        x = np.arange(len(w))
        if np.std(w) > 1e-12:
            trend_vals[t] = np.polyfit(x, w, 1)[0]
        else:
            trend_vals[t] = 0.0

    ac1_smooth = gaussian_filter1d(ac1_vals, sigma=2.0)
    var_smooth = gaussian_filter1d(var_vals, sigma=2.0)
    trend_smooth = gaussian_filter1d(trend_vals, sigma=2.0)

    # 当前感染密度必须达到峰值的一定比例，避免 Case 3 这种早期空转误判
    density_threshold = max(min_peak_density, min_rel_density * max_density)

    # 方差阈值：只在有效非零区域中计算
    valid_var_pool = var_smooth[window:]
    valid_var_pool = valid_var_pool[np.isfinite(valid_var_pool)]

    if len(valid_var_pool) == 0 or np.max(valid_var_pool) <= 0:
        return None

    var_threshold = np.percentile(valid_var_pool, 60)

    valid_mask = (
        (density_history >= density_threshold)
        & (trend_smooth > 0)
        & (var_smooth >= var_threshold)
        & (np.arange(n) >= window + 5)
    )

    # 优先找 AC1 超过阈值的最早候选点
    candidates = np.where(valid_mask & (ac1_smooth >= threshold_ac1))[0]

    if len(candidates) > 0:
        return int(candidates[0])

    # 如果没有 AC1 超阈值，就用综合分数兜底
    var_norm = var_smooth / (np.max(var_smooth) + 1e-12)

    positive_trend = np.maximum(trend_smooth, 0.0)
    trend_norm = positive_trend / (np.max(positive_trend) + 1e-12)

    ac1_norm = np.clip(ac1_smooth, 0.0, 1.0)

    score = 0.4 * ac1_norm + 0.4 * var_norm + 0.2 * trend_norm
    score[~valid_mask] = -np.inf

    if not np.isfinite(np.max(score)):
        return None

    return int(np.argmax(score))
# =========================
# SEIR 空间模型
# =========================
class PatchSEIR:
    """
    二维网格 SEIR 模型。

    每个格点包含：
    S, E, I, R, population, density

    同时考虑：
    1. 人口密度异质性
    2. 近距离传播
    3. 长距离传播
    4. beta 随时间随机变化
    5. 季节因素
    6. 行为反馈
    7. 斑块指标
    """

    def __init__(
        self,
        L=64,
        sim_steps=200,
        beta_params=None,
        sigma=1.0 / 5.5,
        gamma=1.0 / 10.0,
        seed=None,
    ):
        self.L = int(L)
        self.sim_steps = int(sim_steps)
        self.sigma = float(sigma)
        self.gamma = float(gamma)

        self.rng = np.random.default_rng(seed)
        self.time_step = 0

        if beta_params is None:
            beta_params = {
                "initial_range": (0.06, 0.14),
                "daily_volatility": 0.015,
                "weekly_cycle_amp": 0.06,
                "jump_prob": 0.01,
            }

        self.beta_params = beta_params

        self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self._generate_density()
        self._initialize_population()
        self._initialize_beta()
        self._initialize_state()
        self._initialize_density_effects()
        self._initialize_long_distance()

        self.seasonal_amplitude = SEASONAL_AMPLITUDE
        self.seasonal_period = SEASONAL_PERIOD
        self.time_of_year = int(self.rng.integers(0, 365))

        self.behavioral_factor = 1.0
        self.awareness_threshold = 0.01

        self.current_temperature = 20.0
        self.current_humidity = 0.6
        self.effective_beta = self.current_beta

        self.history = {
            "beta": [],
            "I_total": [],
            "density": [],
            "features": [],
            "patch_sizes": [],
            "jsd": [],
        }

    # =========================
    # 初始化
    # =========================
    def _generate_density(self):
        raw = self.rng.lognormal(mean=0.0, sigma=1.0, size=(self.L, self.L))
        raw = gaussian_filter(raw, sigma=DENSITY_SMOOTH)

        raw_min = raw.min()
        raw_max = raw.max()

        density = (raw - raw_min) / (raw_max - raw_min + 1e-12)
        density = density * (MAX_DENSITY_RATIO - MIN_DENSITY_RATIO) + MIN_DENSITY_RATIO

        self.grid_density = density.astype(np.float32)

    def _initialize_population(self):
        pop = np.round(self.grid_density * BASE_GRID_POPULATION).astype(np.int64)
        pop[pop < 1] = 1
        self.grid_population = pop

    def _initialize_beta(self):
        low, high = self.beta_params["initial_range"]

        if high <= low:
            high = low + 0.02

        self.current_beta = float(self.rng.uniform(low, high))
        self.beta_history = [self.current_beta]

        self.weekly_phase = float(self.rng.uniform(0, 2 * np.pi))
        self.monthly_trend = 0.0

        self.external_shocks = []
        n_shocks = int(self.rng.integers(1, 4))
        shock_times = self.rng.choice(
            np.arange(10, max(11, self.sim_steps - 10)),
            size=n_shocks,
            replace=False,
        )

        for t in shock_times:
            self.external_shocks.append(
                {
                    "time": int(t),
                    "magnitude": float(self.rng.uniform(-0.05, 0.08)),
                    "duration": int(self.rng.integers(8, 25)),
                }
            )

    def _initialize_state(self):
        self.S = self.grid_population.copy()
        self.E = np.zeros((self.L, self.L), dtype=np.int64)
        self.I = np.zeros((self.L, self.L), dtype=np.int64)
        self.R = np.zeros((self.L, self.L), dtype=np.int64)

        n_init = max(1, int(self.L * self.L * INITIAL_INFECTION_DENSITY))
        weights = self.grid_density.flatten()
        weights = weights / weights.sum()

        chosen = self.rng.choice(
            self.L * self.L,
            size=n_init,
            replace=False,
            p=weights,
        )

        for idx in chosen:
            i, j = divmod(int(idx), self.L)
            infected = min(int(self.rng.integers(1, 5)), int(self.S[i, j]))
            self.I[i, j] += infected
            self.S[i, j] -= infected

    def _initialize_density_effects(self):
        d = self.grid_density

        self.beta_density_factor = np.ones((self.L, self.L), dtype=np.float32)
        self.beta_density_factor[d < 1.0] = 0.8
        self.beta_density_factor[(d >= 1.0) & (d < 2.0)] = 1.0
        self.beta_density_factor[(d >= 2.0) & (d < 4.0)] = 1.25
        self.beta_density_factor[d >= 4.0] = 1.55

        mobility = 1.0 / (d + 0.5)
        mobility = mobility / np.mean(mobility)
        mobility = np.clip(mobility, 0.5, 2.0)
        self.mobility_density_factor = mobility.astype(np.float32)

    def _initialize_long_distance(self):
        self.transport_hubs = []
        self.hub_connectivity = {}

        n_hubs = max(3, self.L // 20)

        xs = np.arange(self.L)[:, None]
        ys = np.arange(self.L)[None, :]

        weights = self.grid_density.flatten()
        weights = weights / weights.sum()

        hub_indices = self.rng.choice(
            self.L * self.L,
            size=n_hubs,
            replace=False,
            p=weights,
        )

        for idx in hub_indices:
            i, j = divmod(int(idx), self.L)
            hub = (i, j)
            self.transport_hubs.append(hub)

            dist = np.sqrt((xs - i) ** 2 + (ys - j) ** 2)
            conn = np.exp(-dist / (self.L / 5.0 + 1e-8))
            self.hub_connectivity[hub] = conn.astype(np.float32)

    # =========================
    # 动力学更新
    # =========================
    def _update_stochastic_beta(self):
        prev = self.current_beta

        daily_noise = self.rng.normal(0.0, self.beta_params["daily_volatility"])

        weekly = self.beta_params["weekly_cycle_amp"] * math.sin(
            2 * math.pi * self.time_step / 7.0 + self.weekly_phase
        )

        if self.time_step > 0 and self.time_step % 30 == 0:
            self.monthly_trend += float(self.rng.normal(0.0, 0.015))

        jump = 0.0
        if self.rng.random() < self.beta_params["jump_prob"]:
            jump = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.02, 0.06))

        external = 0.0
        for shock in self.external_shocks:
            start = shock["time"]
            end = shock["time"] + shock["duration"]
            if start <= self.time_step < end:
                progress = (self.time_step - start) / max(1, shock["duration"])
                external += shock["magnitude"] * (1.0 - progress)

        total_pop = np.sum(self.grid_population)
        inf_density = np.sum(self.I) / max(1, total_pop)

        behavior_feedback = -0.08 * inf_density if inf_density > 0.01 else 0.0
        mean_reversion = 0.02 * (0.10 - prev)

        new_beta = (
            prev
            + daily_noise
            + weekly * 0.02
            + self.monthly_trend
            + jump
            + external
            + behavior_feedback
            + mean_reversion
        )

        self.current_beta = float(np.clip(new_beta, 0.01, 0.45))
        self.beta_history.append(self.current_beta)

    def _update_seasonal_effects(self):
        self.time_of_year = (self.time_of_year + 1) % 365

        season = 1.0 + self.seasonal_amplitude * math.sin(
            2 * math.pi * self.time_of_year / self.seasonal_period
        )

        base_temp = 15.0 + 10.0 * math.sin(
            2 * math.pi * (self.time_of_year - 105) / 365.0
        )

        self.current_temperature = float(
            np.clip(base_temp + self.rng.normal(0.0, 2.0), -5.0, 35.0)
        )

        temp_factor = math.exp(-abs(self.current_temperature - 20.0) / 14.0)

        self.current_humidity = float(np.clip(0.6 + self.rng.normal(0.0, 0.08), 0.3, 0.9))
        hum_factor = math.exp(-abs(self.current_humidity - 0.55) / 0.35)

        self.effective_beta = float(
            np.clip(self.current_beta * season * temp_factor * hum_factor, 0.005, 0.50)
        )

    def _update_behavior(self):
        total_pop = np.sum(self.grid_population)
        inf_density = np.sum(self.I) / max(1, total_pop)

        if inf_density > self.awareness_threshold:
            self.behavioral_factor = 0.70
        else:
            self.behavioral_factor = 1.0

    def _local_beta_grid(self):
        noise = self.rng.normal(0.0, 0.004, size=(self.L, self.L))
        beta = self.effective_beta * self.beta_density_factor * self.behavioral_factor + noise
        beta = np.clip(beta, 0.005, 0.50)
        return beta.astype(np.float32)

    def _apply_movement(self):
        move_prob = MOVE_PROB * self.mobility_density_factor
        move_prob = np.clip(move_prob, 0.0, 0.08)

        for di, dj in self.neighbors:
            for arr in [self.S, self.E, self.I, self.R]:
                move = self.rng.binomial(arr, move_prob)
                arr -= move
                arr += np.roll(np.roll(move, -di, axis=0), -dj, axis=1)

    def _apply_long_distance_transmission(self):
        total_I = np.sum(self.I)
        if total_I <= 0:
            return

        q = self.I / np.maximum(self.grid_population, 1)

        # 1. 枢纽传播
        for hub in self.transport_hubs:
            hi, hj = hub

            if self.I[hi, hj] <= 0:
                continue

            hub_q = self.I[hi, hj] / max(1, self.grid_population[hi, hj])
            conn = self.hub_connectivity[hub]

            prob = self.effective_beta * AIR_TRAVEL_STRENGTH * conn * hub_q
            prob = np.clip(prob, 0.0, 0.15)

            new_e = self.rng.binomial(self.S, prob)

            self.S -= new_e
            self.E += new_e

        # 2. 随机长距离事件
        if self.rng.random() < LONG_DISTANCE_EVENT_PROB:
            hot = np.argwhere(q > 0.03)

            if len(hot) > 0:
                self.rng.shuffle(hot)

                for source in hot[:3]:
                    si, sj = int(source[0]), int(source[1])
                    source_q = q[si, sj]

                    n_targets = min(40, self.L * self.L)
                    targets = self.rng.integers(0, self.L * self.L, size=n_targets)

                    for tidx in targets:
                        ti, tj = divmod(int(tidx), self.L)

                        if ti == si and tj == sj:
                            continue

                        dist = math.sqrt((si - ti) ** 2 + (sj - tj) ** 2)
                        strength = AIR_TRAVEL_STRENGTH * math.exp(-dist / (self.L / 3.0 + 1e-8))
                        prob = strength * source_q * self.effective_beta * self.beta_density_factor[ti, tj]
                        prob = min(float(prob), 0.15)

                        if prob > 0:
                            new_e = int(self.rng.binomial(self.S[ti, tj], prob))
                            if new_e > 0:
                                self.S[ti, tj] -= new_e
                                self.E[ti, tj] += new_e

        # 3. 超级传播点
        hot_points = np.argwhere(self.I > 5)

        for source in hot_points[:200]:
            if self.rng.random() > SUPER_SPREADER_PROB:
                continue

            si, sj = int(source[0]), int(source[1])
            ti = int(self.rng.integers(0, self.L))
            tj = int(self.rng.integers(0, self.L))

            if ti == si and tj == sj:
                continue

            prob = SUPER_SPREADER_STRENGTH * self.I[si, sj] / max(10.0, self.grid_population[si, sj])
            prob = min(float(prob), 0.25)

            new_e = int(self.rng.binomial(self.S[ti, tj], prob))
            if new_e > 0:
                self.S[ti, tj] -= new_e
                self.E[ti, tj] += new_e

    def step(self):
        """
        一个真实时间步只记录一次 history，只推进一次 time_step。
        """
        self._update_stochastic_beta()
        self._update_seasonal_effects()
        self._update_behavior()

        self._apply_movement()

        q = self.I / np.maximum(self.grid_population, 1)

        pressure = np.zeros((self.L, self.L), dtype=np.float32)
        for di, dj in self.neighbors:
            pressure += np.roll(np.roll(q, di, axis=0), dj, axis=1)

        local_beta = self._local_beta_grid()

        prob = 1.0 - np.exp(-local_beta * pressure)
        prob = np.clip(prob, 0.0, 0.80)

        new_e = self.rng.binomial(self.S, prob)
        new_i = self.rng.binomial(self.E, self.sigma)
        new_r = self.rng.binomial(self.I, self.gamma)

        self.S -= new_e
        self.E += new_e - new_i
        self.I += new_i - new_r
        self.R += new_r

        self._apply_long_distance_transmission()

        self._record_history()

        jsd = self.compute_patch_jsd()
        self.history["jsd"].append(jsd)

        self.time_step += 1

    # =========================
    # 斑块指标
    # =========================
    def compute_patch_metrics(self, threshold=0.005):
        """
        返回：
        patch_metrics: 10 维斑块指标
        patch_sizes: 每个有效斑块面积，用于 JSD
        """
        q = self.I / np.maximum(self.grid_population, 1)
        binary = (q > threshold).astype(np.int32)

        structure = generate_binary_structure(2, 2)
        labeled, num = label(binary, structure=structure)

        areas = []
        intensities = []
        perimeters = []

        for pid in range(1, num + 1):
            mask = labeled == pid
            area = int(np.sum(mask))

            if area < MIN_PATCH_SIZE:
                continue

            areas.append(area)
            intensities.append(float(np.mean(q[mask])))

            eroded = np.zeros_like(mask, dtype=bool)
            eroded[1:-1, 1:-1] = (
                mask[1:-1, 1:-1]
                & mask[:-2, 1:-1]
                & mask[2:, 1:-1]
                & mask[1:-1, :-2]
                & mask[1:-1, 2:]
            )

            boundary = mask & (~eroded)
            perimeters.append(float(np.sum(boundary)))

        if len(areas) == 0:
            return np.zeros(10, dtype=np.float32), np.array([], dtype=np.float32)

        areas = np.asarray(areas, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)
        perimeters = np.asarray(perimeters, dtype=np.float32)

        total_grid_area = float(self.L * self.L)
        total_patch_area = float(np.sum(areas))

        area_ratios = areas / total_grid_area

        largest_patch_ratio = float(np.max(areas) / (total_patch_area + 1e-8))
        patch_m2 = float(np.mean(area_ratios ** 2))
        num_patches = float(len(areas))
        mean_area = float(np.mean(area_ratios))
        area_cv = float(np.std(areas) / (np.mean(areas) + 1e-8))

        sorted_a = np.sort(areas)
        cumsum = np.cumsum(sorted_a)
        gini = float((len(areas) + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-8)) / len(areas))

        boundary_complexity = float(
            np.mean((perimeters ** 2) / (4.0 * np.pi * areas + 1e-8))
        )

        mean_intensity = float(np.mean(intensities))
        intensity_var = float(np.var(intensities))
        occupancy = float(total_patch_area / total_grid_area)

        metrics = np.array(
            [
                largest_patch_ratio,
                patch_m2,
                num_patches,
                mean_area,
                area_cv,
                gini,
                boundary_complexity,
                mean_intensity,
                intensity_var,
                occupancy,
            ],
            dtype=np.float32,
        )

        return metrics, areas.astype(np.float32)

    def compute_patch_jsd(self):
        sizes_hist = self.history.get("patch_sizes", [])

        if len(sizes_hist) < 2:
            return 0.0

        prev_sizes = np.asarray(sizes_hist[-2], dtype=np.float32)
        curr_sizes = np.asarray(sizes_hist[-1], dtype=np.float32)

        if len(prev_sizes) == 0 or len(curr_sizes) == 0:
            return 0.0

        all_sizes = np.concatenate([prev_sizes, curr_sizes])

        if len(all_sizes) < 2:
            return 0.0

        try:
            bins = np.histogram_bin_edges(all_sizes, bins="auto")

            if len(bins) < 3:
                return 0.0

            hist_prev, _ = np.histogram(prev_sizes, bins=bins, density=False)
            hist_curr, _ = np.histogram(curr_sizes, bins=bins, density=False)

            return jensen_shannon_divergence(hist_prev, hist_curr)

        except Exception:
            return 0.0

    def get_causal_features(self):
        """
        返回 19 维特征：
        10 个斑块指标 + 9 个动力学指标

        之后在数据生成函数中再拼接 JSD，最终为 20 维。
        """
        patch_metrics, patch_sizes = self.compute_patch_metrics()

        total_pop = np.sum(self.grid_population)
        density = float(np.sum(self.I) / max(1, total_pop))

        beta_cur = float(self.current_beta)

        if len(self.beta_history) >= 5:
            beta_trend = float((self.beta_history[-1] - self.beta_history[-5]) / 5.0)
        else:
            beta_trend = 0.0

        if len(self.beta_history) >= 10:
            beta_vol = float(np.std(self.beta_history[-10:]))
        else:
            beta_vol = 0.0

        ac1 = 0.0
        var = 0.0

        if len(self.history["I_total"]) >= 20:
            recent_I = np.asarray(self.history["I_total"][-20:], dtype=np.float64)
            ac1 = safe_corrcoef(recent_I[:-1], recent_I[1:])
            var = float(np.var(recent_I))

        if self.current_beta > 1e-8:
            season_factor = float(self.effective_beta / self.current_beta)
        else:
            season_factor = 1.0

        long_dist_idx = 0.0
        total_I = np.sum(self.I)

        if total_I > 0 and len(self.transport_hubs) > 0:
            hub_I = sum(float(self.I[i, j]) for i, j in self.transport_hubs)
            long_dist_idx = float(hub_I / (total_I + 1e-8))

        extra = np.array(
            [
                density,
                beta_cur,
                beta_trend,
                beta_vol,
                ac1,
                var,
                season_factor,
                float(self.behavioral_factor),
                long_dist_idx,
            ],
            dtype=np.float32,
        )

        features = np.concatenate([patch_metrics, extra], axis=0).astype(np.float32)

        return features, density, True, patch_sizes

    def _record_history(self):
        features, density, _, patch_sizes = self.get_causal_features()

        self.history["beta"].append(float(self.current_beta))
        self.history["I_total"].append(float(np.sum(self.I) / max(1, np.sum(self.grid_population))))
        self.history["density"].append(float(density))
        self.history["features"].append(features)
        self.history["patch_sizes"].append(patch_sizes)

    def get_infection_grid(self):
        """
        关键修改：
        不是 I / total_population，
        而是每个格点自己的感染率 I_i / N_i。
        """
        q = self.I.astype(np.float32) / np.maximum(self.grid_population, 1).astype(np.float32)
        return np.clip(q, 0.0, 1.0).astype(np.float32)


# =========================
# 数据生成函数
# =========================
def _generate_one_sample(args):
    (
        sample_idx,
        L,
        sim_steps,
        input_seq_len,
        horizon,
        seed,
    ) = args

    rng = np.random.default_rng(seed + sample_idx)

    beta_low = float(rng.uniform(0.035, 0.075))
    beta_high = float(rng.uniform(0.10, 0.20))

    if beta_high <= beta_low:
        beta_high = beta_low + 0.05

    beta_params = {
        "initial_range": (beta_low, beta_high),
        "daily_volatility": float(rng.uniform(0.008, 0.025)),
        "weekly_cycle_amp": float(rng.uniform(0.03, 0.12)),
        "jump_prob": float(rng.uniform(0.003, 0.020)),
    }

    model = PatchSEIR(
        L=L,
        sim_steps=sim_steps,
        beta_params=beta_params,
        seed=seed + sample_idx,
    )

    density_hist = []
    image_hist = []
    feature_hist = []
    jsd_hist = []

    for _ in range(sim_steps):
        model.step()

        density_hist.append(model.history["density"][-1])
        image_hist.append(model.get_infection_grid())
        feature_hist.append(model.history["features"][-1])

        if len(model.history["jsd"]) > 0:
            jsd_hist.append(model.history["jsd"][-1])
        else:
            jsd_hist.append(0.0)

    density_hist = np.asarray(density_hist, dtype=np.float32)
    image_hist = np.asarray(image_hist, dtype=np.float32)
    feature_hist = np.asarray(feature_hist, dtype=np.float32)
    jsd_hist = np.asarray(jsd_hist, dtype=np.float32)

    critical_point = detect_critical_point(density_hist)

    if critical_point is None:
        return None

    if critical_point <= input_seq_len + 5:
        return None

    sample_imgs = []
    sample_patch_seq = []
    sample_remaining = []
    sample_risk = []
    sample_sim_id = []

    for t in range(input_seq_len, critical_point):
        img_seq = image_hist[t - input_seq_len:t]
        img_seq = img_seq[:, np.newaxis, :, :]  # (T, 1, L, L)

        patch_seq = feature_hist[t - input_seq_len:t]  # (T, 19)
        jsd_seq = jsd_hist[t - input_seq_len:t].reshape(-1, 1)  # (T, 1)

        full_patch_seq = np.concatenate([patch_seq, jsd_seq], axis=-1)  # (T, 20)

        remaining = float(critical_point - t)
        risk = 1.0 if remaining <= horizon else 0.0

        sample_imgs.append(img_seq.astype(np.float32))
        sample_patch_seq.append(full_patch_seq.astype(np.float32))
        sample_remaining.append(remaining)
        sample_risk.append(risk)
        sample_sim_id.append(sample_idx)

    if len(sample_remaining) == 0:
        return None

    return (
        np.asarray(sample_imgs, dtype=np.float32),
        np.asarray(sample_patch_seq, dtype=np.float32),
        np.asarray(sample_remaining, dtype=np.float32),
        np.asarray(sample_risk, dtype=np.float32),
        np.asarray(sample_sim_id, dtype=np.int64),
    )


def generate_patch_dataset(
    num_sims=200,
    L=64,
    sim_steps=200,
    input_seq_len=10,
    horizon=15,
    seed=DEFAULT_SEED,
    num_workers=0,
):
    """
    返回：
    X_img:       (N, T, 1, L, L)
    X_patch:     (N, T, 20)
    y_remaining: (N,)
    y_risk:      (N,)
    sim_id:      (N,)
    """
    tasks = [
        (
            i,
            L,
            sim_steps,
            input_seq_len,
            horizon,
            seed,
        )
        for i in range(num_sims)
    ]

    print(f"\n===== Generating dataset: {num_sims} simulations =====")
    print(f"L={L}, sim_steps={sim_steps}, input_seq_len={input_seq_len}, horizon={horizon}")

    if num_workers is not None and num_workers > 0:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_generate_one_sample, tasks), total=len(tasks)))
    else:
        results = []
        for task in tqdm(tasks):
            results.append(_generate_one_sample(task))

    results = [r for r in results if r is not None]

    if len(results) == 0:
        raise RuntimeError("No valid simulations generated. Try increasing beta range or num_sims.")

    X_img, X_patch, y_remaining, y_risk, sim_id = zip(*results)

    X_img = np.concatenate(X_img, axis=0)
    X_patch = np.concatenate(X_patch, axis=0)
    y_remaining = np.concatenate(y_remaining, axis=0)
    y_risk = np.concatenate(y_risk, axis=0)
    sim_id = np.concatenate(sim_id, axis=0)

    print("\n===== Dataset finished =====")
    print(f"Valid simulations: {len(results)} / {num_sims}")
    print(f"X_img:       {X_img.shape}")
    print(f"X_patch:     {X_patch.shape}")
    print(f"y_remaining: {y_remaining.shape}")
    print(f"y_risk:      {y_risk.shape}")
    print(f"risk ratio:  {float(np.mean(y_risk)):.4f}")

    return X_img, X_patch, y_remaining, y_risk, sim_id