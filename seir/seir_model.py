# seir_model_old_ac1_label.py
# -*- coding: utf-8 -*-

"""
v1.2 label-fix version

核心变化：
1. 不再用 AC1 / variance / trend 定义主标签 critical_point。
2. 新增可观测宏观转变事件 t_event：
   - 默认使用 I_total_ratio >= theta_I 且 infected_area_ratio >= theta_A，并连续 k 步成立。
3. AC1 仍然保留为 Traditional EWS / dynamics feature。
4. 数据集返回新增：
   - time_idx
   - critical_time
   - infected_area
   - dominant_patch
5. 后续 strict first-alarm lead time 分析可以基于 sim_id、time_idx、critical_time 完成。
"""

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
# 基础工具函数
# =========================
def safe_corrcoef(x, y):
    """
    安全计算 Pearson 相关系数。

    用途：
    - 计算 AC1；
    - 后续可以用于局部同步指标；
    - 避免标准差为 0 或 NaN 导致程序崩溃。
    """
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
    """
    Jensen-Shannon divergence.

    用途：
    - 衡量相邻时间步斑块面积分布变化；
    - 作为 dynamic patch / structural change 的一个辅助指标。
    """
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


def compute_csd_series(density_history, window=20):
    """
    计算传统临界慢化序列：AC1、variance、trend。

    注意：
    - 该函数只作为 EWS 特征或可视化诊断；
    - 不再用于定义 ground-truth critical_time。
    """
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


# =========================
# 旧 AC1 标签函数：保留但不作为主标签
# =========================
def detect_critical_point(
    density_history,
    window=20,
    threshold_ac1=0.80,
    min_peak_density=1e-4,
    min_rel_density=0.08,
):
    """
    旧版 AC1-based warning point 检测函数。

    重要说明：
    - 该函数不再用于主标签 critical_time；
    - 只建议用于 preliminary / sensitivity / diagnostic analysis；
    - 最终论文主标签应使用 detect_observable_event_time()。
    """
    density_history = np.asarray(density_history, dtype=np.float64)
    n = len(density_history)

    if n < 2 * window + 5:
        return None

    max_density = float(np.max(density_history))

    if max_density < min_peak_density:
        return None

    ac1_vals = np.zeros(n, dtype=np.float64)
    var_vals = np.zeros(n, dtype=np.float64)
    trend_vals = np.zeros(n, dtype=np.float64)

    for t in range(window, n):
        w = density_history[t - window:t]

        if len(w) < 3:
            continue

        ac1_vals[t] = safe_corrcoef(w[:-1], w[1:])
        var_vals[t] = np.var(w)

        x = np.arange(len(w))
        if np.std(w) > 1e-12:
            trend_vals[t] = np.polyfit(x, w, 1)[0]
        else:
            trend_vals[t] = 0.0

    ac1_smooth = gaussian_filter1d(ac1_vals, sigma=2.0)
    var_smooth = gaussian_filter1d(var_vals, sigma=2.0)
    trend_smooth = gaussian_filter1d(trend_vals, sigma=2.0)

    density_threshold = max(min_peak_density, min_rel_density * max_density)

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

    candidates = np.where(valid_mask & (ac1_smooth >= threshold_ac1))[0]

    if len(candidates) > 0:
        return int(candidates[0])

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
# v1.2 新标签函数：基于可观测宏观转变事件
# =========================
def first_persistent_time(condition, k=3):
    """
    找到布尔序列中第一次连续 k 步为 True 的起点。

    用途：
    - 避免偶然噪声导致单个时间点误判为 transition；
    - 用于 t_event、t_R、visible_patch_time 等事件时间检测。
    """
    condition = np.asarray(condition, dtype=bool)

    if len(condition) < k:
        return None

    for t in range(0, len(condition) - k + 1):
        if np.all(condition[t:t + k]):
            return int(t)

    return None


def detect_observable_event_time(
    density_hist,
    infected_area_hist,
    dominant_patch_hist,
    theta_I=0.05,
    theta_A=0.05,
    theta_L=0.10,
    k=3,
    mode="infection_area",
):
    """
    基于可观测宏观转变事件定义 transition time。

    参数：
    - density_hist:
        I_total / total_population，整体感染比例。
    - infected_area_hist:
        被感染斑块占总网格面积比例，这里来自 patch_metrics 的 occupancy。
    - dominant_patch_hist:
        最大斑块在所有感染斑块面积中的占比。
    - theta_I:
        整体感染比例阈值。
    - theta_A:
        感染面积比例阈值。
    - theta_L:
        主导斑块占比阈值。
    - k:
        连续 k 步满足条件才认为事件发生。
    - mode:
        "infection_area":
            density_hist >= theta_I 且 infected_area_hist >= theta_A
            推荐作为主标签。
        "area_patch":
            infected_area_hist >= theta_A 且 dominant_patch_hist >= theta_L
            更强调斑块形成，但可能更依赖斑块阈值。
        "infection_only":
            只使用 density_hist >= theta_I
            作为快速调试或敏感性分析，不建议作为最终主标签。

    返回：
    - t_event: int 或 None
    """
    density_hist = np.asarray(density_hist, dtype=np.float64)
    infected_area_hist = np.asarray(infected_area_hist, dtype=np.float64)
    dominant_patch_hist = np.asarray(dominant_patch_hist, dtype=np.float64)

    if mode == "infection_area":
        condition = (
            (density_hist >= theta_I)
            & (infected_area_hist >= theta_A)
        )

    elif mode == "area_patch":
        condition = (
            (infected_area_hist >= theta_A)
            & (dominant_patch_hist >= theta_L)
        )

    elif mode == "infection_only":
        condition = density_hist >= theta_I

    else:
        raise ValueError(f"Unknown event mode: {mode}")

    return first_persistent_time(condition, k=k)


def make_horizon_label(time_idx, critical_time, horizon):
    """
    根据 critical_time 构造 horizon-based early-warning label。

    y = 1 当且仅当：
        0 < critical_time - time_idx <= horizon

    含义：
    - 当前样本仍在事件发生之前；
    - 并且事件将在未来 horizon 步内发生。
    """
    remaining = float(critical_time - time_idx)
    risk = 1.0 if 0.0 < remaining <= float(horizon) else 0.0
    return remaining, risk


def compute_reff(beta_t, susceptible_ratio, gamma):
    """
    近似有效再生数 Reff(t)。

    用途：
    - 当前版本主要作为 metadata 或后续理论辅助标签；
    - 不建议第一轮直接用它替代 t_event 作为主标签；
    - 后续可以扩展为空间谱半径版 Reff_spatial。
    """
    return float(beta_t * susceptible_ratio / max(gamma, 1e-12))


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

    v1.2 修改：
    - history 中额外保存 S_total, E_total, R_total, effective_beta, Reff；
    - history 中保存 infected_area 和 dominant_patch，方便打新标签。
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
            "effective_beta": [],
            "S_total": [],
            "E_total": [],
            "I_total": [],
            "R_total": [],
            "Reff": [],
            "density": [],
            "features": [],
            "patch_sizes": [],
            "jsd": [],
            "infected_area": [],
            "dominant_patch": [],
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
                        strength = AIR_TRAVEL_STRENGTH * math.exp(dist * -1.0 / (self.L / 3.0 + 1e-8))
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

        10 维含义：
        0. dominant_patch_ratio:
           最大斑块面积 / 总感染斑块面积
        1. patch_m2:
           斑块面积二阶矩
        2. num_patches:
           有效斑块数量
        3. mean_area:
           平均斑块面积 / 总网格面积
        4. area_cv:
           斑块面积变异系数
        5. gini:
           斑块面积 Gini
        6. boundary_complexity:
           平均边界复杂度
        7. mean_intensity:
           斑块内平均感染强度
        8. intensity_var:
           斑块间感染强度方差
        9. occupancy:
           总感染斑块面积 / 总网格面积
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

        dominant_patch_ratio = float(np.max(areas) / (total_patch_area + 1e-8))
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
                dominant_patch_ratio,
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

        注意：
        - 这里的 AC1 只是 feature；
        - 不能再用 AC1 定义 critical_time。
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

        total_pop = float(np.sum(self.grid_population))
        s_total = float(np.sum(self.S) / max(1.0, total_pop))
        e_total = float(np.sum(self.E) / max(1.0, total_pop))
        i_total = float(np.sum(self.I) / max(1.0, total_pop))
        r_total = float(np.sum(self.R) / max(1.0, total_pop))

        reff = compute_reff(
            beta_t=float(self.effective_beta),
            susceptible_ratio=s_total,
            gamma=float(self.gamma),
        )

        # features[0] = dominant_patch_ratio
        # features[9] = occupancy / infected_area_ratio
        dominant_patch = float(features[0])
        infected_area = float(features[9])

        self.history["beta"].append(float(self.current_beta))
        self.history["effective_beta"].append(float(self.effective_beta))
        self.history["S_total"].append(s_total)
        self.history["E_total"].append(e_total)
        self.history["I_total"].append(i_total)
        self.history["R_total"].append(r_total)
        self.history["Reff"].append(reff)

        self.history["density"].append(float(density))
        self.history["features"].append(features)
        self.history["patch_sizes"].append(patch_sizes)

        self.history["infected_area"].append(infected_area)
        self.history["dominant_patch"].append(dominant_patch)

    def get_infection_grid(self):
        """
        返回每个格点自己的感染率 I_i / N_i。
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
        event_mode,
        theta_I,
        theta_A,
        theta_L,
        persistent_k,
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

    infected_area_hist = []
    dominant_patch_hist = []
    reff_hist = []

    for _ in range(sim_steps):
        model.step()

        density_hist.append(model.history["density"][-1])
        image_hist.append(model.get_infection_grid())
        feature_hist.append(model.history["features"][-1])

        infected_area_hist.append(model.history["infected_area"][-1])
        dominant_patch_hist.append(model.history["dominant_patch"][-1])
        reff_hist.append(model.history["Reff"][-1])

        if len(model.history["jsd"]) > 0:
            jsd_hist.append(model.history["jsd"][-1])
        else:
            jsd_hist.append(0.0)

    density_hist = np.asarray(density_hist, dtype=np.float32)
    image_hist = np.asarray(image_hist, dtype=np.float32)
    feature_hist = np.asarray(feature_hist, dtype=np.float32)
    jsd_hist = np.asarray(jsd_hist, dtype=np.float32)

    infected_area_hist = np.asarray(infected_area_hist, dtype=np.float32)
    dominant_patch_hist = np.asarray(dominant_patch_hist, dtype=np.float32)
    reff_hist = np.asarray(reff_hist, dtype=np.float32)

    # =========================
    # v1.2 关键修改：
    # 用可观测宏观事件 t_event 作为 critical_point
    # 而不是 detect_critical_point(density_hist)
    # =========================
    t_event = detect_observable_event_time(
        density_hist=density_hist,
        infected_area_hist=infected_area_hist,
        dominant_patch_hist=dominant_patch_hist,
        theta_I=theta_I,
        theta_A=theta_A,
        theta_L=theta_L,
        k=persistent_k,
        mode=event_mode,
    )

    if t_event is None:
        return None

    critical_point = int(t_event)

    if critical_point <= input_seq_len + 5:
        return None

    sample_imgs = []
    sample_patch_seq = []
    sample_remaining = []
    sample_risk = []
    sample_sim_id = []

    sample_time_idx = []
    sample_critical_time = []
    sample_infected_area = []
    sample_dominant_patch = []
    sample_reff = []

    for t in range(input_seq_len, critical_point):
        img_seq = image_hist[t - input_seq_len:t]
        img_seq = img_seq[:, np.newaxis, :, :]  # (T, 1, L, L)

        patch_seq = feature_hist[t - input_seq_len:t]  # (T, 19)
        jsd_seq = jsd_hist[t - input_seq_len:t].reshape(-1, 1)  # (T, 1)

        full_patch_seq = np.concatenate([patch_seq, jsd_seq], axis=-1)  # (T, 20)

        remaining, risk = make_horizon_label(
            time_idx=t,
            critical_time=critical_point,
            horizon=horizon,
        )

        sample_imgs.append(img_seq.astype(np.float32))
        sample_patch_seq.append(full_patch_seq.astype(np.float32))
        sample_remaining.append(remaining)
        sample_risk.append(risk)
        sample_sim_id.append(sample_idx)

        sample_time_idx.append(t)
        sample_critical_time.append(critical_point)
        sample_infected_area.append(float(infected_area_hist[t]))
        sample_dominant_patch.append(float(dominant_patch_hist[t]))
        sample_reff.append(float(reff_hist[t]))

    if len(sample_remaining) == 0:
        return None

    return (
        np.asarray(sample_imgs, dtype=np.float32),
        np.asarray(sample_patch_seq, dtype=np.float32),
        np.asarray(sample_remaining, dtype=np.float32),
        np.asarray(sample_risk, dtype=np.float32),
        np.asarray(sample_sim_id, dtype=np.int64),
        np.asarray(sample_time_idx, dtype=np.int64),
        np.asarray(sample_critical_time, dtype=np.int64),
        np.asarray(sample_infected_area, dtype=np.float32),
        np.asarray(sample_dominant_patch, dtype=np.float32),
        np.asarray(sample_reff, dtype=np.float32),
    )


def generate_patch_dataset(
    num_sims=200,
    L=64,
    sim_steps=200,
    input_seq_len=10,
    horizon=15,
    seed=DEFAULT_SEED,
    num_workers=0,
    event_mode="infection_area",
    theta_I=0.05,
    theta_A=0.05,
    theta_L=0.10,
    persistent_k=3,
):
    """
    生成用于训练的 SEIR 斑块预警数据集。

    返回：
    - X_img:          (N, T, 1, L, L)
    - X_patch:        (N, T, 20)
    - y_remaining:    (N,)
                      当前时间点距离可观测宏观转变事件 t_event 的剩余时间。
    - y_risk:         (N,)
                      horizon-based early-warning label。
                      当 0 < t_event - t <= horizon 时为 1。
    - sim_id:         (N,)
                      样本来自哪一条模拟轨迹。
    - time_idx:       (N,)
                      样本对应的模拟时间点。
    - critical_time:  (N,)
                      可观测宏观转变时间 t_event。
    - infected_area:  (N,)
                      当前时刻感染斑块面积比例。
    - dominant_patch: (N,)
                      当前时刻最大斑块占所有感染斑块面积的比例。
    - reff:           (N,)
                      近似 Reff(t)，用于后续辅助分析。

    重要：
    - AC1 不再用于定义 critical_time；
    - AC1 仍然包含在 X_patch 的动力学特征中。
    """
    tasks = [
        (
            i,
            L,
            sim_steps,
            input_seq_len,
            horizon,
            seed,
            event_mode,
            theta_I,
            theta_A,
            theta_L,
            persistent_k,
        )
        for i in range(num_sims)
    ]

    print(f"\n===== Generating dataset: {num_sims} simulations =====")
    print(f"L={L}, sim_steps={sim_steps}, input_seq_len={input_seq_len}, horizon={horizon}")
    print(
        "Event label: "
        f"mode={event_mode}, theta_I={theta_I}, theta_A={theta_A}, "
        f"theta_L={theta_L}, persistent_k={persistent_k}"
    )

    if num_workers is not None and num_workers > 0:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_generate_one_sample, tasks), total=len(tasks)))
    else:
        results = []
        for task in tqdm(tasks):
            results.append(_generate_one_sample(task))

    results = [r for r in results if r is not None]

    if len(results) == 0:
        raise RuntimeError(
            "No valid simulations generated. "
            "Try increasing beta range, sim_steps, num_sims, or lowering event thresholds."
        )

    (
        X_img,
        X_patch,
        y_remaining,
        y_risk,
        sim_id,
        time_idx,
        critical_time,
        infected_area,
        dominant_patch,
        reff,
    ) = zip(*results)

    X_img = np.concatenate(X_img, axis=0)
    X_patch = np.concatenate(X_patch, axis=0)
    y_remaining = np.concatenate(y_remaining, axis=0)
    y_risk = np.concatenate(y_risk, axis=0)
    sim_id = np.concatenate(sim_id, axis=0)

    time_idx = np.concatenate(time_idx, axis=0)
    critical_time = np.concatenate(critical_time, axis=0)
    infected_area = np.concatenate(infected_area, axis=0)
    dominant_patch = np.concatenate(dominant_patch, axis=0)
    reff = np.concatenate(reff, axis=0)

    print("\n===== Dataset finished =====")
    print(f"Valid simulations: {len(results)} / {num_sims}")
    print(f"X_img:          {X_img.shape}")
    print(f"X_patch:        {X_patch.shape}")
    print(f"y_remaining:    {y_remaining.shape}")
    print(f"y_risk:         {y_risk.shape}")
    print(f"sim_id:         {sim_id.shape}")
    print(f"time_idx:       {time_idx.shape}")
    print(f"critical_time:  {critical_time.shape}")
    print(f"risk ratio:     {float(np.mean(y_risk)):.4f}")
    print(f"mean critical_time: {float(np.mean(critical_time)):.2f}")

    return (
        X_img,
        X_patch,
        y_remaining,
        y_risk,
        sim_id,
        time_idx,
        critical_time,
        infected_area,
        dominant_patch,
        reff,
    )
