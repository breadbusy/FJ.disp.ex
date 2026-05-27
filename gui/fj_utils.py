"""FJ 频散分析工具函数模块。

提供 FJ 方法的核心算法函数：
  - trace_fj_ridge: 从起点向左右追踪频散谱的波峰
  - auto_extract_fj_modes: 自动提取多模态频散曲线
  - aggregate_fj_dispersion: 对多个子阵列的多模态频散结果做统计聚合
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def trace_fj_ridge(
    energy: np.ndarray,
    velocities: np.ndarray,
    start_vel_idx: int,
    start_freq_idx: int,
    max_vel_jump_ratio: float = 1.5,
    search_window_ms: float = 200.0,
) -> np.ndarray:
    """从起点向左右追踪 FJ 频散谱的波峰。

    参数
    ----------
    energy: (n_vels, n_freqs) 频散能量谱
    velocities: (n_vels,) 速度轴 (m/s)
    start_vel_idx: 起始速度索引
    start_freq_idx: 起始频率索引
    max_vel_jump_ratio: 相邻频率最大速度跳变倍数
    search_window_ms: 每列搜索窗口半径 (m/s)，与速度分辨率无关

    返回
    -------
    tracked_indices: (n_freqs,) 每个频率对应的速度索引，-1 表示未找到
    """
    n_freqs = energy.shape[1]
    tracked = np.full(n_freqs, -1, dtype=int)
    tracked[start_freq_idx] = start_vel_idx
    n_vels = energy.shape[0]

    # 将速度窗口 (m/s) 换算为索引步数，至少搜索 ±3 个点
    vel_spacing = np.median(np.diff(velocities))
    search_steps = max(3, int(np.ceil(search_window_ms / vel_spacing)))

    # 向左追踪
    prev = start_vel_idx
    for fi in range(start_freq_idx - 1, -1, -1):
        lo = max(0, prev - search_steps)
        hi = min(n_vels, prev + search_steps + 1)
        col = energy[lo:hi, fi]
        if len(col) == 0:
            tracked[fi] = prev
            continue
        peaks, _ = find_peaks(col)
        if len(peaks) == 0:
            tracked[fi] = prev
            continue
        best = lo + peaks[np.argmax(col[peaks])]
        v_ratio = (
            max(velocities[best], velocities[prev])
            / (min(velocities[best], velocities[prev]) + 1e-10)
        )
        tracked[fi] = best if v_ratio <= max_vel_jump_ratio else prev
        prev = tracked[fi]

    # 向右追踪
    prev = start_vel_idx
    for fi in range(start_freq_idx + 1, n_freqs):
        lo = max(0, prev - search_steps)
        hi = min(n_vels, prev + search_steps + 1)
        col = energy[lo:hi, fi]
        if len(col) == 0:
            tracked[fi] = prev
            continue
        peaks, _ = find_peaks(col)
        if len(peaks) == 0:
            tracked[fi] = prev
            continue
        best = lo + peaks[np.argmax(col[peaks])]
        v_ratio = (
            max(velocities[best], velocities[prev])
            / (min(velocities[best], velocities[prev]) + 1e-10)
        )
        tracked[fi] = best if v_ratio <= max_vel_jump_ratio else prev
        prev = tracked[fi]

    return tracked


def auto_extract_fj_modes(
    energy: np.ndarray,
    frequencies: np.ndarray,
    velocities: np.ndarray,
    min_snr: float = 2.5,
    max_vel_jump: float = 1.5,
    n_modes: int = 3,
    min_continuous: int = 4,
) -> List[Dict]:
    """从 FJ 频散谱自动提取多模态频散曲线。

    每个频率切片找多个峰值，按连续性约束分组为不同模态。

    参数
    ----------
    energy: (n_vels, n_freqs) 频散能量谱
    frequencies: (n_freqs,) 频率轴 (Hz)
    velocities: (n_vels,) 速度轴 (m/s)
    min_snr: 信噪比阈值，峰值/均值 < min_snr 的忽略
    max_vel_jump: 相邻频率最大速度跳变倍数
    n_modes: 期望提取的模态数
    min_continuous: 最小连续点数，少于该值的模态段丢弃

    返回
    -------
    modes: 多模态列表，每项包含 frequencies, velocities, quality, score
    """
    n_freqs, n_vels = energy.shape  # energy: (n_freqs, n_vels)

    # 每列的能量均值和信噪比
    col_mean = np.mean(energy, axis=1)
    col_mean[col_mean == 0] = 1e-10

    # 每列找峰
    all_peaks: List[List[int]] = []
    for fi in range(n_freqs):
        col = energy[fi, :]
        snr_col = col / col_mean[fi]
        peaks, props = find_peaks(col, height=col_mean[fi] * min_snr)
        if len(peaks) > 0:
            # 按峰值排序取前 n_modes*2 个
            heights = props["peak_heights"]
            sorted_idx = np.argsort(heights)[::-1][: n_modes * 2]
            peaks = peaks[sorted_idx]
        all_peaks.append(list(peaks))

    # 为每个可能模态追踪
    modes: List[Dict] = []
    used: List[bool] = [False] * n_modes

    for mode_idx in range(n_modes):
        # 找未使用的最佳起始点
        best_fi = -1
        best_pi = -1
        best_val = 0
        for fi in range(n_freqs):
            for pi in all_peaks[fi]:
                # 检查该峰是否接近已有模态
                already_used = False
                for mm in modes:
                    if fi < len(mm.get("indices", [])) and mm["indices"][fi] == pi:
                        already_used = True
                        break
                if already_used:
                    continue
                val = energy[fi, pi]
                if val > best_val:
                    best_val = val
                    best_fi = fi
                    best_pi = pi

        if best_fi < 0:
            break

        # 双向追踪
        indices = np.full(n_freqs, -1, dtype=int)
        indices[best_fi] = best_pi

        # 向左
        prev = best_pi
        for fi in range(best_fi - 1, -1, -1):
            candidate_peaks = all_peaks[fi]
            if not candidate_peaks:
                indices[fi] = prev
                continue
            best_peak = candidate_peaks[0]
            min_dist = abs(velocities[prev] - velocities[best_peak])
            for p in candidate_peaks[1:]:
                dist = abs(velocities[prev] - velocities[p])
                if dist < min_dist:
                    min_dist = dist
                    best_peak = p
            v_ratio = (
                max(velocities[best_peak], velocities[prev])
                / (min(velocities[best_peak], velocities[prev]) + 1e-10)
            )
            indices[fi] = best_peak if v_ratio <= max_vel_jump else prev
            prev = indices[fi]

        # 向右
        prev = best_pi
        for fi in range(best_fi + 1, n_freqs):
            candidate_peaks = all_peaks[fi]
            if not candidate_peaks:
                indices[fi] = prev
                continue
            best_peak = candidate_peaks[0]
            min_dist = abs(velocities[prev] - velocities[best_peak])
            for p in candidate_peaks[1:]:
                dist = abs(velocities[prev] - velocities[p])
                if dist < min_dist:
                    min_dist = dist
                    best_peak = p
            v_ratio = (
                max(velocities[best_peak], velocities[prev])
                / (min(velocities[best_peak], velocities[prev]) + 1e-10)
            )
            indices[fi] = best_peak if v_ratio <= max_vel_jump else prev
            prev = indices[fi]

        # 连续性过滤
        valid_mask = indices >= 0
        if np.sum(valid_mask) < min_continuous:
            continue

        # 计算 quality
        quality = np.zeros(n_freqs)
        for fi in range(n_freqs):
            if indices[fi] >= 0:
                peak_val = energy[fi, indices[fi]]
                quality[fi] = min(peak_val / col_mean[fi] / min_snr, 1.0)

        mask = indices >= 0
        mode_freqs = frequencies[mask]
        mode_vels = velocities[indices[mask]]
        mode_quality = quality[mask]

        modes.append({
            "frequencies": mode_freqs,
            "velocities": mode_vels,
            "quality": mode_quality,
            "score": float(np.mean(mode_quality)) * len(mode_freqs),
            "mode_index": mode_idx,
            "indices": indices,
        })

    return modes


def aggregate_fj_dispersion(
    batch_results: List[Dict],
    n_modes: int = 3,
) -> Dict:
    """对多个子阵列的多模态频散结果做统计聚合。

    参数
    ----------
    batch_results: 批处理结果列表，每项包含 modes 字典
    n_modes: 最大模态数

    返回
    -------
    stats: 包含每个模态的均值、标准差、变异系数等
    """
    mode_stats: Dict[int, Dict] = {}

    for mi in range(n_modes):
        all_freqs = []
        # 收集所有子阵列该模态的频率和速度
        freq_vel_pairs = []
        for r in batch_results:
            modes = r.get("modes", {})
            mode_str = str(mi)
            if mode_str in modes:
                m = modes[mode_str]
                if len(m.get("frequencies", [])) > 0:
                    for f, v in zip(m["frequencies"], m["velocities"]):
                        freq_vel_pairs.append((f, v))

        if len(freq_vel_pairs) == 0:
            mode_stats[mi] = {
                "mean_frequencies": np.array([]),
                "mean_velocities": np.array([]),
                "std_velocities": np.array([]),
                "cov": np.array([]),
                "count": 0,
                "individual": [],
            }
            continue

        # 按频率分组计算统计量
        pairs = sorted(freq_vel_pairs, key=lambda x: x[0])
        binned: Dict[float, List[float]] = {}
        for f, v in pairs:
            key = round(f, 5)  # 按频率值分组
            if key not in binned:
                binned[key] = []
            binned[key].append(v)

        freqs = []
        mean_vels = []
        std_vels = []
        cov_vals = []
        for f in sorted(binned.keys()):
            vals = np.array(binned[f])
            if len(vals) >= 2:
                freqs.append(f)
                mean_vels.append(np.mean(vals))
                std_vels.append(np.std(vals))
                cov_vals.append(np.std(vals) / (np.mean(vals) + 1e-10))

        # 收集个体结果
        individuals = []
        for r in batch_results:
            modes = r.get("modes", {})
            mode_str = str(mi)
            if mode_str in modes:
                m = modes[mode_str]
                individuals.append({
                    "subarray_id": r.get("subarray_id", -1),
                    "frequencies": m.get("frequencies", []),
                    "velocities": m.get("velocities", []),
                })

        mode_stats[mi] = {
            "mean_frequencies": np.array(freqs),
            "mean_velocities": np.array(mean_vels),
            "std_velocities": np.array(std_vels),
            "cov": np.array(cov_vals),
            "count": len(individuals),
            "individual": individuals,
        }

    return mode_stats
