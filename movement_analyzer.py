from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_ACTIONS = {"squat", "throw", "general"}
ACTION_LABELS = {
    "squat": "深蹲",
    "throw": "投掷（实验性）",
    "general": "通用动作记录",
}


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _series_range(values: Sequence[float]) -> Optional[float]:
    valid = _finite(values)
    if len(valid) < 2:
        return None
    return float(np.percentile(valid, 95) - np.percentile(valid, 5))


def _row_mean(series: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2:
        return np.array([], dtype=np.float64)
    counts = np.sum(np.isfinite(array), axis=1)
    totals = np.nansum(array, axis=1)
    return np.divide(
        totals,
        counts,
        out=np.full(len(array), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _weighted_score(values: Dict[str, int], weights: Dict[str, float]) -> int:
    available = [(name, score) for name, score in values.items() if name in weights]
    total_weight = sum(weights[name] for name, _ in available)
    if total_weight <= 0:
        return 50
    return int(round(sum(score * weights[name] for name, score in available) / total_weight))


def _trunk_control_score(angle_series: Dict[str, List[float]]) -> Tuple[int, Optional[float]]:
    trunk = _finite(angle_series.get("trunk_flexion", []))
    if len(trunk) < 3:
        return 50, None
    upper = float(np.percentile(np.abs(trunk), 90))
    if upper <= 20:
        return 100, upper
    return int(round(np.clip(100 - (upper - 20) * 2.0, 40, 100))), upper


def _smoothness_score(values: np.ndarray, timestamps: np.ndarray, repetitions: int) -> int:
    valid = np.isfinite(values) & np.isfinite(timestamps)
    values = values[valid]
    timestamps = timestamps[valid]
    if len(values) < 5:
        return 50
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        return 50
    velocity = np.diff(values) / dt
    deadband = max(5.0, float(np.percentile(np.abs(velocity), 25)))
    signs = np.sign(velocity)
    signs[np.abs(velocity) < deadband] = 0
    signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    expected = max(1, repetitions * 2 - 1)
    extra_reversals = max(0, reversals - expected)
    return int(np.clip(100 - extra_reversals * 12, 40, 100))


def _detect_squat_repetitions(
    knee_angle: np.ndarray,
    timestamps: np.ndarray,
) -> List[Dict]:
    valid = np.isfinite(knee_angle)
    if np.sum(valid) < 5:
        return []
    standing_reference = float(np.percentile(knee_angle[valid], 85))
    threshold = standing_reference - 25.0
    below = valid & (knee_angle <= threshold)

    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, is_below in enumerate(below):
        if is_below and start is None:
            start = index
        elif not is_below and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(below) - 1))

    repetitions: List[Dict] = []
    for segment_start, segment_end in segments:
        if timestamps[segment_end] - timestamps[segment_start] < 0.3:
            continue
        segment = knee_angle[segment_start:segment_end + 1]
        if not np.isfinite(segment).any():
            continue
        bottom = segment_start + int(np.nanargmin(segment))
        repetitions.append({
            "start_time_sec": round(float(timestamps[segment_start]), 3),
            "bottom_time_sec": round(float(timestamps[bottom]), 3),
            "end_time_sec": round(float(timestamps[segment_end]), 3),
            "duration_sec": round(float(timestamps[segment_end] - timestamps[segment_start]), 3),
            "bottom_knee_angle_deg": round(float(knee_angle[bottom]), 2),
        })
    return repetitions


def _tempo_score(repetitions: List[Dict]) -> Optional[int]:
    if not repetitions:
        return None
    balance_scores = []
    for repetition in repetitions:
        descent = repetition["bottom_time_sec"] - repetition["start_time_sec"]
        ascent = repetition["end_time_sec"] - repetition["bottom_time_sec"]
        total = descent + ascent
        if total > 0:
            balance = min(descent, ascent) / max(descent, ascent, 1e-3)
            balance_scores.append(60 + 40 * balance)
    return int(round(np.mean(balance_scores))) if balance_scores else None


def _consistency_score(repetitions: List[Dict]) -> Optional[int]:
    if len(repetitions) < 2:
        return None
    depths = np.array([r["bottom_knee_angle_deg"] for r in repetitions])
    durations = np.array([r["duration_sec"] for r in repetitions])
    depth_variation = float(np.std(depths))
    duration_cv = float(np.std(durations) / max(np.mean(durations), 1e-3))
    return int(round(np.clip(100 - depth_variation * 3 - duration_cv * 80, 30, 100)))


def _squat_assessment(
    angle_series: Dict[str, List[float]],
    timestamps: np.ndarray,
) -> Dict:
    left = np.asarray(angle_series.get("left_knee", []), dtype=np.float64)
    right = np.asarray(angle_series.get("right_knee", []), dtype=np.float64)
    if len(left) != len(timestamps):
        left = np.full(len(timestamps), np.nan)
    if len(right) != len(timestamps):
        right = np.full(len(timestamps), np.nan)
    knees = _row_mean(np.column_stack([left, right]))
    repetitions = _detect_squat_repetitions(knees, timestamps)

    left_range = _series_range(left)
    right_range = _series_range(right)
    available_ranges = [value for value in [left_range, right_range] if value is not None]
    average_range = float(np.mean(available_ranges)) if available_ranges else 0.0
    range_score = int(round(np.clip(average_range / 70.0 * 100, 30, 100))) if available_ranges else 50

    symmetry_score = 50
    if left_range is not None and right_range is not None:
        difference = abs(left_range - right_range)
        symmetry_score = int(round(np.clip(100 - difference * 2.0, 40, 100)))

    trunk_score, trunk_p90 = _trunk_control_score(angle_series)
    smoothness = _smoothness_score(knees, timestamps, len(repetitions))
    tempo = _tempo_score(repetitions)
    consistency = _consistency_score(repetitions)

    dimensions = {
        "下蹲活动范围": range_score,
        "左右协调": symmetry_score,
        "躯干控制": trunk_score,
        "动作平滑": smoothness,
    }
    if tempo is not None:
        dimensions["节奏控制"] = tempo
    if consistency is not None:
        dimensions["重复一致性"] = consistency

    weights = {
        "下蹲活动范围": 0.30,
        "左右协调": 0.15,
        "躯干控制": 0.20,
        "动作平滑": 0.15,
        "节奏控制": 0.10,
        "重复一致性": 0.10,
    }
    observations = []
    if not repetitions:
        observations.append("未识别到完整深蹲周期，节奏和重复一致性未计分。")
    if len(repetitions) == 1:
        observations.append("仅识别到一次动作，重复一致性至少需要两次动作。")

    return {
        "dimensions": dimensions,
        "overall_score": _weighted_score(dimensions, weights),
        "repetitions": repetitions,
        "observations": observations,
        "evidence": {
            "left_knee_range_deg": round(left_range, 2) if left_range is not None else None,
            "right_knee_range_deg": round(right_range, 2) if right_range is not None else None,
            "trunk_flexion_p90_deg": round(trunk_p90, 2) if trunk_p90 is not None else None,
            "repetition_count": len(repetitions),
        },
        "experimental": False,
    }


def _general_assessment(
    angle_series: Dict[str, List[float]],
    timestamps: np.ndarray,
) -> Dict:
    paired_ranges = []
    all_ranges = []
    for left_name, right_name in [
        ("left_knee", "right_knee"),
        ("left_hip", "right_hip"),
        ("left_elbow", "right_elbow"),
    ]:
        left_range = _series_range(angle_series.get(left_name, []))
        right_range = _series_range(angle_series.get(right_name, []))
        if left_range is not None:
            all_ranges.append(left_range)
        if right_range is not None:
            all_ranges.append(right_range)
        if left_range is not None and right_range is not None:
            paired_ranges.append(abs(left_range - right_range))

    movement_range = float(np.mean(all_ranges)) if all_ranges else 0.0
    movement_score = int(round(np.clip(movement_range / 60.0 * 100, 30, 100))) if all_ranges else 50
    coordination = int(round(np.clip(100 - np.mean(paired_ranges) * 2, 40, 100))) if paired_ranges else 50
    trunk_score, trunk_p90 = _trunk_control_score(angle_series)

    primary = np.asarray(angle_series.get("right_knee", []), dtype=np.float64)
    if len(primary) != len(timestamps):
        primary = np.asarray(angle_series.get("right_elbow", []), dtype=np.float64)
    smoothness = _smoothness_score(primary, timestamps, 0) if len(primary) == len(timestamps) else 50
    dimensions = {
        "动作幅度": movement_score,
        "左右协调": coordination,
        "躯干控制": trunk_score,
        "动作平滑": smoothness,
    }
    return {
        "dimensions": dimensions,
        "overall_score": int(round(np.mean(list(dimensions.values())))),
        "repetitions": [],
        "observations": ["通用模式只提供二维动作趋势，不用于专项技术或损伤风险判断。"],
        "evidence": {
            "mean_joint_range_deg": round(movement_range, 2),
            "trunk_flexion_p90_deg": round(trunk_p90, 2) if trunk_p90 is not None else None,
        },
        "experimental": False,
    }


def assess_movement(
    action_type: str,
    angle_series: Dict[str, List[float]],
    timestamps: Sequence[float],
) -> Dict:
    action = action_type if action_type in SUPPORTED_ACTIONS else "general"
    time_array = np.asarray(timestamps, dtype=np.float64)
    if action == "squat":
        result = _squat_assessment(angle_series, time_array)
    else:
        result = _general_assessment(angle_series, time_array)
        if action == "throw":
            result["experimental"] = True
            result["observations"].insert(
                0,
                "投掷动作需要更高帧率和侧后方机位；当前结果仅作动作趋势参考。",
            )
    result["action_type"] = action
    result["action_label"] = ACTION_LABELS[action]
    return result
