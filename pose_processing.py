from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PoseSequenceQuality:
    total_frames: int
    detected_frames: int
    valid_keypoint_ratio: float
    mean_keypoint_confidence: float
    interpolated_keypoints: int
    score: int
    level: str
    is_reliable: bool
    messages: List[str]

    def as_dict(self) -> Dict:
        return {
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate": round(
                self.detected_frames / max(self.total_frames, 1), 3
            ),
            "valid_keypoint_ratio": round(self.valid_keypoint_ratio, 3),
            "mean_keypoint_confidence": round(self.mean_keypoint_confidence, 3),
            "interpolated_keypoints": self.interpolated_keypoints,
            "score": self.score,
            "level": self.level,
            "is_reliable": self.is_reliable,
            "messages": self.messages,
        }


@dataclass
class ProcessedPoseSequence:
    keypoints: np.ndarray
    original_valid_mask: np.ndarray
    interpolated_mask: np.ndarray
    body_scale: np.ndarray
    quality: PoseSequenceQuality


def _as_keypoint_array(frames: Sequence[Optional[Iterable]]) -> np.ndarray:
    output = np.full((len(frames), 17, 3), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        if frame is None:
            continue
        array = np.asarray(frame, dtype=np.float64)
        if array.shape != (17, 3):
            continue
        output[frame_index] = array
    return output


def _interpolate_short_gaps(
    values: np.ndarray,
    timestamps: np.ndarray,
    max_gap_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    interpolated = np.zeros(values.shape, dtype=bool)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if len(valid_indices) < 2:
        return result, interpolated

    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        duration = timestamps[right] - timestamps[left]
        if not np.isfinite(duration) or duration <= 0:
            continue
        for index in range(left + 1, right):
            ratio = (timestamps[index] - timestamps[left]) / duration
            result[index] = values[left] + ratio * (values[right] - values[left])
            interpolated[index] = True
    return result, interpolated


def _time_aware_ema(
    values: np.ndarray,
    timestamps: np.ndarray,
    smoothing_time_sec: float,
) -> np.ndarray:
    result = values.copy()
    previous_value = np.nan
    previous_time = np.nan

    for index, value in enumerate(values):
        if not np.isfinite(value):
            previous_value = np.nan
            previous_time = np.nan
            continue
        if not np.isfinite(previous_value):
            result[index] = value
        else:
            dt = max(float(timestamps[index] - previous_time), 1e-3)
            alpha = 1.0 - np.exp(-dt / max(smoothing_time_sec, 1e-3))
            result[index] = previous_value + alpha * (value - previous_value)
        previous_value = result[index]
        previous_time = timestamps[index]

    return result


def _body_scale_for_frame(keypoints: np.ndarray) -> float:
    def distance(first: int, second: int) -> float:
        points = keypoints[[first, second], :2]
        if not np.isfinite(points).all():
            return np.nan
        return float(np.linalg.norm(points[0] - points[1]))

    candidates = [
        distance(5, 6),
        distance(11, 12),
        distance(5, 11),
        distance(6, 12),
    ]
    valid = [value for value in candidates if np.isfinite(value) and value > 1.0]
    return float(np.median(valid)) if valid else np.nan


def _fill_body_scale(scales: np.ndarray) -> np.ndarray:
    result = scales.copy()
    valid = np.flatnonzero(np.isfinite(result) & (result > 1.0))
    if len(valid) == 0:
        return np.ones_like(result)
    fallback = float(np.median(result[valid]))
    result[~np.isfinite(result) | (result <= 1.0)] = fallback
    return result


def _quality_summary(
    raw: np.ndarray,
    original_valid: np.ndarray,
    interpolated: np.ndarray,
) -> PoseSequenceQuality:
    total_frames = len(raw)
    detected_by_coordinates = np.isfinite(raw[:, :, :2]).any(axis=(1, 2))
    detected_frames = int(np.sum(detected_by_coordinates))
    valid_ratio = float(np.mean(original_valid)) if total_frames else 0.0
    confidences = raw[:, :, 2][original_valid]
    mean_confidence = float(np.mean(confidences)) if len(confidences) else 0.0
    detection_rate = detected_frames / max(total_frames, 1)

    score = int(round(100 * (
        0.45 * detection_rate
        + 0.40 * valid_ratio
        + 0.15 * np.clip(mean_confidence, 0.0, 1.0)
    )))
    level = "high" if score >= 80 else ("medium" if score >= 60 else "low")
    messages: List[str] = []
    if detection_rate < 0.8:
        messages.append("部分抽样帧未检测到人体，请减少遮挡并保证全身入镜。")
    if valid_ratio < 0.75:
        messages.append("有效关键点较少，关节角度和时序结果可能不稳定。")
    if mean_confidence < 0.55:
        messages.append("关键点平均置信度偏低，建议改善光线或拍摄距离。")
    if not messages:
        messages.append("关键点数据完整度良好，可用于动作趋势分析。")

    return PoseSequenceQuality(
        total_frames=total_frames,
        detected_frames=detected_frames,
        valid_keypoint_ratio=valid_ratio,
        mean_keypoint_confidence=mean_confidence,
        interpolated_keypoints=int(np.sum(interpolated.any(axis=2))),
        score=score,
        level=level,
        is_reliable=score >= 60 and detected_frames >= 3,
        messages=messages,
    )


def process_pose_sequence(
    frames: Sequence[Optional[Iterable]],
    timestamps: Sequence[float],
    confidence_threshold: float = 0.3,
    max_gap_frames: int = 2,
    smoothing_time_sec: float = 0.12,
) -> ProcessedPoseSequence:
    """Clean a sequence without turning missing observations into real coordinates."""
    raw = _as_keypoint_array(frames)
    time_array = np.asarray(timestamps, dtype=np.float64)
    if len(raw) != len(time_array):
        raise ValueError("关键点帧数与时间戳数量不一致")
    if len(time_array) > 1 and np.any(np.diff(time_array) <= 0):
        raise ValueError("时间戳必须严格递增")

    confidence = raw[:, :, 2]
    original_valid = (
        np.isfinite(raw[:, :, 0])
        & np.isfinite(raw[:, :, 1])
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
    )

    cleaned = raw.copy()
    cleaned[:, :, :2][~original_valid] = np.nan
    interpolated_mask = np.zeros_like(cleaned, dtype=bool)

    for keypoint_index in range(17):
        for coordinate_index in range(2):
            series, mask = _interpolate_short_gaps(
                cleaned[:, keypoint_index, coordinate_index],
                time_array,
                max_gap_frames,
            )
            cleaned[:, keypoint_index, coordinate_index] = _time_aware_ema(
                series,
                time_array,
                smoothing_time_sec,
            )
            interpolated_mask[:, keypoint_index, coordinate_index] = mask

        coordinate_valid = np.isfinite(cleaned[:, keypoint_index, :2]).all(axis=1)
        was_interpolated = interpolated_mask[:, keypoint_index, :2].any(axis=1)
        cleaned[:, keypoint_index, 2] = np.where(
            original_valid[:, keypoint_index],
            raw[:, keypoint_index, 2],
            np.where(was_interpolated & coordinate_valid, confidence_threshold, np.nan),
        )

    body_scale = _fill_body_scale(
        np.array([_body_scale_for_frame(frame) for frame in cleaned])
    )
    quality = _quality_summary(raw, original_valid, interpolated_mask)
    return ProcessedPoseSequence(
        keypoints=cleaned,
        original_valid_mask=original_valid,
        interpolated_mask=interpolated_mask,
        body_scale=body_scale,
        quality=quality,
    )


def json_safe_keypoints(keypoints: np.ndarray) -> List[List[Optional[float]]]:
    output: List[List[Optional[float]]] = []
    for point in np.asarray(keypoints, dtype=np.float64):
        output.append([
            round(float(value), 4) if np.isfinite(value) else None
            for value in point
        ])
    return output
