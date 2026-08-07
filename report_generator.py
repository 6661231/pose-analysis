import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from movement_analyzer import ACTION_LABELS, SUPPORTED_ACTIONS, assess_movement
from pose_engine import PoseEngine
from pose_processing import json_safe_keypoints, process_pose_sequence

logger = logging.getLogger(__name__)


@dataclass
class RawFrame:
    frame_idx: int
    timestamp_sec: float
    keypoints: Optional[List[List[float]]]


def _nullable_number(value: float, digits: int = 3) -> Optional[float]:
    return round(float(value), digits) if np.isfinite(value) else None


class PoseReportGenerator:
    """Collect raw observations and produce one cleaned, time-aligned report."""

    def __init__(
        self,
        action_type: str = "squat",
        confidence_threshold: float = 0.3,
        max_gap_frames: int = 2,
    ):
        self.action_type = action_type if action_type in SUPPORTED_ACTIONS else "general"
        self.confidence_threshold = confidence_threshold
        self.max_gap_frames = max_gap_frames
        self.frames: List[RawFrame] = []
        self.engine = PoseEngine()

    def add_frame(
        self,
        frame_idx: int,
        timestamp: float,
        angles: Optional[Dict[str, float]] = None,
        offsets: Optional[Dict[str, float]] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> None:
        """Store raw detector output. Angles/offsets remain accepted for API compatibility."""
        del angles, offsets
        points = None
        if keypoints is not None:
            array = np.asarray(keypoints, dtype=np.float64)
            if array.shape == (17, 3):
                points = array.tolist()
        self.frames.append(RawFrame(
            frame_idx=int(frame_idx),
            timestamp_sec=round(float(timestamp), 6),
            keypoints=points,
        ))

    @staticmethod
    def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            return {"min": None, "max": None, "mean": None, "std": None, "range": None, "valid_count": 0}
        return {
            "min": round(float(np.min(array)), 2),
            "max": round(float(np.max(array)), 2),
            "mean": round(float(np.mean(array)), 2),
            "std": round(float(np.std(array)), 2),
            "range": round(float(np.max(array) - np.min(array)), 2),
            "valid_count": int(len(array)),
        }

    @staticmethod
    def _aligned_series(metric_rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
        keys = sorted({key for row in metric_rows for key in row})
        return {
            key: [float(row.get(key, np.nan)) for row in metric_rows]
            for key in keys
        }

    @staticmethod
    def _action_phases(
        values: Sequence[float],
        timestamps: Sequence[float],
        joint_name: str,
    ) -> List[Dict]:
        values_array = np.asarray(values, dtype=np.float64)
        time_array = np.asarray(timestamps, dtype=np.float64)
        phases: List[Dict] = []
        if len(values_array) < 3:
            return phases

        current_state: Optional[str] = None
        start_index = 0
        for index in range(1, len(values_array)):
            if not np.isfinite(values_array[index - 1:index + 1]).all():
                state = "unknown"
            else:
                dt = max(float(time_array[index] - time_array[index - 1]), 1e-3)
                velocity = float(values_array[index] - values_array[index - 1]) / dt
                state = "flexion" if velocity < -8 else ("extension" if velocity > 8 else "hold")

            if current_state is None:
                current_state = state
                start_index = index - 1
                continue
            if state != current_state:
                duration = float(time_array[index - 1] - time_array[start_index])
                if current_state != "unknown" and duration >= 0.1:
                    phases.append({
                        "phase_type": current_state,
                        "start_time_sec": round(float(time_array[start_index]), 3),
                        "end_time_sec": round(float(time_array[index - 1]), 3),
                        "duration_sec": round(duration, 3),
                        "joint": joint_name,
                    })
                current_state = state
                start_index = index - 1

        duration = float(time_array[-1] - time_array[start_index])
        if current_state not in {None, "unknown"} and duration >= 0.1:
            phases.append({
                "phase_type": current_state,
                "start_time_sec": round(float(time_array[start_index]), 3),
                "end_time_sec": round(float(time_array[-1]), 3),
                "duration_sec": round(duration, 3),
                "joint": joint_name,
            })
        return phases

    @staticmethod
    def _velocity_series(
        angle_series: Dict[str, List[float]],
        timestamps: Sequence[float],
    ) -> Dict[str, List[Optional[float]]]:
        time_array = np.asarray(timestamps, dtype=np.float64)
        output: Dict[str, List[Optional[float]]] = {}
        if len(time_array) < 2:
            return output
        dt = np.diff(time_array)
        for joint, values in angle_series.items():
            array = np.asarray(values, dtype=np.float64)
            velocity = np.full(max(len(array) - 1, 0), np.nan)
            valid = np.isfinite(array[:-1]) & np.isfinite(array[1:]) & (dt > 0)
            velocity[valid] = np.diff(array)[valid] / dt[valid]
            output[joint] = [_nullable_number(value, 2) for value in velocity]
        return output

    @staticmethod
    def _measurement_alerts(
        angle_series: Dict[str, List[float]],
        timestamps: Sequence[float],
        relevant_joints: Optional[set] = None,
    ) -> List[Dict]:
        """Flag likely measurement jumps; this is not a medical safety assessment."""
        time_array = np.asarray(timestamps, dtype=np.float64)
        alerts: List[Dict] = []
        if len(time_array) < 2:
            return alerts
        dt = np.diff(time_array)
        for joint, values in angle_series.items():
            if relevant_joints is not None and joint not in relevant_joints:
                continue
            array = np.asarray(values, dtype=np.float64)
            valid = np.isfinite(array[:-1]) & np.isfinite(array[1:]) & (dt > 0)
            speeds = np.full(len(dt), np.nan)
            speeds[valid] = np.abs(np.diff(array)[valid] / dt[valid])
            for index in np.flatnonzero(speeds > 500):
                alerts.append({
                    "type": "possible_measurement_jump",
                    "joint": joint,
                    "timestamp_sec": round(float(time_array[index + 1]), 3),
                    "angular_speed_deg_s": round(float(speeds[index]), 2),
                    "severity": "data_quality",
                })
        return alerts

    def generate(self, video_info: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.frames:
            return {"error": "无抽样帧数据"}

        ordered_frames = sorted(self.frames, key=lambda frame: frame.frame_idx)
        raw_timestamps = np.asarray(
            [frame.timestamp_sec for frame in ordered_frames],
            dtype=np.float64,
        )
        positive_intervals = np.diff(raw_timestamps)
        positive_intervals = positive_intervals[positive_intervals > 0]
        fallback_interval = float(np.median(positive_intervals)) if len(positive_intervals) else 0.1
        timestamp_corrections = 0
        if not np.isfinite(raw_timestamps[0]):
            raw_timestamps[0] = 0.0
            timestamp_corrections += 1
        for index in range(1, len(raw_timestamps)):
            if not np.isfinite(raw_timestamps[index]) or raw_timestamps[index] <= raw_timestamps[index - 1]:
                raw_timestamps[index] = raw_timestamps[index - 1] + fallback_interval
                timestamp_corrections += 1
        timestamps = raw_timestamps.tolist()
        raw_keypoints = [frame.keypoints for frame in ordered_frames]
        processed = process_pose_sequence(
            raw_keypoints,
            timestamps,
            confidence_threshold=self.confidence_threshold,
            max_gap_frames=self.max_gap_frames,
        )

        sample_rate = 0.0
        if len(timestamps) > 1:
            intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
            valid_intervals = intervals[intervals > 0]
            if len(valid_intervals):
                sample_rate = float(1.0 / np.median(valid_intervals))

        angle_rows = [self.engine.compute_joint_angles(points) for points in processed.keypoints]
        offset_rows = [self.engine.compute_limb_offsets(points) for points in processed.keypoints]
        angle_series = self._aligned_series(angle_rows)
        offset_series = self._aligned_series(offset_rows)

        joint_stats = {name: self._stats(values) for name, values in angle_series.items()}
        offset_stats = {name: self._stats(values) for name, values in offset_series.items()}
        phases: List[Dict] = []
        for joint in ["left_knee", "right_knee"]:
            if joint in angle_series:
                phases.extend(self._action_phases(angle_series[joint], timestamps, joint))

        assessment = assess_movement(self.action_type, angle_series, timestamps)
        quality = processed.quality.as_dict()
        quality["timestamp_corrections"] = timestamp_corrections
        if timestamp_corrections:
            quality["messages"].append(
                f"视频时间戳有 {timestamp_corrections} 处重复或倒退，已按帧顺序修正。"
            )
        minimum_temporal_fps = 15.0 if self.action_type == "throw" else 8.0
        quality["temporal_resolution_ok"] = sample_rate >= minimum_temporal_fps
        quality["minimum_recommended_fps"] = minimum_temporal_fps
        if not quality["temporal_resolution_ok"]:
            quality["score"] = min(quality["score"], 55)
            quality["level"] = "low"
            quality["is_reliable"] = False
            quality["messages"].insert(
                0,
                f"实际分析帧率仅 {sample_rate:.1f} FPS，低于该动作建议的 {minimum_temporal_fps:.0f} FPS。",
            )
        assessment["observations"] = quality["messages"] + assessment["observations"]
        relevant_alert_joints = None
        if self.action_type == "squat":
            relevant_alert_joints = {
                "left_knee",
                "right_knee",
                "left_hip",
                "right_hip",
                "trunk_flexion",
                "neck_flexion",
            }
        alerts = self._measurement_alerts(
            angle_series,
            timestamps,
            relevant_joints=relevant_alert_joints,
        )

        frame_data = []
        for index, frame in enumerate(ordered_frames):
            interpolated_indices = np.flatnonzero(
                processed.interpolated_mask[index, :, :2].any(axis=1)
            ).astype(int).tolist()
            frame_data.append({
                "frame_idx": frame.frame_idx,
                "timestamp_sec": round(float(timestamps[index]), 3),
                "detected": frame.keypoints is not None,
                "interpolated_keypoint_indices": interpolated_indices,
                "joint_angles": {
                    key: _nullable_number(value, 2)
                    for key, value in angle_rows[index].items()
                },
                "limb_offsets": {
                    key: _nullable_number(value, 4)
                    for key, value in offset_rows[index].items()
                },
                "keypoints": json_safe_keypoints(processed.keypoints[index]),
            })

        report = {
            "report_meta": {
                "generated_at": datetime.now().isoformat(),
                "version": "4.0.0",
                "engine": "YOLOv8-Pose + confidence-aware temporal analysis",
                "action_type": self.action_type,
                "action_label": ACTION_LABELS[self.action_type],
                "total_frames": len(ordered_frames),
                "detected_frames": quality["detected_frames"],
                "duration_sec": round(timestamps[-1] - timestamps[0], 2) if len(timestamps) > 1 else 0,
                "sample_rate_fps": round(sample_rate, 2),
            },
            "video_info": video_info or {},
            "data_quality": quality,
            "summary": {
                "overall_score": assessment["overall_score"],
                "dimensions": assessment["dimensions"],
                "is_reliable": quality["is_reliable"],
                "quality_score": quality["score"],
                "measurement_alert_count": len(alerts),
                "anomaly_count": len(alerts),
                "action_phase_count": len(phases),
                "repetition_count": len(assessment["repetitions"]),
            },
            "movement_assessment": assessment,
            "chain_details": {},
            "velocity_series": self._velocity_series(angle_series, timestamps),
            "joint_statistics": joint_stats,
            "offset_statistics": offset_stats,
            "action_phases": phases,
            "anomalies": alerts,
            "frame_data": frame_data,
            "disclaimer": "结果基于单目二维视频，仅用于动作趋势反馈，不构成医疗诊断、损伤风险结论或真实力学测量。",
        }
        return report

    def save(self, output_path: Path, video_info: Optional[Dict] = None) -> Dict[str, Any]:
        report = self.generate(video_info)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        logger.info("[Report] 报告已保存: %s (%d 帧)", output_path, len(self.frames))
        return report
