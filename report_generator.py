import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    frame_idx: int
    timestamp_sec: float
    joint_angles: Dict[str, float]
    limb_offsets: Dict[str, float]
    keypoints: List[List[float]]  # 17 x 3，JSON 可序列化


class PoseReportGenerator:
    """
    姿态数据分析报告生成器
    聚合多帧量化数据，输出包含统计摘要、时序曲线、动作阶段、异常告警的 JSON
    """

    def __init__(self):
        self.frames: List[FrameMetrics] = []

    def add_frame(
        self,
        frame_idx: int,
        timestamp: float,
        angles: Dict[str, float],
        offsets: Dict[str, float],
        keypoints: np.ndarray
    ):
        """添加单帧分析数据"""
        self.frames.append(FrameMetrics(
            frame_idx=frame_idx,
            timestamp_sec=round(timestamp, 3),
            joint_angles={k: round(v, 2) for k, v in angles.items()},
            limb_offsets={k: round(v, 4) for k, v in offsets.items()},
            keypoints=np.nan_to_num(keypoints, nan=0.0).tolist()
        ))

    @staticmethod
    def _stats(values: List[float]) -> Dict[str, Optional[float]]:
        """数值列表统计量"""
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return {"min": None, "max": None, "mean": None, "std": None, "range": None}
        return {
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "range": round(float(np.max(arr) - np.min(arr)), 2)
        }

    def _action_phases(self, values: List[float], timestamps: List[float], joint_name: str) -> List[Dict]:
        """
        基于角度变化率划分动作阶段：屈曲(flexion) / 伸展(extension) / 静止(hold)
        """
        phases = []
        if len(values) < 3:
            return phases

        diffs = np.diff(values)
        state = "hold"
        start_i = 0

        for i, d in enumerate(diffs):
            new_state = "flexion" if d < -3 else ("extension" if d > 3 else "hold")
            if new_state != state:
                if i - start_i >= 1:
                    phases.append({
                        "phase_type": state,
                        "start_time_sec": round(timestamps[start_i], 2),
                        "end_time_sec": round(timestamps[i], 2),
                        "duration_sec": round(timestamps[i] - timestamps[start_i], 2),
                        "joint": joint_name
                    })
                state = new_state
                start_i = i

        # 收尾
        if start_i < len(timestamps) - 1:
            phases.append({
                "phase_type": state,
                "start_time_sec": round(timestamps[start_i], 2),
                "end_time_sec": round(timestamps[-1], 2),
                "duration_sec": round(timestamps[-1] - timestamps[start_i], 2),
                "joint": joint_name
            })

        return phases

    def _anomaly_detection(
        self,
        angle_series: Dict[str, List[float]],
        timestamps: List[float]
    ) -> List[Dict]:
        """异常姿态检测：超限、突变、不对称"""
        alerts = []

        # 关节安全范围（示例阈值，可按动作类型动态配置）
        SAFE_RANGES = {
            "left_knee": (5, 175),
            "right_knee": (5, 175),
            "left_elbow": (5, 175),
            "right_elbow": (5, 175),
            "trunk_flexion": (0, 80),
            "neck_flexion": (0, 70)
        }

        # 范围超限检测
        for joint, vals in angle_series.items():
            safe = SAFE_RANGES.get(joint)
            if not safe:
                continue
            for i, v in enumerate(vals):
                if v < safe[0] or v > safe[1]:
                    alerts.append({
                        "type": "joint_range_exceeded",
                        "joint": joint,
                        "timestamp_sec": round(timestamps[i], 2),
                        "value_deg": round(v, 2),
                        "safe_range": safe,
                        "severity": "warning" if v < safe[1] + 10 else "critical"
                    })

        # 帧间突变检测 (> 15°/帧 视为异常抖动或快速动作)
        for joint, vals in angle_series.items():
            diffs = np.abs(np.diff(vals))
            for i, d in enumerate(diffs):
                if d > 15:
                    alerts.append({
                        "type": "sudden_angle_change",
                        "joint": joint,
                        "timestamp_sec": round(timestamps[i + 1], 2),
                        "delta_deg": round(float(d), 2),
                        "severity": "warning"
                    })

        return alerts

    def generate(self, video_info: Optional[Dict] = None) -> Dict[str, Any]:
        """生成完整 JSON 报告字典"""
        if not self.frames:
            return {"error": "无有效帧数据"}

        timestamps = [f.timestamp_sec for f in self.frames]

        # 按关节聚合时序
        angle_series: Dict[str, List[float]] = {}
        offset_series: Dict[str, List[float]] = {}
        for f in self.frames:
            for k, v in f.joint_angles.items():
                angle_series.setdefault(k, []).append(v)
            for k, v in f.limb_offsets.items():
                offset_series.setdefault(k, []).append(v)

        # 统计摘要
        joint_stats = {k: self._stats(v) for k, v in angle_series.items()}
        offset_stats = {k: self._stats(v) for k, v in offset_series.items()}

        # 动作阶段（以膝关节为主）
        phases = []
        for jname in ["left_knee", "right_knee"]:
            if jname in angle_series:
                phases.extend(self._action_phases(angle_series[jname], timestamps, jname))

        # 异常检测
        anomalies = self._anomaly_detection(angle_series, timestamps)

        # 综合评分（示例算法：基于肩髋对称性）
        symmetry_score = 100.0
        if "shoulder_height_diff_px" in offset_series:
            diffs = [v for v in offset_series["shoulder_height_diff_px"] if not np.isnan(v)]
            if diffs:
                avg_diff = np.mean(diffs)
                symmetry_score -= min(avg_diff / 3.0, 40)  # 每 3 像素扣 10 分，上限 40

        report = {
            "report_meta": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "engine": "YOLOv8-Pose + atan2_float64",
                "total_frames": len(self.frames),
                "duration_sec": round(timestamps[-1] - timestamps[0], 2) if len(timestamps) > 1 else 0
            },
            "video_info": video_info or {},
            "summary": {
                "symmetry_score": round(max(0, symmetry_score), 1),
                "anomaly_count": len(anomalies),
                "action_phase_count": len(phases)
            },
            "joint_statistics": joint_stats,
            "offset_statistics": offset_stats,
            "action_phases": phases,
            "anomalies": anomalies,
            "frame_data": [asdict(f) for f in self.frames]  # 逐帧明细（可选，数据量大时可关闭）
        }

        return report

    def save(self, output_path: Path, video_info: Optional[Dict] = None) -> Dict[str, Any]:
        """保存报告为 JSON 文件"""
        report = self.generate(video_info)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[Report] 报告已保存: {output_path} ({len(self.frames)} 帧)")
        return report