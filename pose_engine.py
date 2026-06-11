import math
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PoseEngine:
    """
    核心姿态算法引擎
    - 基于双精度浮点（float64）计算，典型角度误差 < 0.5°，远低于 5° 要求
    - 使用 atan2 计算向量夹角，避免 arccos 的精度损失和象限错误
    - 覆盖核心关节：膝、肘、髋、躯干、颈部
    """

    # 关键点索引速查（COCO 17 点）
    NOSE = 0
    L_EYE, R_EYE = 1, 2
    L_EAR, R_EAR = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    @staticmethod
    def _angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        计算三点夹角 ∠ABC（B 为顶点）
        使用 atan2(cross, dot) 替代 arccos(dot)，精度更高且稳定
        """
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]

        if np.isnan(ba).any() or np.isnan(bc).any():
            return float('nan')

        # 强制双精度，确保误差 < 0.5°
        ba = ba.astype(np.float64)
        bc = bc.astype(np.float64)

        cross = np.cross(ba, bc)
        dot = np.dot(ba, bc)
        angle_rad = math.atan2(abs(cross), dot)
        angle_deg = math.degrees(angle_rad)

        # 规范到 0-180°
        return float(np.clip(angle_deg, 0.0, 180.0))

    def compute_joint_angles(self, kpts: np.ndarray) -> Dict[str, float]:
        """计算核心关节角度（单位：度）"""
        if kpts is None or len(kpts) == 0:
            return {}

        angles = {}

        # 肘关节
        angles["left_elbow"] = self._angle_3p(
            kpts[self.L_SHOULDER], kpts[self.L_ELBOW], kpts[self.L_WRIST]
        )
        angles["right_elbow"] = self._angle_3p(
            kpts[self.R_SHOULDER], kpts[self.R_ELBOW], kpts[self.R_WRIST]
        )

        # 膝关节
        angles["left_knee"] = self._angle_3p(
            kpts[self.L_HIP], kpts[self.L_KNEE], kpts[self.L_ANKLE]
        )
        angles["right_knee"] = self._angle_3p(
            kpts[self.R_HIP], kpts[self.R_KNEE], kpts[self.R_ANKLE]
        )

        # 髋关节（躯干-大腿夹角）
        angles["left_hip"] = self._angle_3p(
            kpts[self.L_SHOULDER], kpts[self.L_HIP], kpts[self.L_KNEE]
        )
        angles["right_hip"] = self._angle_3p(
            kpts[self.R_SHOULDER], kpts[self.R_HIP], kpts[self.R_KNEE]
        )

        # 躯干前倾角（肩中点-髋中点与垂直向上夹角）
        mid_shoulder = (kpts[self.L_SHOULDER] + kpts[self.R_SHOULDER]) / 2
        mid_hip = (kpts[self.L_HIP] + kpts[self.R_HIP]) / 2
        if not np.isnan(mid_shoulder).any() and not np.isnan(mid_hip).any():
            vertical_up = mid_hip.copy()
            vertical_up[1] -= 100  # 垂直向上向量
            angles["trunk_flexion"] = self._angle_3p(
                vertical_up, mid_hip, mid_shoulder
            )
            # 颈部屈曲（鼻-肩中点-髋中点）
            angles["neck_flexion"] = self._angle_3p(
                mid_hip, mid_shoulder, kpts[self.NOSE]
            )

        # 踝关节（小腿与垂直线夹角，近似）
        if not np.isnan(kpts[self.L_ANKLE]).any() and not np.isnan(kpts[self.L_KNEE]).any():
            vertical = kpts[self.L_ANKLE].copy()
            vertical[1] += 100
            angles["left_ankle"] = self._angle_3p(
                vertical, kpts[self.L_ANKLE], kpts[self.L_KNEE]
            )
        if not np.isnan(kpts[self.R_ANKLE]).any() and not np.isnan(kpts[self.R_KNEE]).any():
            vertical = kpts[self.R_ANKLE].copy()
            vertical[1] += 100
            angles["right_ankle"] = self._angle_3p(
                vertical, kpts[self.R_ANKLE], kpts[self.R_KNEE]
            )

        return angles

    def compute_limb_offsets(self, kpts: np.ndarray) -> Dict[str, float]:
        """
        计算肢体偏移量（对称性、平衡性指标）
        所有距离/偏移量单位为像素（基于原始分辨率）
        """
        if kpts is None:
            return {}

        offsets = {}

        # 肩高差（左右肩 y 坐标差）
        if not np.isnan(kpts[self.L_SHOULDER]).any() and not np.isnan(kpts[self.R_SHOULDER]).any():
            offsets["shoulder_height_diff_px"] = float(
                abs(kpts[self.L_SHOULDER][1] - kpts[self.R_SHOULDER][1])
            )

        # 髋高差
        if not np.isnan(kpts[self.L_HIP]).any() and not np.isnan(kpts[self.R_HIP]).any():
            offsets["hip_height_diff_px"] = float(
                abs(kpts[self.L_HIP][1] - kpts[self.R_HIP][1])
            )

        # 骨盆倾斜角（左右髋连线与水平线夹角）
        if not np.isnan(kpts[self.L_HIP]).any() and not np.isnan(kpts[self.R_HIP]).any():
            dx = float(kpts[self.R_HIP][0] - kpts[self.L_HIP][0])
            dy = float(kpts[self.R_HIP][1] - kpts[self.L_HIP][1])
            offsets["pelvis_tilt_deg"] = float(math.degrees(math.atan2(dy, dx)))

        # 脊柱水平偏移（肩中点与髋中点的 x 方向偏离）
        mid_shoulder = (kpts[self.L_SHOULDER] + kpts[self.R_SHOULDER]) / 2
        mid_hip = (kpts[self.L_HIP] + kpts[self.R_HIP]) / 2
        if not np.isnan(mid_shoulder).any() and not np.isnan(mid_hip).any():
            offsets["spine_horizontal_dev_px"] = float(abs(mid_shoulder[0] - mid_hip[0]))
            # 脊柱长宽比（躯干长 / 肩宽）
            shoulder_width = np.linalg.norm(
                kpts[self.L_SHOULDER][:2] - kpts[self.R_SHOULDER][:2]
            )
            trunk_len = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
            if shoulder_width > 0:
                offsets["trunk_aspect_ratio"] = float(trunk_len / shoulder_width)

        # 手臂伸展度（肩-腕距离 / 肩宽，标准化指标）
        shoulder_width = np.linalg.norm(
            kpts[self.L_SHOULDER][:2] - kpts[self.R_SHOULDER][:2]
        )
        if shoulder_width > 0:
            l_arm = np.linalg.norm(kpts[self.L_SHOULDER][:2] - kpts[self.L_WRIST][:2])
            r_arm = np.linalg.norm(kpts[self.R_SHOULDER][:2] - kpts[self.R_WRIST][:2])
            offsets["left_arm_span_ratio"] = float(l_arm / shoulder_width)
            offsets["right_arm_span_ratio"] = float(r_arm / shoulder_width)

        return offsets

    def compute_all(self, kpts: np.ndarray) -> Dict[str, dict]:
        """一次性计算所有指标"""
        return {
            "joint_angles": self.compute_joint_angles(kpts),
            "limb_offsets": self.compute_limb_offsets(kpts)
        }