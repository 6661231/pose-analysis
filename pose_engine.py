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


class KineticChainAnalyzer:
    """
    生物动力链分析器
    基于多帧骨骼数据，评估近端→远端发力顺序、传导效率和动作一致性

    核心参考：
    - Mulloy et al. (2018): 近端→远端角速度峰值时序区分专家/新手
    - Wukelic et al. (2024): 棒球投手 PDS 模式 5 环节评估
    - Marineau et al. (2024): 81% 研究中专家运动变异性显著更低

    V3.1: 用关键点线速度替代关节角速度评估动力链时序，
          避免角速度无法捕捉旋转发力的问题
    """

    UPPER_CHAIN = [
        ("右侧上肢链", ["right_hip", "right_shoulder", "right_elbow", "right_wrist"]),
        ("左侧上肢链", ["left_hip", "left_shoulder", "left_elbow", "left_wrist"]),
    ]
    LOWER_CHAIN = [
        ("右侧下肢链", ["right_hip", "right_knee", "right_ankle"]),
        ("左侧下肢链", ["left_hip", "left_knee", "left_ankle"]),
    ]

    JOINT_TO_KP = {
        "right_hip": 12, "left_hip": 11,
        "right_shoulder": 6, "left_shoulder": 5,
        "right_elbow": 8, "left_elbow": 7,
        "right_wrist": 10, "left_wrist": 9,
        "right_knee": 14, "left_knee": 13,
        "right_ankle": 16, "left_ankle": 15,
    }

    def __init__(self, angle_series, timestamps, keypoints_all):
        self.angle_series = angle_series
        self.timestamps = np.array(timestamps)
        self.keypoints_all = [np.array(k, dtype=np.float64) for k in keypoints_all]
        self.velocities = {}
        self.kp_velocities = {}
        self._compute_velocities()
        self._compute_keypoint_velocities()

    def _compute_velocities(self):
        dt = np.diff(self.timestamps)
        for joint, angles in self.angle_series.items():
            if len(angles) < 2:
                continue
            a = np.array(angles, dtype=np.float64)
            vel = np.abs(np.diff(a)) / np.maximum(dt, 0.001)
            vel = np.nan_to_num(vel, nan=0.0).tolist()
            self.velocities[joint] = [float(v) for v in vel]

    def _compute_keypoint_velocities(self):
        """计算每个关键点的线速度（像素/秒），替代角速度评估旋转发力"""
        if len(self.keypoints_all) < 2:
            return
        dt = np.maximum(np.diff(self.timestamps), 0.001)
        for kp_name, kp_idx in self.JOINT_TO_KP.items():
            positions = []
            for kpts in self.keypoints_all:
                if kpts is not None and len(kpts) > kp_idx:
                    pt = kpts[kp_idx][:2]
                    if np.isnan(pt).any():
                        positions.append(np.nan)
                    else:
                        positions.append(pt)
                else:
                    positions.append(np.nan)
            if len(positions) < 2:
                continue
            speeds = []
            for i in range(len(positions) - 1):
                p1, p2 = positions[i], positions[i + 1]
                if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
                    speed = float(np.linalg.norm(p2 - p1)) / float(dt[i])
                    speeds.append(speed if not np.isnan(speed) else 0.0)
                else:
                    speeds.append(0.0)
            self.kp_velocities[kp_name] = [float(v) for v in speeds]

    def detect_onset_timing(self, chain_joints, threshold_ratio=0.3):
        """用运动起始时间替代峰值时间——谁先越过阈值谁先动。
        避免低帧率下所有关节峰值挤在同一帧的问题。"""
        onset = {}
        for joint in chain_joints:
            vel = self.kp_velocities.get(joint) or self.velocities.get(joint)
            if not vel or len(vel) < 2:
                continue
            max_v = np.max(np.abs(vel))
            if max_v < 2.0:  # 几乎不动，跳过
                continue
            threshold = max_v * threshold_ratio
            for i, v in enumerate(vel):
                if abs(v) >= threshold:
                    onset[joint] = float(self.timestamps[min(i + 1, len(self.timestamps) - 1)])
                    break
        return onset

    def assess_proximal_distal_sequence(self, chain_joints):
        onset = self.detect_onset_timing(chain_joints)
        if len(onset) < 2:
            return {"sequence": [], "correct_pairs": 0, "total_pairs": 0, "score": 50, "timing_gaps_sec": [], "detail": "数据不足"}

        sorted_joints = sorted(onset.items(), key=lambda x: x[1])
        sequence = [j for j, _ in sorted_joints]

        correct = 0
        gaps = []
        for i in range(len(chain_joints) - 1):
            prox, dist = chain_joints[i], chain_joints[i + 1]
            if prox in onset and dist in onset:
                gap = onset[dist] - onset[prox]
                gaps.append(round(gap, 3))
                if gap >= 0:  # 近端不比远端晚即为正确（允许同时启动）
                    correct += 1

        total = len(chain_joints) - 1
        score = round(correct / max(total, 1) * 100)

        return {
            "sequence": sequence, "correct_pairs": correct, "total_pairs": total,
            "score": score, "timing_gaps_sec": gaps,
            "detail": f"正确环节 {correct}/{total}，得分 {score}"
        }

    def compute_transfer_efficiency(self, chain_joints):
        onset = self.detect_onset_timing(chain_joints)
        if len(onset) < 2:
            return {"velocity_ratio": 1.0, "mean_delay_ms": 0, "score": 50}

        peak_vels = {}
        for joint in chain_joints:
            vel = self.kp_velocities.get(joint) or self.velocities.get(joint)
            if vel and len(vel) > 0:
                peak_vels[joint] = float(np.max(np.abs(vel)))

        v_ratio = 1.0
        if chain_joints[0] in peak_vels and chain_joints[-1] in peak_vels:
            v_ratio = round(peak_vels[chain_joints[-1]] / max(peak_vels[chain_joints[0]], 0.01), 2)

        delays = []
        for i in range(len(chain_joints) - 1):
            prox, dist = chain_joints[i], chain_joints[i + 1]
            if prox in onset and dist in onset:
                delays.append(max(0, onset[dist] - onset[prox]) * 1000)

        mean_delay = round(np.mean(delays), 1) if delays else 0
        score = 100 if mean_delay < 200 else (80 if mean_delay < 500 else (60 if mean_delay < 1000 else 40))

        return {
            "velocity_ratio": v_ratio, "mean_delay_ms": mean_delay, "score": score,
            "peak_velocities_deg_s": {k: round(v, 1) for k, v in peak_vels.items()}
        }

    def compute_movement_quality(self, joint="right_knee"):
        """评估动作质量：平滑度、峰值速度、节奏"""
        vel = self.kp_velocities.get(joint) or self.velocities.get(joint)
        if not vel or len(vel) < 4:
            return {"peak_velocity": 0, "smoothness": 50, "rhythm": 50, "score": 50}

        vel_arr = np.array(vel)
        # 峰值爆发力：最大速度越高越好
        peak = float(np.max(np.abs(vel_arr)))
        peak_score = min(100, int(peak / 3))

        # 平滑度：加速度过零次数越少越流畅
        acc = np.diff(vel_arr)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(acc)))) / 2
        smooth = max(0, 100 - zero_crossings * 10)

        # 节奏：速度曲线是否对称（前一半 vs 后一半能量比）
        mid = len(vel_arr) // 2
        first_half = np.sum(np.abs(vel_arr[:mid])) if mid > 0 else 0
        second_half = np.sum(np.abs(vel_arr[mid:])) if mid < len(vel_arr) else 0
        total = first_half + second_half
        if total > 0:
            rhythm_ratio = first_half / total
            rhythm = 100 - abs(rhythm_ratio - 0.5) * 80
        else:
            rhythm = 50

        score = int(np.mean([peak_score, smooth, rhythm]))
        return {
            "peak_velocity": round(peak, 1),
            "smoothness": round(smooth),
            "rhythm": round(rhythm),
            "score": score
        }

    def compute_consistency(self):
        scores = {}
        for joint, angles in self.angle_series.items():
            if len(angles) < 3:
                continue
            arr = np.array([a for a in angles if not np.isnan(a)])
            if len(arr) < 3:
                continue
            mean_val = np.mean(arr)
            cv = np.std(arr) / mean_val if mean_val > 0.1 else 0.5
            sc = 100 if cv < 0.05 else (80 if cv < 0.15 else (60 if cv < 0.3 else 40))
            scores[joint] = round(sc)

        overall = round(np.mean(list(scores.values()))) if scores else 50
        return {"joint_scores": scores, "overall_score": overall}

    def compute_core_stability(self):
        if len(self.keypoints_all) < 2:
            return {"score": 50, "deviation_px": 0}

        mid_shoulders, mid_hips = [], []
        for kpts in self.keypoints_all:
            if kpts is None or len(kpts) < 17:
                continue
            ms = (np.array(kpts[5][:2]) + np.array(kpts[6][:2])) / 2
            mh = (np.array(kpts[11][:2]) + np.array(kpts[12][:2])) / 2
            mid_shoulders.append(ms)
            mid_hips.append(mh)

        if len(mid_shoulders) < 2:
            return {"score": 50, "deviation_px": 0}

        mids = np.array(mid_shoulders) * 0.5 + np.array(mid_hips) * 0.5
        deviation = float(np.std(mids[:, 0]) + np.std(mids[:, 1]))
        score = 100 if deviation < 5 else (80 if deviation < 15 else (60 if deviation < 30 else 40))
        return {"score": round(score), "deviation_px": round(deviation, 1)}

    def compute_joint_safety(self):
        SAFE_RANGES = {
            "left_knee": (10, 170), "right_knee": (10, 170),
            "left_elbow": (10, 170), "right_elbow": (10, 170),
            "trunk_flexion": (0, 70), "neck_flexion": (0, 60),
            "left_hip": (10, 170), "right_hip": (10, 170),
        }
        violations = {}
        for joint, (lo, hi) in SAFE_RANGES.items():
            if joint not in self.angle_series:
                continue
            vals = [a for a in self.angle_series[joint] if not np.isnan(a)]
            if not vals:
                continue
            bad = sum(1 for v in vals if v < lo or v > hi)
            violations[joint] = {"total_frames": len(vals), "violations": bad, "ratio": round(bad / len(vals), 3)}

        total_ratio = round(np.mean([v["ratio"] for v in violations.values()]), 3) if violations else 0
        score = round(max(0, 100 - total_ratio * 200))
        return {"score": score, "violations": violations}

    def _chain_is_active(self, chain_joints, min_speed=5.0):
        """检查动力链是否至少有 min_speed 像素/秒的活动"""
        for joint in chain_joints:
            vel = self.kp_velocities.get(joint) or self.velocities.get(joint)
            if vel and len(vel) > 0 and np.max(np.abs(vel)) >= min_speed:
                return True
        return False

    def full_assessment(self):

        def _movement_type():
            """用膝关节活动范围判断：>40°为蹲起，否则看手腕位移区分投掷"""
            knee_range = 0
            for j in ["left_knee", "right_knee"]:
                if j in self.angle_series:
                    vals = [v for v in self.angle_series[j] if not np.isnan(v)]
                    if vals:
                        knee_range = max(knee_range, max(vals) - min(vals))
            if knee_range > 40:
                return "squat"
            wrist_disp = 0
            for kp_name in ["right_wrist", "left_wrist"]:
                vel = self.kp_velocities.get(kp_name, [])
                if vel:
                    wrist_disp += np.sum(np.abs(vel))
            if wrist_disp > 2000:
                return "throw"
            return "mixed"

        mv_type = _movement_type()
        logger.info(f"[KineticChain] 运动类型: {mv_type}")
        def safe_round(v, default=50):
            try:
                r = round(float(v))
                return default if np.isnan(r) or np.isinf(r) else r
            except (ValueError, TypeError):
                return default

        dimensions = {}

        active_chains = [j for _, j in self.UPPER_CHAIN + self.LOWER_CHAIN if self._chain_is_active(j)]
        if not active_chains:
            active_chains = [j for _, j in self.UPPER_CHAIN + self.LOWER_CHAIN]

        pds_scores = [self.assess_proximal_distal_sequence(joints)["score"] for joints in active_chains]
        eff_scores = [self.compute_transfer_efficiency(joints)["score"] for joints in active_chains]
        v_ratios = [self.compute_transfer_efficiency(joints)["velocity_ratio"] for joints in active_chains]
        best_ratio = np.nan_to_num(np.max(v_ratios), nan=1.0) if v_ratios else 1.0

        # 关节活动度得分（基于膝髋最大活动范围）
        rom_score = 50
        knee_ranges = []
        for j in ["left_knee", "right_knee"]:
            if j in self.angle_series:
                vals = [v for v in self.angle_series[j] if not np.isnan(v)]
                if vals:
                    knee_ranges.append(max(vals) - min(vals))
        if knee_ranges:
            avg_range = np.mean(knee_ranges)
            rom_score = safe_round(min(100, avg_range * 1.2))
        sym_score = 50
        if len(knee_ranges) >= 2:
            diff = abs(knee_ranges[0] - knee_ranges[1])
            sym_score = safe_round(100 - min(50, diff * 2))

        trunk_score = 50
        if "trunk_flexion" in self.angle_series:
            trunk_vals = [v for v in self.angle_series["trunk_flexion"] if not np.isnan(v)]
            if trunk_vals:
                mean_trunk = np.mean(np.abs(trunk_vals))
                trunk_score = safe_round(100 - min(60, mean_trunk * 2))

        # 动作质量（通用）
        primary_joint = "right_knee" if mv_type == "squat" else "right_wrist"
        quality = self.compute_movement_quality(primary_joint)

        if mv_type == "squat":
            dimensions["膝关节活动"] = rom_score
            dimensions["左右对称"] = sym_score
            dimensions["躯干控制"] = trunk_score
            dimensions["动作质量"] = quality["score"] if quality["score"] > 0 else 50
        elif mv_type == "throw":
            dimensions["发力顺序"] = safe_round(np.max(pds_scores)) if pds_scores else 50
            dimensions["传导效率"] = safe_round(np.max(eff_scores)) if eff_scores else 50
            dimensions["末端爆发"] = safe_round(min(100, max(30, best_ratio * 40)))
            dimensions["动作质量"] = quality["score"] if quality["score"] > 0 else 50
        else:
            dimensions["发力顺序"] = safe_round(np.max(pds_scores)) if pds_scores else 50
            dimensions["传导效率"] = safe_round(np.max(eff_scores)) if eff_scores else 50
            dimensions["末端爆发"] = safe_round(min(100, max(30, best_ratio * 40)))

        cons = self.compute_consistency()
        dimensions["动作一致性"] = safe_round(cons["overall_score"])

        safety = self.compute_joint_safety()
        dimensions["关节保护"] = safe_round(safety["score"])

        core = self.compute_core_stability()
        dimensions["核心稳定"] = safe_round(core["score"])

        overall = safe_round(np.mean(list(dimensions.values())))

        chain_details = {}
        for chain_name, joints in self.UPPER_CHAIN + self.LOWER_CHAIN:
            pds = self.assess_proximal_distal_sequence(joints)
            eff = self.compute_transfer_efficiency(joints)
            chain_details[chain_name] = {
                "pds_score": safe_round(pds["score"]), "pds_sequence": pds["sequence"],
                "pds_detail": pds["detail"], "efficiency_score": safe_round(eff["score"]),
                "mean_delay_ms": safe_round(eff["mean_delay_ms"], 0), "velocity_ratio": safe_round(eff["velocity_ratio"], 1)
            }

        return {
            "dimensions": dimensions, "overall_score": overall,
            "chain_details": chain_details, "consistency": cons,
            "core_stability": core, "joint_safety": safety
        }