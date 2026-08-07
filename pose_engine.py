import math
from typing import Dict

import numpy as np


class PoseEngine:
    """Compute stable 2D projection measurements from COCO-17 keypoints.

    These measurements describe the camera-plane projection. They are not a
    guarantee of real-world 3D joint-angle accuracy.
    """

    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    @staticmethod
    def _angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Return angle ABC in degrees, or NaN when a vector is unavailable."""
        first = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
        second = np.asarray(c, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            return float("nan")
        if np.linalg.norm(first) < 1e-9 or np.linalg.norm(second) < 1e-9:
            return float("nan")

        cross = first[0] * second[1] - first[1] * second[0]
        dot = float(np.dot(first, second))
        angle = math.degrees(math.atan2(abs(float(cross)), dot))
        return float(np.clip(angle, 0.0, 180.0))

    @staticmethod
    def _valid_shape(keypoints: np.ndarray) -> bool:
        return keypoints is not None and np.asarray(keypoints).shape == (17, 3)

    @staticmethod
    def _xy_valid(keypoints: np.ndarray, *indices: int) -> bool:
        return np.isfinite(keypoints[list(indices), :2]).all()

    def compute_joint_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        if not self._valid_shape(keypoints):
            return {}
        points = np.asarray(keypoints, dtype=np.float64)
        angles = {
            "left_elbow": self._angle_3p(
                points[self.L_SHOULDER], points[self.L_ELBOW], points[self.L_WRIST]
            ),
            "right_elbow": self._angle_3p(
                points[self.R_SHOULDER], points[self.R_ELBOW], points[self.R_WRIST]
            ),
            "left_knee": self._angle_3p(
                points[self.L_HIP], points[self.L_KNEE], points[self.L_ANKLE]
            ),
            "right_knee": self._angle_3p(
                points[self.R_HIP], points[self.R_KNEE], points[self.R_ANKLE]
            ),
            "left_hip": self._angle_3p(
                points[self.L_SHOULDER], points[self.L_HIP], points[self.L_KNEE]
            ),
            "right_hip": self._angle_3p(
                points[self.R_SHOULDER], points[self.R_HIP], points[self.R_KNEE]
            ),
        }

        if self._xy_valid(
            points,
            self.L_SHOULDER,
            self.R_SHOULDER,
            self.L_HIP,
            self.R_HIP,
        ):
            shoulder_midpoint = (
                points[self.L_SHOULDER, :2] + points[self.R_SHOULDER, :2]
            ) / 2
            hip_midpoint = (points[self.L_HIP, :2] + points[self.R_HIP, :2]) / 2
            vertical_up = hip_midpoint + np.array([0.0, -100.0])
            angles["trunk_flexion"] = self._angle_3p(
                vertical_up, hip_midpoint, shoulder_midpoint
            )
            torso_extension = shoulder_midpoint + (shoulder_midpoint - hip_midpoint)
            angles["neck_flexion"] = self._angle_3p(
                torso_extension, shoulder_midpoint, points[self.NOSE]
            )

        # COCO-17 has no heel/toe points, so this is shank inclination, not ankle angle.
        for side, knee_index, ankle_index in [
            ("left", self.L_KNEE, self.L_ANKLE),
            ("right", self.R_KNEE, self.R_ANKLE),
        ]:
            if self._xy_valid(points, knee_index, ankle_index):
                vertical_up = points[ankle_index, :2] + np.array([0.0, -100.0])
                angles[f"{side}_shank_inclination"] = self._angle_3p(
                    vertical_up,
                    points[ankle_index],
                    points[knee_index],
                )
        return angles

    def compute_limb_offsets(self, keypoints: np.ndarray) -> Dict[str, float]:
        if not self._valid_shape(keypoints):
            return {}
        points = np.asarray(keypoints, dtype=np.float64)
        offsets: Dict[str, float] = {}

        shoulder_width = float("nan")
        if self._xy_valid(points, self.L_SHOULDER, self.R_SHOULDER):
            shoulder_delta = points[self.R_SHOULDER, :2] - points[self.L_SHOULDER, :2]
            shoulder_width = float(np.linalg.norm(shoulder_delta))
            offsets["shoulder_height_diff_px"] = abs(float(shoulder_delta[1]))

        if self._xy_valid(points, self.L_HIP, self.R_HIP):
            hip_delta = points[self.R_HIP, :2] - points[self.L_HIP, :2]
            offsets["hip_height_diff_px"] = abs(float(hip_delta[1]))
            offsets["pelvis_tilt_deg"] = abs(float(math.degrees(
                math.atan2(abs(float(hip_delta[1])), max(abs(float(hip_delta[0])), 1e-9))
            )))

        if self._xy_valid(
            points,
            self.L_SHOULDER,
            self.R_SHOULDER,
            self.L_HIP,
            self.R_HIP,
        ):
            shoulder_midpoint = (
                points[self.L_SHOULDER, :2] + points[self.R_SHOULDER, :2]
            ) / 2
            hip_midpoint = (points[self.L_HIP, :2] + points[self.R_HIP, :2]) / 2
            trunk_length = float(np.linalg.norm(shoulder_midpoint - hip_midpoint))
            offsets["spine_horizontal_dev_px"] = abs(
                float(shoulder_midpoint[0] - hip_midpoint[0])
            )
            if shoulder_width > 1e-9:
                offsets["trunk_aspect_ratio"] = trunk_length / shoulder_width

        if shoulder_width > 1e-9:
            for pixel_name, ratio_name in [
                ("shoulder_height_diff_px", "shoulder_height_diff_ratio"),
                ("hip_height_diff_px", "hip_height_diff_ratio"),
                ("spine_horizontal_dev_px", "spine_horizontal_dev_ratio"),
            ]:
                if pixel_name in offsets:
                    offsets[ratio_name] = offsets[pixel_name] / shoulder_width

            for side, shoulder_index, wrist_index in [
                ("left", self.L_SHOULDER, self.L_WRIST),
                ("right", self.R_SHOULDER, self.R_WRIST),
            ]:
                if self._xy_valid(points, shoulder_index, wrist_index):
                    arm_span = np.linalg.norm(
                        points[shoulder_index, :2] - points[wrist_index, :2]
                    )
                    offsets[f"{side}_arm_span_ratio"] = float(arm_span / shoulder_width)
        return offsets

    def compute_all(self, keypoints: np.ndarray) -> Dict[str, dict]:
        return {
            "joint_angles": self.compute_joint_angles(keypoints),
            "limb_offsets": self.compute_limb_offsets(keypoints),
        }
