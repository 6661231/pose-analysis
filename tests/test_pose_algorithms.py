import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from movement_analyzer import assess_movement
from pose_engine import PoseEngine
from pose_processing import process_pose_sequence
from report_generator import PoseReportGenerator


def upright_keypoints() -> np.ndarray:
    points = np.full((17, 3), np.nan, dtype=np.float64)
    coordinates = {
        0: (100, 50),
        5: (90, 100), 6: (110, 100),
        7: (80, 150), 8: (120, 150),
        9: (75, 200), 10: (125, 200),
        11: (92, 160), 12: (108, 160),
        13: (92, 220), 14: (108, 220),
        15: (92, 280), 16: (108, 280),
    }
    for index, (x, y) in coordinates.items():
        points[index] = [x, y, 0.9]
    return points


class PoseEngineTests(unittest.TestCase):
    def test_upright_neck_angle_is_near_zero(self):
        angles = PoseEngine().compute_joint_angles(upright_keypoints())
        self.assertAlmostEqual(angles["neck_flexion"], 0.0, delta=0.1)

    def test_degenerate_angle_is_missing(self):
        point = np.array([1.0, 1.0, 0.9])
        angle = PoseEngine._angle_3p(point, point, point)
        self.assertTrue(np.isnan(angle))


class PoseProcessingTests(unittest.TestCase):
    def test_short_missing_gap_is_interpolated_without_becoming_zero(self):
        first = upright_keypoints()
        middle = upright_keypoints()
        last = upright_keypoints()
        middle[9] = [np.nan, np.nan, 0.1]
        last[9, 0] += 10

        result = process_pose_sequence([first, middle, last], [0.0, 0.1, 0.2])

        self.assertFalse(result.original_valid_mask[1, 9])
        self.assertTrue(result.interpolated_mask[1, 9, 0])
        self.assertTrue(np.isfinite(result.keypoints[1, 9, :2]).all())
        self.assertFalse(np.allclose(result.keypoints[1, 9, :2], [0, 0]))

    def test_long_missing_gap_remains_missing(self):
        frames = [upright_keypoints() for _ in range(5)]
        for index in [1, 2, 3]:
            frames[index][9] = [np.nan, np.nan, 0.1]

        result = process_pose_sequence(
            frames,
            [0.0, 0.1, 0.2, 0.3, 0.4],
            max_gap_frames=2,
        )

        self.assertTrue(np.isnan(result.keypoints[2, 9, :2]).all())


class MovementAnalyzerTests(unittest.TestCase):
    def test_complete_squat_repetition_is_detected(self):
        timestamps = np.arange(0.0, 2.1, 0.1)
        descent = np.linspace(175, 85, 11)
        ascent = np.linspace(94, 175, 10)
        knees = np.concatenate([descent, ascent])
        angles = {
            "left_knee": knees.tolist(),
            "right_knee": (knees + 2).tolist(),
            "trunk_flexion": np.linspace(8, 24, len(knees)).tolist(),
        }

        result = assess_movement("squat", angles, timestamps)

        self.assertEqual(len(result["repetitions"]), 1)
        self.assertGreaterEqual(result["dimensions"]["下蹲活动范围"], 90)
        self.assertIn("左右协调", result["dimensions"])


class ReportGeneratorTests(unittest.TestCase):
    def test_missing_frame_is_preserved_and_json_is_strict(self):
        generator = PoseReportGenerator(action_type="squat")
        for index in range(5):
            generator.add_frame(
                frame_idx=index,
                timestamp=index * 0.1,
                keypoints=None if index == 2 else upright_keypoints(),
            )

        report = generator.generate()
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)

        self.assertTrue(encoded)
        self.assertEqual(report["report_meta"]["total_frames"], 5)
        self.assertEqual(report["report_meta"]["detected_frames"], 4)
        self.assertFalse(report["frame_data"][2]["detected"])
        self.assertNotEqual(report["frame_data"][2]["keypoints"][0][:2], [0.0, 0.0])

    def test_save_writes_valid_json(self):
        generator = PoseReportGenerator(action_type="general")
        generator.add_frame(0, 0.0, keypoints=upright_keypoints())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            generator.save(path)
            with open(path, "r", encoding="utf-8") as file:
                saved = json.load(file)
        self.assertEqual(saved["report_meta"]["version"], "4.0.0")

    def test_duplicate_video_timestamps_are_repaired(self):
        generator = PoseReportGenerator(action_type="general")
        generator.add_frame(0, 0.0, keypoints=upright_keypoints())
        generator.add_frame(1, 0.0, keypoints=upright_keypoints())
        generator.add_frame(2, 0.1, keypoints=upright_keypoints())

        report = generator.generate()

        self.assertGreaterEqual(report["data_quality"]["timestamp_corrections"], 1)
        timestamps = [frame["timestamp_sec"] for frame in report["frame_data"]]
        self.assertGreater(timestamps[1], timestamps[0])


if __name__ == "__main__":
    unittest.main()
