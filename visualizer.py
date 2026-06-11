import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# COCO 骨架连接定义
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 面部
    (5, 6),                               # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),      # 手臂
    (11, 12),                             # 髋
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿
    (5, 11), (6, 12)                      # 躯干
]

# 颜色配置 BGR
COLOR_LEFT = (0, 255, 0)       # 绿 - 左侧肢体
COLOR_RIGHT = (0, 0, 255)      # 红 - 右侧肢体
COLOR_CENTER = (255, 255, 0)   # 青 - 中轴
COLOR_BONE = (255, 200, 100)   # 骨骼线
COLOR_JOINT = (255, 0, 255)    # 关节点
COLOR_TEXT = (255, 255, 255)   # 文字
COLOR_PANEL = (0, 0, 0)        # 信息面板背景


class PoseVisualizer:
    """
    骨骼可视化器
    - 绘制骨架连线、关节圆点
    - 叠加关节角度数值
    - 支持左右肢体颜色区分
    """

    def __init__(
        self,
        line_thickness: int = 2,
        joint_radius: int = 4,
        font_scale: float = 0.5,
        show_angles: bool = True,
        show_indices: bool = False
    ):
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self.font_scale = font_scale
        self.show_angles = show_angles
        self.show_indices = show_indices

    def _joint_color(self, idx: int) -> Tuple[int, int, int]:
        if idx in [1, 3, 5, 7, 9, 11, 13, 15]:
            return COLOR_LEFT
        if idx in [2, 4, 6, 8, 10, 12, 14, 16]:
            return COLOR_RIGHT
        return COLOR_CENTER

    def draw(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        angles: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        在图像上绘制骨骼、关节点和角度标注

        Args:
            image: 目标图像（可为缩放后的图）
            keypoints: [17, 3] 图像坐标（与 image 同分辨率）
            angles: 关节角度字典

        Returns:
            绘制后的 BGR 图像
        """
        canvas = image.copy()
        valid = ~np.isnan(keypoints[:, 0])

        # 1. 绘制骨骼连线
        for i, j in SKELETON_CONNECTIONS:
            if valid[i] and valid[j]:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(canvas, pt1, pt2, COLOR_BONE, self.line_thickness, cv2.LINE_AA)

        # 2. 绘制关节点
        for i in range(17):
            if valid[i]:
                x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                color = self._joint_color(i)
                cv2.circle(canvas, (x, y), self.joint_radius, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), self.joint_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
                if self.show_indices:
                    cv2.putText(canvas, str(i), (x + 6, y - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # 3. 绘制角度信息面板
        if self.show_angles and angles:
            y_cursor = 25
            panel_items = []
            for name, val in angles.items():
                if val == val:  # 排除 NaN
                    panel_items.append(f"{name}: {val:.1f}°")

            # 计算面板尺寸
            max_w = 0
            for text in panel_items:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
                max_w = max(max_w, tw)

            # 绘制半透明背景
            panel_h = len(panel_items) * 22 + 10
            overlay = canvas.copy()
            cv2.rectangle(overlay, (8, 8), (14 + max_w, 10 + panel_h), COLOR_PANEL, -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

            # 绘制文字
            for text in panel_items:
                cv2.putText(canvas, text, (12, y_cursor),
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, COLOR_TEXT, 1, cv2.LINE_AA)
                y_cursor += 22

        return canvas