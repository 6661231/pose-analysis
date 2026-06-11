import math
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

# COCO 17 关键点定义（YOLOv8-Pose 标准输出）
# 注：YOLOv8-Pose 基于 COCO 数据集，标准输出为 17 个关键点，
# 覆盖全身主要关节，满足常规姿态分析需求。
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO 骨架连接对（用于可视化与偏移量计算）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 面部
    (5, 6),                               # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),      # 手臂
    (11, 12),                             # 髋
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿
    (5, 11), (6, 12)                      # 躯干
]

# 左右索引分组（用于对称性分析）
LEFT_KP_IDXS = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_KP_IDXS = [2, 4, 6, 8, 10, 12, 14, 16]


class YOLOv8PoseDetector:
    """
    YOLOv8-Pose 推理封装器
    - 自动适配 CPU / CUDA
    - 支持单帧与批量推理
    - 输出 [17, 3] 关键点数组 (x, y, confidence)
    - 若画面有多人，自动选择面积最大的人体
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        kpt_conf_threshold: float = 0.3,
        inference_size: int = 640
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.inference_size = inference_size

        if not Path(model_path).exists():
            logger.warning(f"[Detector] 本地未找到 {model_path}，将自动下载...")

        logger.info(f"[Detector] 加载模型: {model_path} -> 设备: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # 预热，消除首次推理延迟
        _ = self.model(torch.zeros(1, 3, inference_size, inference_size).to(self.device), verbose=False)
        logger.info("[Detector] 模型预热完成")

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        单帧推理

        Args:
            image: BGR 图像 (H, W, 3)，可为缩放后的图像

        Returns:
            np.ndarray: [17, 3]，每行 (x, y, confidence)
                       坐标为输入图像的绝对像素坐标
                       若未检测到人体，返回 None
        """
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        if not results or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or result.keypoints.data.shape[0] == 0:
            return None

        # keypoints.data: [num_persons, 17, 3]
        kpts_all = result.keypoints.data.cpu().numpy()

        # 若有多人，取检测框面积最大者
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))
            kpts = kpts_all[best_idx]
        else:
            kpts = kpts_all[0]

        # 低置信度关键点置 NaN，避免污染后续计算
        low_conf_mask = kpts[:, 2] < self.kpt_conf_threshold
        kpts[low_conf_mask, :2] = np.nan

        return kpts

    def detect_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量推理（利用 YOLO 内部 batch 优化）"""
        results = self.model(images, verbose=False, conf=self.conf_threshold)
        outputs = []
        for result in results:
            if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                outputs.append(None)
                continue
            kpts_all = result.keypoints.data.cpu().numpy()
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(np.argmax(areas))
                kpts = kpts_all[best_idx]
            else:
                kpts = kpts_all[0]
            kpts[kpts[:, 2] < self.kpt_conf_threshold, :2] = np.nan
            outputs.append(kpts)
        return outputs