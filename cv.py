import cv2
import time
import logging
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Callable, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
from mp4file import VideoStorageManager

logger = logging.getLogger(__name__)




@dataclass
class FramePacket:
    """抽帧数据包（同前，增加任务关联）"""
    image: np.ndarray
    frame_idx: int
    timestamp_sec: float
    source_fps: float
    scale_factor: float = 1.0
    task_id: str = ""            # 所属任务ID
    landmarks: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None


@dataclass
class VideoTask:
    """单个视频处理任务"""
    task_id: str                      # 唯一任务ID
    input_path: Path                  # 原始视频路径
    file_id: str                      # 存储层 file_id（关联原始视频）
    status: str = "pending"           # pending / running / success / failed
    progress: float = 0.0             # 0.0 ~ 1.0
    extracted_frames: int = 0         # 已抽帧数
    total_frames: int = 0             # 总帧数（预读）
    output_skeleton_path: Optional[Path] = None
    error_msg: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    
    # 内部用：存储该视频所有抽帧结果（若内存允许），或流式处理
    _frame_buffer: List[FramePacket] = field(default_factory=list, repr=False)


# ==================== 并行抽帧器 ====================

class ParallelFrameExtractor:
    """
    多线程视频抽帧器
    每个线程持有独立的 cv2.VideoCapture，避免 GIL 竞争和线程安全问题
    """

    def __init__(
        self,
        target_fps: float = 1.5,
        max_dimension: Optional[int] = 640,
        strategy: str = "balanced",
        max_workers: int = 4            # 并发解码线程数
    ):
        self.target_fps = target_fps
        self.max_dimension = max_dimension
        self.strategy = strategy
        self.max_workers = max_workers
        
        # 线程锁仅用于日志/状态更新，不用于解码本身
        self._lock = threading.Lock()

    def _open_capture(self, path: Path) -> Tuple[cv2.VideoCapture, Any]:
        """在线程内部打开视频"""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开: {path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, (fps, total, width, height)

    def _smart_resize(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.max_dimension is None:
            return frame, 1.0
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dimension:
            return frame, 1.0
        scale = self.max_dimension / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR), 1.0 / scale

    def _extract_single(
        self,
        task: VideoTask,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[FramePacket]:
        """
        单个视频的抽帧逻辑（运行在线程池中）
        返回该视频所有抽帧结果
        """
        task.status = "running"
        task.start_time = time.perf_counter()
        
        cap, (fps, total, width, height) = self._open_capture(task.input_path)
        task.total_frames = total
        
        effective_target = min(self.target_fps, fps) if fps > 0 else self.target_fps
        interval_ms = 1000.0 / effective_target
        step_approx = max(1, int(round(fps / effective_target))) if fps > 0 else 1
        
        next_target_ms = 0.0
        frame_idx = 0
        extracted = 0
        packets: List[FramePacket] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if current_ms <= 0 and fps > 0:
                    current_ms = (frame_idx / fps) * 1000.0

                should_extract = False

                if self.strategy == "fast":
                    if frame_idx % step_approx == 0:
                        should_extract = True
                elif self.strategy == "accurate":
                    if current_ms >= next_target_ms:
                        should_extract = True
                        next_target_ms += interval_ms
                        while next_target_ms <= current_ms:
                            next_target_ms += interval_ms
                else:  # balanced
                    if frame_idx % step_approx == 0:
                        expected_ms = extracted * interval_ms
                        if abs(current_ms - expected_ms) <= (interval_ms * 0.5):
                            should_extract = True
                        elif current_ms >= expected_ms:
                            should_extract = True
                            next_target_ms = current_ms + interval_ms

                if should_extract:
                    resized, scale = self._smart_resize(frame)
                    pkt = FramePacket(
                        image=resized,
                        frame_idx=frame_idx,
                        timestamp_sec=current_ms / 1000.0,
                        source_fps=fps,
                        scale_factor=scale,
                        task_id=task.task_id
                    )
                    packets.append(pkt)
                    extracted += 1

                frame_idx += 1
                
                # 每 30 帧报告一次进度（减少锁竞争）
                if frame_idx % 30 == 0 and progress_callback:
                    progress = frame_idx / total if total > 0 else 0
                    with self._lock:
                        task.progress = progress
                    progress_callback(task.task_id, progress)

        finally:
            cap.release()
            task.progress = 1.0
            task.extracted_frames = extracted
            task.end_time = time.perf_counter()
            
        return packets


# ==================== 批量处理器 ====================

class VideoBatchProcessor:
    """
    多视频并行处理编排器
    职责：并发抽帧 -> 骨骼检测 -> 生成标记视频 -> 存入存储层
    """

    def __init__(
        self,
        extractor: ParallelFrameExtractor,
        storage_manager: Optional[Any] = None,   # 传入 VideoStorageManager 实例
        pose_detector: Optional[Any] = None      # 骨骼检测模型实例（线程安全或每个线程独立）
    ):
        self.extractor = extractor
        self.storage = storage_manager
        self.pose_detector = pose_detector
        
        # 任务注册表
        self._tasks: Dict[str, VideoTask] = {}
        self._lock = threading.Lock()

    def register_task(self, input_path: Path, file_id: str) -> str:
        """注册一个待处理视频"""
        task_id = hashlib.md5(f"{input_path}_{time.time()}".encode()).hexdigest()[:12]
        task = VideoTask(
            task_id=task_id,
            input_path=Path(input_path),
            file_id=file_id
        )
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[VideoTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[VideoTask]:
        with self._lock:
            return list(self._tasks.values())

    def process_batch(
        self,
        task_inputs: List[Tuple[Path, str]],  # [(video_path, file_id), ...]
        skeleton_callback: Optional[Callable[[FramePacket], Optional[np.ndarray]]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, VideoTask]:
        """
        批量并行处理入口
        
        Args:
            task_inputs: 待处理视频列表，每项为 (路径, file_id)
            skeleton_callback: 骨骼检测回调，接收 FramePacket，返回标注图（None 则跳过该帧）
                               若不提供，仅做抽帧，不生成骨骼视频
            progress_callback: 进度回调 (task_id, progress_ratio)
            output_dir: 骨骼视频临时输出目录（默认系统 temp）
        
        Returns:
            {task_id: VideoTask} 所有任务结果
        """
        # 1. 注册所有任务
        task_map: Dict[str, VideoTask] = {}
        for path, fid in task_inputs:
            tid = self.register_task(path, fid)
            task_map[tid] = self._tasks[tid]

        # 2. 并行抽帧阶段
        logger.info(f"[Batch] 启动并行抽帧: {len(task_inputs)} 个视频, "
                   f"并发数={self.extractor.max_workers}")
        
        with ThreadPoolExecutor(max_workers=self.extractor.max_workers) as exe:
            future_to_tid = {
                exe.submit(
                    self.extractor._extract_single,
                    task,
                    progress_callback
                ): tid
                for tid, task in task_map.items()
            }

            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                task = task_map[tid]
                try:
                    packets = future.result()
                    task._frame_buffer = packets
                    logger.info(f"[Batch] 抽帧完成 [{tid}]: {len(packets)} 帧, "
                               f"耗时 {task.end_time - task.start_time:.2f}s")
                except Exception as e:
                    task.status = "failed"
                    task.error_msg = str(e)
                    logger.error(f"[Batch] 抽帧失败 [{tid}]: {e}")

        # 3. 骨骼检测 + 视频合成阶段（若提供回调）
        if skeleton_callback:
            self._generate_skeleton_videos(task_map, skeleton_callback, output_dir)
        
        return task_map

    def _generate_skeleton_videos(
        self,
        task_map: Dict[str, VideoTask],
        skeleton_callback: Callable[[FramePacket], Optional[np.ndarray]],
        output_dir: Optional[Path]
    ):
        """
        将抽帧结果逐帧做骨骼检测，并合成为 MP4
        此阶段也可并行化（若骨骼模型线程安全），这里默认单线程顺序处理每个视频
        """
        out_dir = output_dir or Path("./temp_skeletons")
        out_dir.mkdir(parents=True, exist_ok=True)

        for tid, task in task_map.items():
            if task.status == "failed" or not task._frame_buffer:
                continue

            # 预读原始视频信息以确定输出尺寸
            cap = cv2.VideoCapture(str(task.input_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            packet_times = np.array(
                [packet.timestamp_sec for packet in task._frame_buffer],
                dtype=np.float64,
            )
            positive_intervals = np.diff(packet_times)
            positive_intervals = positive_intervals[positive_intervals > 0]
            output_fps = (
                float(1.0 / np.median(positive_intervals))
                if len(positive_intervals)
                else min(self.extractor.target_fps, fps if fps > 0 else self.extractor.target_fps)
            )

            # 输出路径：使用 file_id 命名，便于后续存入 StorageManager
            out_path = out_dir / f"{task.file_id}_skeleton.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
            need_convert = False
            if not writer.isOpened():
                logger.warning(f"[Batch] avc1 不可用，回退 mp4v+ffmpeg 转换")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
                need_convert = True
            if not writer.isOpened():
                task.status = "failed"
                task.error_msg = "当前 OpenCV 环境没有可用的 MP4 视频编码器"
                task.end_time = time.perf_counter()
                logger.error(f"[Batch] 无法创建输出视频 [{tid}]: {out_path}")
                writer.release()
                continue

            processed = 0
            try:
                for pkt in task._frame_buffer:
                    annotated = skeleton_callback(pkt)
                    if annotated is not None:
                        # 若回调返回的是缩放后的图，需还原到原始尺寸
                        if annotated.shape[1] != width or annotated.shape[0] != height:
                            annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_LINEAR)
                        writer.write(annotated)
                        processed += 1

                task.output_skeleton_path = out_path
                task.status = "success"
                task.end_time = time.perf_counter()
                logger.info(f"[Batch] 骨骼视频生成 [{tid}]: {out_path}, "
                           f"有效帧 {processed}/{len(task._frame_buffer)}")
            except Exception as e:
                task.status = "failed"
                task.error_msg = f"骨骼生成失败: {e}"
                task.end_time = time.perf_counter()
                logger.error(f"[Batch] 骨骼生成失败 [{tid}]: {e}")
            finally:
                writer.release()

            # 若 mp4v 不兼容浏览器，用 ffmpeg 转 H.264
            if need_convert:
                import subprocess, shutil
                tmp_path = out_path.with_suffix('.tmp.mp4')
                try:
                    shutil.move(str(out_path), str(tmp_path))
                    subprocess.run(
                        ['ffmpeg', '-i', str(tmp_path), '-c:v', 'libx264', '-crf', '28', '-preset', 'fast', '-movflags', '+faststart', '-y', str(out_path)],
                        capture_output=True, timeout=60
                    )
                    tmp_path.unlink(missing_ok=True)
                    logger.info(f"[Batch] ffmpeg 转换完成: {out_path}")
                except Exception as e2:
                    logger.warning(f"[Batch] ffmpeg 转换失败: {e2}, 保留原始格式")
                    if tmp_path.exists():
                        shutil.move(str(tmp_path), str(out_path))

            # 释放内存
            task._frame_buffer.clear()


# ==================== 完整集成示例 ====================

def demo_batch_pipeline():
    """
    完整演示：批量上传 -> 并行抽帧 -> 骨骼检测 -> 存储结果
    """
  # 之前的存储类
    
    # 1. 初始化组件
    storage = VideoStorageManager(base_dir="./video_storage")
    extractor = ParallelFrameExtractor(
        target_fps=1.5,
        max_dimension=640,
        strategy="balanced",
        max_workers=4          # 同时解码 4 个视频
    )
    processor = VideoBatchProcessor(extractor=extractor, storage_manager=storage)

    # 2. 模拟批量上传（实际中来自 HTTP 上传接口）
    uploaded_files = [
        "./uploads/video_01.mp4",
        "./uploads/video_02.mp4",
        "./uploads/video_03.mp4",
        "./uploads/video_04.mp4",
    ]
    
    # 存入存储层并收集 (path, file_id)
    task_inputs: List[Tuple[Path, str]] = []
    for f in uploaded_files:
        info = storage.store_original_video(f)
        task_inputs.append((info.original_path, info.file_id))

    # 3. 定义骨骼检测回调（MediaPipe Pose 示例）
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

    def detect_skeleton(pkt: FramePacket) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return None
        
        annotated = pkt.image.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        return annotated

    # 4. 进度回调（可推送到前端 WebSocket）
    def on_progress(task_id: str, progress: float):
        pct = int(progress * 100)
        if pct % 10 == 0:  # 每 10% 打印一次
            print(f"  >> [{task_id}] 进度: {pct}%")

    # 5. 执行批量处理
    results = processor.process_batch(
        task_inputs=task_inputs,
        skeleton_callback=detect_skeleton,
        progress_callback=on_progress,
        output_dir=Path("./temp_skeletons")
    )

    # 6. 将生成的骨骼视频存入存储层
    for tid, task in results.items():
        if task.status == "success" and task.output_skeleton_path:
            storage.store_skeleton_video(
                task.output_skeleton_path,
                related_file_id=task.file_id,
                overwrite=True
            )
            print(f"[完成] {task.file_id} -> {task.output_skeleton_path}")

    # 7. 汇总报告
    success = sum(1 for t in results.values() if t.status == "success")
    failed = sum(1 for t in results.values() if t.status == "failed")
    print(f"\n处理完成: 成功 {success}, 失败 {failed}")


# ==================== 进阶：流式处理（低内存模式）====================

class StreamingBatchProcessor:
    """
    超大规模视频处理（单视频数小时）的流式并行方案
    不缓存全部帧，抽一帧处理一帧，内存占用恒定
    """

    def __init__(self, extractor: ParallelFrameExtractor):
        self.extractor = extractor
        self._queue: Queue = Queue(maxsize=16)  # 帧缓冲队列，背压控制

    def _producer(self, task: VideoTask):
        """生产者：抽帧入队"""
        packets = self.extractor._extract_single(task)
        for pkt in packets:
            self._queue.put(pkt)
        self._queue.put(None)  # 结束标记

    def _consumer(
        self,
        task: VideoTask,
        skeleton_callback: Callable[[FramePacket], Optional[np.ndarray]],
        writer: cv2.VideoWriter
    ):
        """消费者：出队检测写入"""
        while True:
            pkt = self._queue.get()
            if pkt is None:
                break
            annotated = skeleton_callback(pkt)
            if annotated is not None:
                writer.write(annotated)

    def process_streaming(
        self,
        task: VideoTask,
        skeleton_callback: Callable[[FramePacket], Optional[np.ndarray]],
        output_path: Path
    ):
        """单视频流式处理（生产者-消费者模式）"""
        cap = cv2.VideoCapture(str(task.input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.extractor.target_fps, (w, h))

        t_prod = threading.Thread(target=self._producer, args=(task,))
        t_cons = threading.Thread(target=self._consumer, args=(task, skeleton_callback, writer))

        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()
        writer.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # 运行完整演示
    # demo_batch_pipeline()
    
    # 快速测试单视频并行抽帧
    # extractor = ParallelFrameExtractor(target_fps=1.5, max_workers=2)
    # task = VideoTask(task_id="test", input_path=Path("test.mp4"), file_id="fid123")
    # packets = extractor._extract_single(task, progress_callback=lambda tid, p: print(f"{tid}: {p:.0%}"))
    # print(f"共抽帧 {len(packets)} 张")
