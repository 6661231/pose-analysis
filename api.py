import os
import sys
import json
import logging
import asyncio
import threading
import time
import hashlib
import math
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime


def _ensure_compatible_python() -> None:
    """Use an installed modern Python before importing third-party packages."""
    if sys.version_info[:2] >= (3, 10):
        return

    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    candidates = []
    for version in (313, 312, 311, 310):
        candidates.extend([
            local_app_data / "Programs" / "Python" / f"Python{version}" / "python.exe",
            program_files / f"Python{version}" / "python.exe",
        ])

    for executable in candidates:
        if executable.is_file():
            os.execv(
                str(executable),
                [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
    raise RuntimeError("需要 Python 3.10 或更高版本，且未找到兼容解释器。")


_ensure_compatible_python()

import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 已有模块
from mp4file import VideoStorageManager
from cv import ParallelFrameExtractor, VideoBatchProcessor, FramePacket

# 新生成的模块
from pose_detector import YOLOv8PoseDetector
from pose_engine import PoseEngine
from visualizer import PoseVisualizer
from report_generator import PoseReportGenerator
from movement_analyzer import SUPPORTED_ACTIONS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ========== 全局单例（ lifespan 中初始化）==========
storage_mgr: VideoStorageManager
detector: YOLOv8PoseDetector
engine: PoseEngine
visualizer: PoseVisualizer

TARGET_ANALYSIS_FPS = float(np.clip(float(os.getenv("POSE_TARGET_FPS", "10")), 2.0, 30.0))
MAX_UPLOAD_BYTES = int(os.getenv("POSE_MAX_UPLOAD_MB", "1024")) * 1024 * 1024


def _analysis_fps(action_type: str) -> float:
    minimum = 20.0 if action_type == "throw" else 10.0
    return float(min(30.0, max(TARGET_ANALYSIS_FPS, minimum)))

# SSE 进度推送存储: {task_id: {"current": int, "total": int, "done": bool}}
progress_store: dict = {}
progress_store_lock = threading.Lock()


def _set_progress(task_id: str, value: float) -> None:
    with progress_store_lock:
        if task_id in progress_store:
            progress_store[task_id]["current"] = max(
                progress_store[task_id]["current"],
                min(99.0, float(value)),
            )


def _probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("上传的视频无法打开")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    duration = total_frames / source_fps if source_fps > 0 else 0.0
    return {
        "source_fps": round(source_fps, 3) if source_fps > 0 else 0.0,
        "source_total_frames": total_frames,
        "duration_sec": round(duration, 3),
        "width": width,
        "height": height,
    }


def build_pose_callback(
    task_id: str,
    report_generator: PoseReportGenerator,
    expected_frames: int,
):
    """
    构建 skeleton_callback，兼容 VideoBatchProcessor 的接口
    所有抽样帧都进入报告；未检测帧保留为空观察，不压缩时间轴。
    """
    processed_frames = 0

    def callback(pkt: FramePacket) -> Optional[np.ndarray]:
        nonlocal processed_frames
        # 1. YOLO 检测（在缩放后的图上进行，加速推理）
        kpts_scaled = detector.detect(pkt.image, tracking_id=task_id)
        if kpts_scaled is None:
            report_generator.add_frame(
                frame_idx=pkt.frame_idx,
                timestamp=pkt.timestamp_sec,
                keypoints=None,
            )
            processed_frames += 1
            _set_progress(task_id, 30 + 69 * processed_frames / max(expected_frames, 1))
            return pkt.image.copy()

        # 2. 坐标还原到原始分辨率（用于报告和精确计算）
        kpts_orig = kpts_scaled.copy()
        if pkt.scale_factor != 1.0:
            kpts_orig[:, 0] *= pkt.scale_factor
            kpts_orig[:, 1] *= pkt.scale_factor

        # 3. 即时姿态只用于可视化；正式报告会对整段序列清洗后重新计算
        angles = engine.compute_joint_angles(kpts_orig)

        # 4. 收集原始观察
        report_generator.add_frame(
            frame_idx=pkt.frame_idx,
            timestamp=pkt.timestamp_sec,
            keypoints=kpts_orig
        )

        # 5. 更新进度
        processed_frames += 1
        _set_progress(task_id, 30 + 69 * processed_frames / max(expected_frames, 1))

        # 6. 可视化绘制（在缩放图上绘制，后续由 VideoBatchProcessor resize 回原始尺寸）
        annotated = visualizer.draw(pkt.image, kpts_scaled, angles)
        return annotated

    return callback


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage_mgr, detector, engine, visualizer
    logger.info("[API] 正在初始化服务组件...")

    # 初始化存储、检测器、引擎、可视化器
    storage_mgr = VideoStorageManager(base_dir="./video_storage", retention_days=30)
    detector = YOLOv8PoseDetector(model_path="yolov8n-pose.pt")
    engine = PoseEngine()
    visualizer = PoseVisualizer(show_angles=True)
    

    logger.info("[API] 服务就绪")
    yield

    logger.info("[API] 服务关闭，清理资源")


app = FastAPI(
    title="Pose Analysis API",
    description="基于 YOLOv8-Pose 的人体姿态分析服务",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 后台处理结果缓存: {task_id: result_dict}
results_store: dict = {}
results_store_lock = threading.Lock()


def process_video_background(
    temp_path: Path,
    stored,
    tid: str,
    file_id: str,
    orig_filename: str,
    action_type: str,
):
    """后台处理视频（在线程中运行）"""
    try:
        video_info = _probe_video(stored.original_path)
        analysis_fps = _analysis_fps(action_type)
        expected_frames = max(
            1,
            int(math.ceil(video_info["duration_sec"] * analysis_fps)) + 1,
        )
        report_gen = PoseReportGenerator(action_type=action_type)
        processor = VideoBatchProcessor(
            extractor=ParallelFrameExtractor(
                target_fps=analysis_fps,
                max_dimension=640,
                strategy="accurate",
                max_workers=2,
            ),
            storage_manager=storage_mgr
        )
        with progress_store_lock:
            progress_store[tid] = {"current": 0.0, "total": 100.0, "done": False}

        callback = build_pose_callback(tid, report_gen, expected_frames)

        def extraction_progress(_internal_task_id: str, progress: float) -> None:
            _set_progress(tid, progress * 30)

        results = processor.process_batch(
            task_inputs=[(stored.original_path, file_id)],
            skeleton_callback=callback,
            progress_callback=extraction_progress,
            output_dir=Path("./temp_skeletons")
        )
        task = list(results.values())[0]

        # 归档骨骼视频
        skeleton_stored_path = None
        if task.status == "success" and task.output_skeleton_path:
            if task.output_skeleton_path.exists() and task.output_skeleton_path.stat().st_size > 0:
                skeleton_stored_path = storage_mgr.store_skeleton_video(
                    task.output_skeleton_path, related_file_id=file_id, overwrite=True
                )

        # 生成报告
        report_data = None
        report_path = None
        if report_gen.frames:
            report_file = storage_mgr.base_dir / "reports" / f"{file_id}_report.json"
            report_data = report_gen.save(report_file, video_info={
                "file_id": file_id,
                "original_filename": orig_filename,
                "original_path": str(stored.original_path),
                "action_type": action_type,
                **video_info,
            })
            report_path = report_file

        # 存结果
        with results_store_lock:
            results_store[tid] = {
                "task_id": tid, "file_id": file_id, "status": task.status,
                "message": "处理成功" if task.status == "success" else f"处理失败: {task.error_msg}",
                "original_video": str(stored.original_path),
                "skeleton_video": str(skeleton_stored_path) if skeleton_stored_path else None,
                "report_json": str(report_path) if report_path else None,
                "extracted_frames": task.extracted_frames,
                "processing_time_sec": round(task.end_time - task.start_time, 2) if task.end_time else None,
                "action_type": action_type,
                "summary": report_data.get("summary") if report_data else None
            }
    except Exception as e:
        logger.error(f"[Background] 处理失败 [{tid}]: {e}")
        with results_store_lock:
            results_store[tid] = {"task_id": tid, "file_id": file_id, "status": "failed", "message": str(e)}
    finally:
        detector.reset_tracking(tid)
        with progress_store_lock:
            if tid in progress_store:
                progress_store[tid]["current"] = 100.0
                progress_store[tid]["done"] = True
        temp_path.unlink(missing_ok=True)


@app.post("/upload", response_class=JSONResponse)
async def upload_video(
    file: UploadFile = File(...),
    action_type: str = Form("squat"),
):
    """
    上传视频并异步启动分析流程（立即返回 task_id，前端通过 SSE 获取进度）
    """
    safe_filename = Path(file.filename or "video.mp4").name
    if not safe_filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, detail="仅支持 MP4/MOV/AVI/MKV 视频格式")
    if action_type not in SUPPORTED_ACTIONS:
        raise HTTPException(400, detail="不支持的动作类型")

    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    upload_token = hashlib.md5(f"{safe_filename}_{time.time()}".encode()).hexdigest()[:12]
    temp_path = temp_dir / f"{upload_token}_{safe_filename}"
    written = 0
    try:
        with open(temp_path, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        detail=f"视频不能超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                    )
                output.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    try:
        _probe_video(temp_path)
        stored = storage_mgr.store_original_video(temp_path, custom_filename=safe_filename)
    except Exception as error:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(400, detail=f"视频校验失败: {error}") from error

    # 预注册任务获取 task_id
    tid = hashlib.md5(f"{stored.original_path}_{time.time()}".encode()).hexdigest()[:12]

    # 初始化进度
    with progress_store_lock:
        progress_store[tid] = {"current": 0, "total": 100, "done": False}

    # 异步后台处理
    threading.Thread(
        target=process_video_background,
        args=(temp_path, stored, tid, stored.file_id, safe_filename, action_type),
        daemon=True
    ).start()

    return JSONResponse({
        "task_id": tid,
        "status": "processing",
        "action_type": action_type,
        "analysis_fps": _analysis_fps(action_type),
        "message": "已开始分析",
    })


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取异步处理结果"""
    with results_store_lock:
        result = results_store.get(task_id)
    if not result:
        # 还没完成，返回处理中状态
        with progress_store_lock:
            p = progress_store.get(task_id, {})
        return {"status": "processing", "progress": p.get("current", 0), "total": p.get("total", 1), "done": p.get("done", False)}
    return result


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """查询后台任务状态。"""
    with results_store_lock:
        result = results_store.get(task_id)
    if result:
        return result
    with progress_store_lock:
        progress = progress_store.get(task_id)
    if not progress:
        raise HTTPException(404, detail="任务不存在")
    return {"task_id": task_id, "status": "processing", **progress}


@app.get("/download/skeleton/{file_id}")
async def download_skeleton(file_id: str):
    """下载骨骼标记视频"""
    _, skel = storage_mgr.get_pair_paths(file_id)
    if not skel or not skel.exists():
        raise HTTPException(404, detail="骨骼视频不存在")
    return FileResponse(
        str(skel),
        filename=f"{file_id}_skeleton.mp4",
        media_type="video/mp4"
    )


@app.get("/download/report/{file_id}")
async def download_report(file_id: str):
    """下载 JSON 分析报告"""
    report_file = storage_mgr.base_dir / "reports" / f"{file_id}_report.json"
    if not report_file.exists():
        raise HTTPException(404, detail="报告不存在")
    return FileResponse(
        str(report_file),
        filename=f"{file_id}_report.json",
        media_type="application/json"
    )


@app.get("/preview/report/{file_id}")
async def preview_report(file_id: str):
    """直接预览 JSON 报告内容"""
    report_file = storage_mgr.base_dir / "reports" / f"{file_id}_report.json"
    if not report_file.exists():
        raise HTTPException(404, detail="报告不存在")
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/files")
async def list_files():
    """列出所有已存储的分析记录"""
    storage_mgr.sync_from_disk()
    records = storage_mgr.list_all_records()
    reports_dir = storage_mgr.base_dir / "reports"
    items = []
    for r in records:
        report_file = reports_dir / f"{r.file_id}_report.json"
        items.append({
            "file_id": r.file_id,
            "timestamp": r.timestamp,
            "created_at": r.created_at,
            "original_path": str(r.original_path) if r.original_path else None,
            "has_skeleton": r.skeleton_path is not None and r.skeleton_path.exists(),
            "has_report": report_file.exists()
        })
    return JSONResponse({"total": len(items), "files": items})


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """删除指定记录及关联文件"""
    # 删除存储的视频文件
    deleted = storage_mgr.delete_by_file_id(file_id)
    # 同时删除报告
    report_file = storage_mgr.base_dir / "reports" / f"{file_id}_report.json"
    if report_file.exists():
        report_file.unlink()
        deleted = True
    if not deleted:
        raise HTTPException(404, detail="文件不存在")
    return {"status": "deleted", "file_id": file_id}


@app.delete("/files")
async def delete_all_files():
    """删除所有记录及关联文件"""
    storage_mgr.sync_from_disk()
    records = storage_mgr.list_all_records()
    count = 0
    for r in records:
        storage_mgr.delete_by_file_id(r.file_id)
        rpt = storage_mgr.base_dir / "reports" / f"{r.file_id}_report.json"
        if rpt.exists():
            rpt.unlink()
        count += 1
    return {"status": "deleted", "count": count}


@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    """SSE 端点：推送视频分析实时进度"""
    async def generate():
        # 初始化进度（如果还没有）
        with progress_store_lock:
            if task_id not in progress_store:
                progress_store[task_id] = {"current": 0, "total": 100, "done": False}
        while True:
            await asyncio.sleep(0.3)
            with progress_store_lock:
                p = progress_store.get(task_id, {"current": 0, "total": 100, "done": True})
                current, total, done = p["current"], p["total"], p["done"]
            progress = min(100, int(current / max(total, 1) * 100)) if total > 0 else 0
            yield f"data: {json.dumps({'progress': progress, 'current': current, 'total': total, 'done': done})}\n\n"
            if done:
                break
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector": "yolov8n-pose",
        "device": detector.device if detector else "unknown",
        "timestamp": datetime.now().isoformat()
    }


# 静态文件挂载（必须放在所有 API 路由之后）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
