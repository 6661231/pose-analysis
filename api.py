import os
import shutil
import json
import logging
import asyncio
import base64
import threading
import time
import hashlib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ========== 全局单例（ lifespan 中初始化）==========
storage_mgr: VideoStorageManager
detector: YOLOv8PoseDetector
engine: PoseEngine
visualizer: PoseVisualizer

# 每个任务独立的报告生成器缓存
# key: task_id (由 VideoBatchProcessor 生成)
report_cache: dict = {}

# SSE 进度推送存储: {task_id: {"current": int, "total": int, "done": bool}}
progress_store: dict = {}
progress_store_lock = threading.Lock()


def build_pose_callback():
    """
    构建 skeleton_callback，兼容 VideoBatchProcessor 的接口
    利用 FramePacket.task_id 为每个视频维护独立的 PoseReportGenerator
    同时将进度推送到 progress_store 供 SSE 读取
    """
    def callback(pkt: FramePacket) -> Optional[np.ndarray]:
        # 1. YOLO 检测（在缩放后的图上进行，加速推理）
        kpts_scaled = detector.detect(pkt.image)
        if kpts_scaled is None:
            # 仍然更新进度
            with progress_store_lock:
                if pkt.task_id in progress_store:
                    progress_store[pkt.task_id]["current"] += 1
            return None

        # 2. 坐标还原到原始分辨率（用于报告和精确计算）
        kpts_orig = kpts_scaled.copy()
        if pkt.scale_factor != 1.0:
            kpts_orig[:, 0] *= pkt.scale_factor
            kpts_orig[:, 1] *= pkt.scale_factor

        # 3. 姿态计算
        angles = engine.compute_joint_angles(kpts_orig)
        offsets = engine.compute_limb_offsets(kpts_orig)

        # 4. 报告数据收集（按 task_id 隔离）
        if pkt.task_id not in report_cache:
            report_cache[pkt.task_id] = PoseReportGenerator()
        report_cache[pkt.task_id].add_frame(
            frame_idx=pkt.frame_idx,
            timestamp=pkt.timestamp_sec,
            angles=angles,
            offsets=offsets,
            keypoints=kpts_orig
        )

        # 5. 更新进度
        with progress_store_lock:
            if pkt.task_id in progress_store:
                progress_store[pkt.task_id]["current"] += 1

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
    version="1.0.0",
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


def process_video_background(temp_path: Path, stored, tid: str, file_id: str, orig_filename: str):
    """后台处理视频（在线程中运行）"""
    try:
        processor = VideoBatchProcessor(
            extractor=ParallelFrameExtractor(target_fps=1.5, max_dimension=640, strategy="balanced", max_workers=2),
            storage_manager=storage_mgr
        )
        # 预注册任务
        processor.register_task(stored.original_path, file_id)
        total_frames = 100
        with progress_store_lock:
            progress_store[tid] = {"current": 0, "total": total_frames, "done": False}

        callback = build_pose_callback()
        results = processor.process_batch(
            task_inputs=[(stored.original_path, file_id)],
            skeleton_callback=callback,
            output_dir=Path("./temp_skeletons")
        )
        task = list(results.values())[0]
        internal_tid = task.task_id

        # 同步进度和报告（callback 使用 internal_tid，需映射到 upload tid）
        with progress_store_lock:
            if internal_tid in progress_store:
                progress_store[tid] = dict(progress_store[internal_tid])
        if internal_tid in report_cache:
            report_cache[tid] = report_cache[internal_tid]

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
        if tid in report_cache:
            report_gen = report_cache[tid]
            report_file = storage_mgr.base_dir / "reports" / f"{file_id}_report.json"
            report_data = report_gen.save(report_file, video_info={
                "file_id": file_id, "original_filename": orig_filename,
                "original_path": str(stored.original_path),
                "source_fps": task.total_frames / (task.end_time - task.start_time) if (task.end_time and task.end_time > task.start_time) else 0
            })
            report_path = report_file
            del report_cache[tid]

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
                "summary": report_data.get("summary") if report_data else None
            }
    except Exception as e:
        logger.error(f"[Background] 处理失败 [{tid}]: {e}")
        with results_store_lock:
            results_store[tid] = {"task_id": tid, "file_id": file_id, "status": "failed", "message": str(e)}
    finally:
        with progress_store_lock:
            if tid in progress_store:
                progress_store[tid]["done"] = True
        temp_path.unlink(missing_ok=True)


@app.post("/upload", response_class=JSONResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频并异步启动分析流程（立即返回 task_id，前端通过 SSE 获取进度）
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, detail="仅支持 MP4/MOV/AVI/MKV 视频格式")

    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stored = storage_mgr.store_original_video(temp_path, custom_filename=file.filename)

    # 预注册任务获取 task_id
    tid = hashlib.md5(f"{stored.original_path}_{time.time()}".encode()).hexdigest()[:12]

    # 初始化进度
    with progress_store_lock:
        progress_store[tid] = {"current": 0, "total": 100, "done": False}

    # 异步后台处理
    threading.Thread(
        target=process_video_background,
        args=(temp_path, stored, tid, stored.file_id, file.filename),
        daemon=True
    ).start()

    return JSONResponse({"task_id": tid, "status": "processing", "message": "已开始分析"})


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
    with results_store_lock:
        del results_store[task_id]
    return result


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """
    查询任务状态（当前版本直接在上传接口返回结果，此接口预留用于异步查询）
    """
    return {"task_id": task_id, "note": "请通过 /upload 接口直接获取同步处理结果"}


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