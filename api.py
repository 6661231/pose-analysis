import os
import shutil
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
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


def build_pose_callback():
    """
    构建 skeleton_callback，兼容 VideoBatchProcessor 的接口
    利用 FramePacket.task_id 为每个视频维护独立的 PoseReportGenerator
    """
    def callback(pkt: FramePacket) -> Optional[np.ndarray]:
        # 1. YOLO 检测（在缩放后的图上进行，加速推理）
        kpts_scaled = detector.detect(pkt.image)
        if kpts_scaled is None:
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

        # 5. 可视化绘制（在缩放图上绘制，后续由 VideoBatchProcessor resize 回原始尺寸）
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


@app.post("/upload", response_class=JSONResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频并启动完整分析流程：
    存储 -> 并行抽帧 -> YOLO 检测 -> 角度计算 -> 骨骼视频合成 -> JSON 报告生成
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, detail="仅支持 MP4/MOV/AVI/MKV 视频格式")

    # 保存临时文件
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 存入归档存储层
    stored = storage_mgr.store_original_video(temp_path, custom_filename=file.filename)

    # 构建并行处理器
    extractor = ParallelFrameExtractor(
        target_fps=1.5,
        max_dimension=640,
        strategy="balanced",
        max_workers=2  # 根据服务器 CPU 核心数调整
    )
    processor = VideoBatchProcessor(extractor=extractor, storage_manager=storage_mgr)

    # 执行批量处理（单视频）
    callback = build_pose_callback()
    results = processor.process_batch(
        task_inputs=[(stored.original_path, stored.file_id)],
        skeleton_callback=callback,
        output_dir=Path("./temp_skeletons")
    )

    # 获取任务结果
    task = list(results.values())[0]
    tid = task.task_id

    # 归档骨骼视频
    skeleton_stored_path = None
    if task.status == "success" and task.output_skeleton_path:
        if task.output_skeleton_path.exists() and task.output_skeleton_path.stat().st_size > 0:
            skeleton_stored_path = storage_mgr.store_skeleton_video(
                task.output_skeleton_path,
                related_file_id=stored.file_id,
                overwrite=True
            )
        else:
            logger.warning(f"[API] 骨骼视频文件无效或为空: {task.output_skeleton_path}")

    # 生成并保存 JSON 报告
    report_path = None
    report_data = None
    if tid in report_cache:
        report_gen = report_cache[tid]
        report_file = storage_mgr.base_dir / "reports" / f"{stored.file_id}_report.json"
        report_data = report_gen.save(
            report_file,
            video_info={
                "file_id": stored.file_id,
                "original_filename": file.filename,
                "original_path": str(stored.original_path),
                "source_fps": task.total_frames / (task.end_time - task.start_time) if (task.end_time and task.end_time > task.start_time) else 0
            }
        )
        report_path = report_file
        # 清理缓存
        del report_cache[tid]

    # 清理临时上传文件
    temp_path.unlink(missing_ok=True)

    return {
        "task_id": tid,
        "file_id": stored.file_id,
        "status": task.status,
        "message": "处理成功" if task.status == "success" else f"处理失败: {task.error_msg}",
        "original_video": str(stored.original_path),
        "skeleton_video": str(skeleton_stored_path) if skeleton_stored_path else None,
        "report_json": str(report_path) if report_path else None,
        "extracted_frames": task.extracted_frames,
        "processing_time_sec": round(task.end_time - task.start_time, 2) if task.end_time else None,
        "summary": report_data.get("summary") if report_data else None
    }


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