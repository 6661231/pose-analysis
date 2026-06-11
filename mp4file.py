import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import hashlib


@dataclass
class StoredVideoInfo:
    """存储视频文件的元数据"""
    original_path: Path          # 原始上传视频路径
    skeleton_path: Optional[Path] = None  # 骨骼标记视频路径（生成后填充）
    timestamp: str = ""           # 时间戳标识
    file_id: str = ""            # 唯一文件ID
    created_at: float = 0.0       # 创建时间戳（秒）


class VideoStorageManager:
    """
    本地视频存储管理器
    - 按时间戳命名规则存储原始视频和骨骼标记视频
    - 自动管理目录结构
    - 支持文件清理和元数据追踪
    """
    
    def __init__(
        self,
        base_dir: str = "./video_storage",
        original_subdir: str = "original",
        skeleton_subdir: str = "skeleton",
        timestamp_format: str = "%Y%m%d_%H%M%S",
        retention_days: Optional[int] = 7
    ):
        """
        初始化存储管理器
        
        Args:
            base_dir: 存储根目录
            original_subdir: 原始视频子目录名
            skeleton_subdir: 骨骼标记视频子目录名
            timestamp_format: 时间戳格式，默认 年月日_时分秒
            retention_days: 自动清理天数，None 表示不自动清理
        """
        self.base_dir = Path(base_dir).resolve()
        self.original_dir = self.base_dir / original_subdir
        self.skeleton_dir = self.base_dir / skeleton_subdir
        self.timestamp_format = timestamp_format
        self.retention_days = retention_days
        
        # 确保目录存在
        self._ensure_directories()
        
        # 内存中维护的存储记录（可选，用于快速查询）
        self._storage_records: List[StoredVideoInfo] = []
    
    def _ensure_directories(self) -> None:
        """创建必要的目录结构"""
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_timestamp(self) -> str:
        """生成当前时间戳字符串"""
        return datetime.now().strftime(self.timestamp_format)
    
    def _generate_file_id(self, timestamp: str, original_filename: str) -> str:
        """
        生成唯一文件ID，基于时间戳+原文件名哈希
        确保同一秒内上传的多个文件不冲突
        """
        hash_input = f"{timestamp}_{original_filename}_{time.time()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{timestamp}_{short_hash}"
    
    def _generate_filename(self, file_id: str, suffix: str = "") -> str:
        """
        生成标准文件名
        格式: {file_id}{suffix}.mp4
        """
        suffix_str = f"_{suffix}" if suffix else ""
        return f"{file_id}{suffix_str}.mp4"
    
    def store_original_video(
        self, 
        source_path: Union[str, Path], 
        custom_filename: Optional[str] = None
    ) -> StoredVideoInfo:
        """
        存储用户上传的原始视频
        
        Args:
            source_path: 源视频文件路径（上传后的临时路径）
            custom_filename: 可选的自定义原始文件名（仅用于生成ID，不影响最终命名）
        
        Returns:
            StoredVideoInfo: 存储后的文件信息
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"源文件不存在: {source}")
        
        if not source.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}:
            # 虽然内部统一存为mp4，但接受常见视频格式
            pass
        
        # 生成时间戳和唯一ID
        timestamp = self._generate_timestamp()
        orig_name = custom_filename or source.name
        file_id = self._generate_file_id(timestamp, orig_name)
        
        # 目标路径: original/YYYYMMDD_HHMMSS_xxxxxx.mp4
        target_filename = self._generate_filename(file_id, "orig")
        target_path = self.original_dir / target_filename
        
        # 复制文件（保留原始文件不动，或者可以改为 move 如果源是临时文件）
        shutil.copy2(source, target_path)
        
        # 构建元数据
        info = StoredVideoInfo(
            original_path=target_path,
            timestamp=timestamp,
            file_id=file_id,
            created_at=time.time()
        )
        
        self._storage_records.append(info)
        return info
    
    def store_skeleton_video(
        self,
        source_path: Union[str, Path],
        related_file_id: str,
        overwrite: bool = False
    ) -> Path:
        """
        存储生成的骨骼标记视频
        
        Args:
            source_path: 骨骼标记视频源路径（通常是临时生成文件）
            related_file_id: 关联的原始视频 file_id（确保命名对应）
            overwrite: 是否覆盖已存在的骨骼视频
        
        Returns:
            Path: 存储后的目标路径
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"骨骼视频源文件不存在: {source}")
        
        # 目标路径: skeleton/YYYYMMDD_HHMMSS_xxxxxx_skeleton.mp4
        target_filename = self._generate_filename(related_file_id, "skeleton")
        target_path = self.skeleton_dir / target_filename
        
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"骨骼视频已存在: {target_path}，设置 overwrite=True 可覆盖")
        
        shutil.copy2(source, target_path)
        
        # 更新对应记录
        for record in self._storage_records:
            if record.file_id == related_file_id:
                record.skeleton_path = target_path
                break
        
        return target_path
    
    def get_pair_paths(self, file_id: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        通过 file_id 获取原始视频和骨骼视频的路径对
        
        Returns:
            (original_path, skeleton_path) 骨骼视频可能为 None
        """
        for record in self._storage_records:
            if record.file_id == file_id:
                return record.original_path, record.skeleton_path
        
        # 如果内存中没有，尝试从磁盘推断
        orig = self.original_dir / self._generate_filename(file_id, "orig")
        skel = self.skeleton_dir / self._generate_filename(file_id, "skeleton")
        
        orig_path = orig if orig.exists() else None
        skel_path = skel if skel.exists() else None
        
        return orig_path, skel_path
    
    def list_all_records(self) -> List[StoredVideoInfo]:
        """列出所有存储记录"""
        return self._storage_records.copy()
    
    def list_disk_files(self) -> dict:
        """
        扫描磁盘实际文件（与内存记录无关，用于同步检查）
        
        Returns:
            {"original": [...], "skeleton": [...]}
        """
        originals = sorted(self.original_dir.glob("*.mp4"))
        skeletons = sorted(self.skeleton_dir.glob("*.mp4"))
        return {
            "original": originals,
            "skeleton": skeletons
        }
    
    def get_storage_stats(self) -> dict:
        """获取存储统计信息"""
        orig_files = list(self.original_dir.glob("*.mp4"))
        skel_files = list(self.skeleton_dir.glob("*.mp4"))
        
        orig_size = sum(f.stat().st_size for f in orig_files) if orig_files else 0
        skel_size = sum(f.stat().st_size for f in skel_files) if skel_files else 0
        
        return {
            "original_count": len(orig_files),
            "skeleton_count": len(skel_files),
            "original_size_mb": round(orig_size / (1024 * 1024), 2),
            "skeleton_size_mb": round(skel_size / (1024 * 1024), 2),
            "total_size_mb": round((orig_size + skel_size) / (1024 * 1024), 2),
            "base_dir": str(self.base_dir)
        }
    
    def clean_old_files(self, days: Optional[int] = None) -> List[Path]:
        """
        清理超过指定天数的旧文件
        
        Args:
            days: 清理阈值（天），默认使用初始化时的 retention_days
        
        Returns:
            List[Path]: 被删除的文件列表
        """
        threshold_days = days or self.retention_days
        if threshold_days is None:
            return []
        
        cutoff = time.time() - (threshold_days * 86400)
        deleted = []
        
        for directory in [self.original_dir, self.skeleton_dir]:
            for file_path in directory.glob("*.mp4"):
                if file_path.stat().st_mtime < cutoff:
                    file_path.unlink()
                    deleted.append(file_path)
                    
                    # 同步清理内存记录
                    self._storage_records = [
                        r for r in self._storage_records 
                        if r.original_path != file_path and r.skeleton_path != file_path
                    ]
        
        return deleted
    
    def delete_by_file_id(self, file_id: str) -> bool:
        """
        通过 file_id 删除关联的原始视频和骨骼视频
        
        Returns:
            bool: 是否成功删除至少一个文件
        """
        deleted_any = False
        orig, skel = self.get_pair_paths(file_id)
        
        if orig and orig.exists():
            orig.unlink()
            deleted_any = True
        
        if skel and skel.exists():
            skel.unlink()
            deleted_any = True
        
        # 清理内存记录
        self._storage_records = [
            r for r in self._storage_records if r.file_id != file_id
        ]
        
        return deleted_any
    
    def sync_from_disk(self) -> None:
        """
        从磁盘同步文件到内存记录（服务重启后恢复状态）
        根据文件名解析 file_id 和时间戳
        """
        self._storage_records.clear()
        
        # 扫描原始视频
        orig_files = {f.stem.replace("_orig", ""): f for f in self.original_dir.glob("*.mp4")}
        skel_files = {f.stem.replace("_skeleton", ""): f for f in self.skeleton_dir.glob("*.mp4")}
        
        all_ids = set(orig_files.keys()) | set(skel_files.keys())
        
        for fid in all_ids:
            orig = orig_files.get(fid)
            skel = skel_files.get(fid)
            
            # 从 file_id 解析时间戳 (格式: YYYYMMDD_HHMMSS_hash)
            timestamp = fid[:15] if len(fid) >= 15 else fid
            
            record = StoredVideoInfo(
                original_path=orig,
                skeleton_path=skel,
                timestamp=timestamp,
                file_id=fid,
                created_at=orig.stat().st_mtime if orig else (skel.stat().st_mtime if skel else 0)
            )
            self._storage_records.append(record)




    