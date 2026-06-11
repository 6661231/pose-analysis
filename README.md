# Pose Analysis System v2.0

基于 YOLOv8-Pose 的人体姿态智能分析系统。上传视频 → 自动骨骼检测 → 六维能力评估 → AI 运动处方。

## 环境要求

- **Python 3.10+**（推荐 3.13）
- **Windows 10/11**（Mac/Linux 也可，但 run.bat 不适用）
- 首次运行需要联网下载依赖（约 2GB，含 PyTorch）

## 快速启动（Windows）

1. 解压项目文件夹
2. 双击 **`run.bat`**
3. 浏览器打开 **`http://localhost:8080`**

`run.bat` 会自动检查并安装缺失的依赖。

## 手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python api.py

# 3. 浏览器打开
# http://localhost:8080
```

> 如果 pip 下载慢，用国内镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

## 使用说明

| 功能 | 操作 |
|------|------|
| **上传分析** | 拖拽/点击上传 .mp4/.mov/.avi/.mkv 视频，自动完成姿态分析 |
| **六维能力图** | 对称性、关节活动度、核心稳定、平衡能力、动作流畅、姿态控制 |
| **AI 运动处方** | 低分维度自动生成训练建议（带组数/次数） |
| **历史趋势** | 多次分析结果对比折线图，追踪进步 |
| **导出 PDF** | 一键下载含雷达图和处方的完整报告 |
| **重新分析** | 结果页底部点击，可立即分析新视频 |

## 项目文件说明

```
├── api.py              # FastAPI 后端主入口
├── pose_detector.py    # YOLOv8-Pose 检测器
├── pose_engine.py      # 关节角度 / 偏移量计算
├── visualizer.py       # 骨骼可视化绘制
├── report_generator.py # JSON 报告生成
├── cv.py               # 并行抽帧 / 批量处理
├── mp4file.py          # 视频文件存储管理
├── static/
│   └── index.html      # 前端页面
├── requirements.txt    # Python 依赖清单
├── run.bat             # Windows 一键启动脚本
└── yolov8n-pose.pt     # YOLOv8 模型权重（可选，缺失时自动下载）
```

## 常见问题

**Q: 启动报错 `ModuleNotFoundError: No module named 'xxx'`**
A: 运行 `pip install -r requirements.txt` 安装依赖。

**Q: 浏览器打开 `http://localhost:8080` 是空白页**
A: 确认终端显示 `Uvicorn running on http://0.0.0.0:8080`，再刷新浏览器。

**Q: 上传视频后处理很慢**
A: 默认用 CPU 推理，有 NVIDIA 显卡会自动切换 CUDA 加速。

**Q: 端口 8080 被占用**
A: 修改 `api.py` 最后一行 `port=8080` 为其他端口，比如 `port=9090`。
