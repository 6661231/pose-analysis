# Pose Analysis System v4.0

一个本地运行的单目二维动作分析工具。系统使用 YOLOv8-Pose 提取人体关键点，对完整视频序列进行置信度过滤、短缺失插值和平滑，再生成动作报告与骨骼视频。

当前重点支持深蹲分析；通用模式只记录动作趋势，投掷模式仍属于实验功能。

## 能做什么

- 深蹲活动范围、左右协调、躯干控制、动作平滑和节奏分析
- 自动识别深蹲重复次数与最低点
- 输出关键点、二维关节角、动作阶段和统计数据
- 单独报告检测率、关键点完整度和数据可信度
- 生成骨骼标记视频、JSON 报告、历史趋势和 PDF

## 不能做什么

- 不能保证真实三维关节角误差小于 5 度
- 不能从普通视频直接计算真实力量、功率或关节负荷
- 不能替代医生、康复师或专业动作捕捉设备
- 通用模式的分数不能跨不同动作直接比较

## 环境要求

- Python 3.10-3.13（当前已在 Python 3.13.3 验证）
- Windows 10/11；macOS 和 Linux 可手动启动
- 建议 8 GB 以上内存
- NVIDIA 显卡可提升分析速度，没有显卡也可使用 CPU

首次安装依赖约需 2 GB，主要来自 PyTorch。

## 快速启动

Windows 可以双击 `run.bat`。也可以在项目目录运行：

```powershell
python -m pip install -r requirements.txt
python api.py
```

浏览器访问 `http://localhost:8080`。

`run.bat` 会优先复用电脑上已经安装的 Python 3.13 和依赖，不再创建虚拟环境。直接执行 `python api.py` 时，如果 PATH 仍指向旧 Python 3.8，程序也会自动切换到现有的 Python 3.13。

## 拍摄建议

深蹲视频建议满足：

- 全身和双脚始终在画面内
- 相机固定，避免手持晃动和变焦
- 相机大致位于髋部高度
- 分析膝髋角时优先使用正侧面
- 分析左右对称时优先使用正面
- 光线均匀，衣服不要遮住主要关节
- 视频原始帧率至少 10 FPS，建议 30 FPS

数据可信度低于 60 分时，应先改善拍摄条件，不应直接根据动作分数做判断。

## 分析模式

| 模式 | 分析内容 | 建议分析帧率 |
|------|----------|--------------|
| 深蹲 | 活动范围、左右协调、躯干控制、平滑度、节奏、重复一致性 | 10 FPS |
| 通用动作 | 二维角度和动作趋势 | 10 FPS |
| 投掷（实验性） | 当前仅提供通用趋势，不输出力量结论 | 20 FPS |

可以通过环境变量修改默认抽样帧率和上传限制：

```powershell
$env:POSE_TARGET_FPS="15"
$env:POSE_MAX_UPLOAD_MB="512"
python api.py
```

## 测试

安装 NumPy 后，可以运行不需要加载 YOLO 模型的算法测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖颈部角度定义、退化角处理、缺失点插值、严格 JSON 输出和深蹲周期识别。

## 主要模块

```text
api.py                FastAPI 接口、后台任务和进度管理
cv.py                 视频抽帧和骨骼视频生成
pose_detector.py      YOLOv8-Pose 检测与多人连续跟踪
pose_processing.py    置信度过滤、缺失插值、平滑和质量评分
pose_engine.py        二维几何角度与归一化偏移量
movement_analyzer.py  深蹲专项和通用动作评分
report_generator.py   时序聚合与 JSON 报告
visualizer.py         骨骼可视化
static/index.html     浏览器界面
tests/                算法回归测试
```

## 报告版本

v4 报告新增 `data_quality` 和 `movement_assessment`，并使用 `null` 表示缺失数值，不再使用 `0` 伪装缺失关键点。旧报告仍可在历史页面中预览，但旧评分不会自动变成新评分。
