# video-compact

视频压缩质量评估与自动参数筛选工具，支持命令行批量评估和 Streamlit 可视化分析。

## 功能

- 编码器：HEVC (`libx265`)、AV1 (`libsvtav1`)
- 指标：VMAF、PSNR-HVS、SSIM、MS-SSIM、SNR
- 统一入口：`main.py` 支持 `benchmark` / `autotune` / `web` 子命令
- 自动筛选：两阶段粗到细，按目标 VMAF 推荐最小体积参数

## 环境要求

- Python `>=3.12`
- 系统命令：`ffmpeg`、`ffprobe`、`vmaf`

## 安装依赖

推荐：

```bash
uv sync
```

可选（请在项目根目录执行）：

```bash
pip install -e .
```

## 使用方式

### 1) 运行评估

```bash
uv run python main.py benchmark video.mp4
uv run python main.py benchmark video.mp4 -e hevc av1 -s 5 -o results
uv run python main.py benchmark video.mp4 --no-strict
uv run python main.py benchmark video.mp4 --vmaf-threads 24 --vmaf-io-mode fifo
```

### 2) 自动筛选参数

```bash
uv run python main.py autotune sample1.mp4 sample2.mp4
uv run python main.py autotune sample1.mp4 sample2.mp4 --target-vmaf 95 --coarse-duration 10 --coarse-scale 1280 --jobs 2
uv run python main.py autotune sample1.mp4 sample2.mp4 --vmaf-threads 24 --vmaf-io-mode auto
```

默认策略：

- 目标 VMAF：`95`
- 粗扫网格：
  - HEVC: `22,26,30,34`
  - AV1: `30,36,42,48`
- 精扫范围：
  - HEVC: 粗扫最优 `±2`
  - AV1: 粗扫最优 `±3`
- VMAF 线程：默认取当前进程可用 CPU 核数（可用 `--vmaf-threads` 覆盖）
- VMAF I/O：默认 `auto`（优先 FIFO 管道，不落盘；失败自动回退 `file`）

### 3) 启动可视化

```bash
uv run python main.py web
uv run python main.py web --host 0.0.0.0 --port 8501
uv run python main.py web --no-server-headless
```

Web 页面包含三种模式：

- `查看结果`
- `运行评估`
- `自动筛选`（后台任务 + 进度日志 + 结果表）

## 结果输出

### benchmark

```text
results/
  <输入文件名>/
    results.json
    encoded/
    vmaf_*.json
```

### autotune

```text
results_autotune/
  run_<timestamp>/
    autotune_summary.json
    autotune_report.md
    <sample_dir>/
      coarse_clip.mp4
      <encoder>/coarse/...
      <encoder>/fine/...
```

`autotune_summary.json` 包含每视频每编码器推荐值、候选排名、是否达到阈值、推荐区间等信息。
