# video-compact

视频压缩质量评估工具，支持命令行批量评估和 Streamlit 可视化分析。

## 功能

- 编码器：HEVC (`libx265`)、AV1 (`libsvtav1`)
- 指标：VMAF、PSNR-HVS、SSIM、MS-SSIM、SNR
- 统一入口：`main.py` 支持 `benchmark` / `web` 子命令

## 环境要求

- Python `>=3.12`
- 系统命令：`ffmpeg`、`ffprobe`、`vmaf`

## 安装依赖

```bash
uv sync
```

或：

```bash
pip install -e .
```

## 使用方式

### 1) 运行评估

```bash
python main.py benchmark video.mp4
```

常用参数：

```bash
python main.py benchmark video.mp4 -e hevc av1 -s 5 -o results
python main.py benchmark video.mp4 --no-strict
```

帮助：

```bash
python main.py benchmark --help
```

### 2) 启动可视化

```bash
python main.py web
```

指定地址和端口：

```bash
python main.py web --host 0.0.0.0 --port 8501
python main.py web --no-server-headless
```

帮助：

```bash
python main.py web --help
```

## 结果输出

默认输出目录结构：

```text
results/
  <输入文件名>/
    results.json
    encoded/
    vmaf_*.json
```

其中 `results.json` 包含每组参数的编码耗时、文件大小、压缩比和各项质量指标。
