#!/usr/bin/env python3
"""
视频质量评估工具 - Web 界面

使用 Streamlit 构建的交互式界面，支持：
- 配置编码参数
- 运行评估
- 展示结果曲线图
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from benchmark import (
    ENCODERS,
    BenchmarkResult,
    load_results,
    run_single_benchmark,
    save_results,
)


st.set_page_config(
    page_title="视频质量评估工具",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 视频质量评估工具")
st.markdown("自动化视频编码和 VMAF 质量评估")


def get_existing_results() -> list[str]:
    """获取已有的结果文件"""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    
    result_files = []
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            results_json = subdir / "results.json"
            if results_json.exists():
                result_files.append(str(results_json))
    
    return result_files


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """将结果转换为 DataFrame"""
    data = []
    for r in results:
        encoder_config = ENCODERS.get(r.encoder, None)
        encoder_name = encoder_config.name if encoder_config else r.encoder
        row = {
            "编码器": encoder_name,
            "编码器ID": r.encoder,
            "参数名": r.param_name,
            "参数值": r.param_value,
            # VMAF 指标
            "VMAF 平均": r.vmaf_mean,
            "VMAF 最小": r.vmaf_min,
            "VMAF 最大": r.vmaf_max,
            # PSNR-HVS 指标
            "PSNR-HVS 平均": getattr(r, 'psnr_hvs_mean', 0),
            "PSNR-HVS 最小": getattr(r, 'psnr_hvs_min', 0),
            "PSNR-HVS 最大": getattr(r, 'psnr_hvs_max', 0),
            # SSIM 指标
            "SSIM 平均": getattr(r, 'ssim_mean', 0),
            "SSIM 最小": getattr(r, 'ssim_min', 0),
            "SSIM 最大": getattr(r, 'ssim_max', 0),
            # MS-SSIM 指标
            "MS-SSIM 平均": getattr(r, 'ms_ssim_mean', 0),
            "MS-SSIM 最小": getattr(r, 'ms_ssim_min', 0),
            "MS-SSIM 最大": getattr(r, 'ms_ssim_max', 0),
            # SNR 指标
            "SNR 平均 (dB)": getattr(r, 'snr_mean', 0),
            "SNR 最小 (dB)": getattr(r, 'snr_min', 0),
            "SNR 最大 (dB)": getattr(r, 'snr_max', 0),
            # 基本信息
            "文件大小 (MB)": r.output_size_mb,
            "压缩比 (%)": r.compression_ratio,
            "编码耗时 (秒)": r.encode_time_seconds,
        }
        data.append(row)
    return pd.DataFrame(data)


# 定义可用的指标选项
METRIC_OPTIONS = {
    "VMAF": {"col": "VMAF 平均", "min_col": "VMAF 最小", "max_col": "VMAF 最大", "range": [0, 100], "format": ".2f"},
    "PSNR-HVS": {"col": "PSNR-HVS 平均", "min_col": "PSNR-HVS 最小", "max_col": "PSNR-HVS 最大", "range": None, "format": ".2f"},
    "SSIM": {"col": "SSIM 平均", "min_col": "SSIM 最小", "max_col": "SSIM 最大", "range": [0, 1], "format": ".4f"},
    "MS-SSIM": {"col": "MS-SSIM 平均", "min_col": "MS-SSIM 最小", "max_col": "MS-SSIM 最大", "range": [0, 1], "format": ".4f"},
    "SNR": {"col": "SNR 平均 (dB)", "min_col": "SNR 最小 (dB)", "max_col": "SNR 最大 (dB)", "range": None, "format": ".2f"},
}


def plot_metric_vs_param(df: pd.DataFrame, metric_name: str = "VMAF"):
    """
    绘制指标 vs 参数值 曲线图
    
    Args:
        df: 数据框
        metric_name: 指标名称 (VMAF, PSNR-HVS, SSIM, MS-SSIM, SNR)
    """
    metric_config = METRIC_OPTIONS.get(metric_name, METRIC_OPTIONS["VMAF"])
    col = metric_config["col"]
    min_col = metric_config["min_col"]
    max_col = metric_config["max_col"]
    y_range = metric_config["range"]
    
    fig = px.line(
        df,
        x="参数值",
        y=col,
        color="编码器",
        markers=True,
        title=f"质量参数 vs {metric_name} 分数",
        labels={"参数值": "质量参数值", col: f"{metric_name} 分数"},
    )
    
    # 添加误差范围（如果列存在且有有效值）
    if min_col in df.columns and max_col in df.columns:
        for encoder in df["编码器"].unique():
            encoder_df = df[df["编码器"] == encoder]
            if encoder_df[min_col].sum() > 0:  # 只在有数据时添加
                fig.add_trace(go.Scatter(
                    x=encoder_df["参数值"],
                    y=encoder_df[min_col],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=encoder_df["参数值"],
                    y=encoder_df[max_col],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.1)",
                    showlegend=False,
                    hoverinfo="skip",
                ))
    
    layout_update = {
        "xaxis_title": "质量参数值 (crf)",
        "yaxis_title": f"{metric_name} 分数",
        "legend_title": "编码器",
    }
    if y_range:
        layout_update["yaxis_range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig


def plot_vmaf_vs_param(df: pd.DataFrame):
    """绘制 VMAF vs 参数值 曲线图（向后兼容）"""
    return plot_metric_vs_param(df, "VMAF")


def plot_vmaf_vs_size(df: pd.DataFrame):
    """绘制 VMAF vs 文件大小 曲线图"""
    fig = px.scatter(
        df,
        x="文件大小 (MB)",
        y="VMAF 平均",
        color="编码器",
        size="压缩比 (%)",
        hover_data=["参数值", "压缩比 (%)"],
        title="文件大小 vs VMAF 分数",
    )
    
    # 为每个编码器添加连线
    for encoder in df["编码器"].unique():
        encoder_df = df[df["编码器"] == encoder].sort_values("文件大小 (MB)")
        fig.add_trace(go.Scatter(
            x=encoder_df["文件大小 (MB)"],
            y=encoder_df["VMAF 平均"],
            mode="lines",
            line=dict(dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
    
    fig.update_layout(
        xaxis_title="文件大小 (MB)",
        yaxis_title="VMAF 分数",
        yaxis_range=[0, 100],
        legend_title="编码器",
    )
    
    return fig


def plot_compression_efficiency(df: pd.DataFrame):
    """绘制压缩效率图 (VMAF / 压缩比)"""
    df_copy = df.copy()
    df_copy["效率"] = df_copy["VMAF 平均"] / df_copy["压缩比 (%)"]
    
    fig = px.bar(
        df_copy,
        x="参数值",
        y="效率",
        color="编码器",
        barmode="group",
        title="压缩效率 (VMAF / 压缩比)",
        labels={"效率": "效率分数", "参数值": "质量参数值"},
    )
    
    fig.update_layout(
        xaxis_title="质量参数值",
        yaxis_title="效率分数 (越高越好)",
        legend_title="编码器",
    )
    
    return fig


def plot_quadrant(df: pd.DataFrame):
    """
    绘制四象限图：VMAF 分数 vs 压缩比
    
    - X 轴：压缩比（%），越低越好（文件越小）
    - Y 轴：VMAF 分数，越高越好
    - 理想区域：左上角（低压缩比 + 高 VMAF）
    """
    # 计算分割线的阈值
    vmaf_threshold = 90  # VMAF 90 分作为高质量阈值
    compression_threshold = df["压缩比 (%)"].median()  # 使用中位数作为压缩比阈值
    
    # 创建标签列，用于显示参数值
    df_copy = df.copy()
    df_copy["标签"] = df_copy.apply(
        lambda r: f"{r['参数名']}={r['参数值']}", axis=1
    )
    
    # 创建散点图
    fig = px.scatter(
        df_copy,
        x="压缩比 (%)",
        y="VMAF 平均",
        color="编码器",
        text="标签",
        size_max=15,
        hover_data={
            "参数值": True,
            "文件大小 (MB)": ":.2f",
            "VMAF 最小": ":.2f",
            "VMAF 最大": ":.2f",
            "标签": False,
        },
        title="四象限图：质量 vs 压缩比",
    )
    
    # 调整文本位置
    fig.update_traces(
        textposition="top center",
        textfont_size=10,
        marker=dict(size=12),
    )
    
    # 获取坐标轴范围
    x_min, x_max = df_copy["压缩比 (%)"].min(), df_copy["压缩比 (%)"].max()
    x_padding = (x_max - x_min) * 0.1
    
    # 添加水平分割线（VMAF 阈值）
    fig.add_hline(
        y=vmaf_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"VMAF {vmaf_threshold}",
        annotation_position="right",
    )
    
    # 添加垂直分割线（压缩比阈值）
    fig.add_vline(
        x=compression_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"压缩比 {compression_threshold:.1f}%",
        annotation_position="top",
    )
    
    # 添加四象限标注
    annotations = [
        # 左上：最佳区域
        dict(
            x=x_min + x_padding,
            y=95,
            text="✅ 最佳<br>(高质量+高压缩)",
            showarrow=False,
            font=dict(size=12, color="green"),
            bgcolor="rgba(0,255,0,0.1)",
        ),
        # 右上：高质量但文件大
        dict(
            x=x_max - x_padding,
            y=95,
            text="⚠️ 质量好但文件大",
            showarrow=False,
            font=dict(size=12, color="orange"),
            bgcolor="rgba(255,165,0,0.1)",
        ),
        # 左下：压缩好但质量差
        dict(
            x=x_min + x_padding,
            y=75,
            text="⚠️ 文件小但质量差",
            showarrow=False,
            font=dict(size=12, color="orange"),
            bgcolor="rgba(255,165,0,0.1)",
        ),
        # 右下：最差区域
        dict(
            x=x_max - x_padding,
            y=75,
            text="❌ 最差<br>(低质量+大文件)",
            showarrow=False,
            font=dict(size=12, color="red"),
            bgcolor="rgba(255,0,0,0.1)",
        ),
    ]
    
    fig.update_layout(
        xaxis_title="压缩比 (%) - 越低越好 ←",
        yaxis_title="VMAF 分数 - 越高越好 ↑",
        yaxis_range=[min(60, df_copy["VMAF 平均"].min() - 5), 100],
        legend_title="编码器",
        annotations=annotations,
    )
    
    return fig


AUTOTUNE_TASK_KEY = "autotune_task"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def discover_videos(directory: str) -> list[str]:
    path = Path(directory).expanduser()
    if not path.exists() or not path.is_dir():
        return []
    videos = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(str(p) for p in videos)


def tail_file(path: str, lines: int = 80) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        content = f.readlines()
    return "".join(content[-lines:])


def extract_marker(path: str, marker: str) -> str | None:
    text = tail_file(path, lines=200)
    for line in reversed(text.splitlines()):
        if line.startswith(marker):
            return line.split(":", 1)[1].strip()
    return None


def get_existing_autotune_summaries(output_root: str) -> list[str]:
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("run_*/autotune_summary.json"))


def poll_autotune_task() -> dict[str, Any] | None:
    task = st.session_state.get(AUTOTUNE_TASK_KEY)
    if not task:
        return None
    if task.get("status") != "running":
        return task
    proc = task.get("process")
    if proc is None:
        task["status"] = "failed"
        task["error"] = "任务进程丢失"
        st.session_state[AUTOTUNE_TASK_KEY] = task
        return task
    return_code = proc.poll()
    if return_code is None:
        return task
    task["returncode"] = return_code
    task["status"] = "success" if return_code == 0 else "failed"
    task["summary_path"] = extract_marker(task["log_path"], "AUTOTUNE_SUMMARY_PATH")
    task["report_path"] = extract_marker(task["log_path"], "AUTOTUNE_REPORT_PATH")
    st.session_state[AUTOTUNE_TASK_KEY] = task
    return task


def load_autotune_summary(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def autotune_summary_to_df(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video in summary.get("videos", []):
        video_path = video.get("input")
        for encoder, encoder_data in video.get("encoders", {}).items():
            recommendation = encoder_data.get("recommendation")
            if recommendation:
                rows.append(
                    {
                        "视频": video_path,
                        "编码器": encoder,
                        "推荐 CRF": recommendation.get("crf"),
                        "VMAF": recommendation.get("vmaf_mean"),
                        "大小(MB)": recommendation.get("output_size_mb"),
                        "压缩比(%)": recommendation.get("compression_ratio"),
                        "阈值是否满足": not recommendation.get("threshold_unmet", True),
                        "来源阶段": recommendation.get("source_stage"),
                    }
                )
            else:
                rows.append(
                    {
                        "视频": video_path,
                        "编码器": encoder,
                        "推荐 CRF": None,
                        "VMAF": None,
                        "大小(MB)": None,
                        "压缩比(%)": None,
                        "阈值是否满足": False,
                        "来源阶段": None,
                    }
                )
    return pd.DataFrame(rows)


def render_autotune_summary(summary_path: str) -> None:
    if not summary_path or not os.path.exists(summary_path):
        st.warning("未找到自动筛选结果 summary 文件。")
        return
    try:
        summary = load_autotune_summary(summary_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"读取 summary 失败: {exc}")
        return

    st.success(f"已加载结果: {summary_path}")
    stats = summary.get("stats", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("视频数", stats.get("videos_total", 0))
    with col2:
        st.metric("成功视频数", stats.get("videos_succeeded", 0))
    with col3:
        st.metric(
            "成功推荐数",
            f"{stats.get('successful_recommendations', 0)}/{stats.get('recommendations_total', 0)}",
        )

    df = autotune_summary_to_df(summary)
    if not df.empty:
        st.dataframe(df, use_container_width=True)

    st.subheader("推荐区间")
    range_rows: list[dict[str, Any]] = []
    for encoder, value in summary.get("encoder_recommendation_ranges", {}).items():
        if not value:
            range_rows.append({"编码器": encoder, "推荐区间": "无"})
            continue
        range_rows.append(
            {
                "编码器": encoder,
                "推荐区间": f"CRF {int(value['min_crf'])} ~ {int(value['max_crf'])}",
            }
        )
    if range_rows:
        st.table(pd.DataFrame(range_rows))


def start_autotune_task(
    *,
    inputs: list[str],
    encoders: list[str],
    target_vmaf: float,
    coarse_duration: int,
    coarse_scale: int,
    output_root: str,
    jobs: int,
    strict_mode: bool,
    vmaf_threads: int,
    vmaf_io_mode: str,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    task_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = root / f"autotune_task_{task_id}.log"
    cmd = [
        sys.executable,
        "main.py",
        "autotune",
        *inputs,
        "--encoders",
        *encoders,
        "--target-vmaf",
        str(target_vmaf),
        "--coarse-duration",
        str(coarse_duration),
        "--coarse-scale",
        str(coarse_scale),
        "--output",
        output_root,
        "--jobs",
        str(jobs),
        "--strict" if strict_mode else "--no-strict",
        "--vmaf-threads",
        str(vmaf_threads),
        "--vmaf-io-mode",
        vmaf_io_mode,
    ]

    with log_path.open("w", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )

    st.session_state[AUTOTUNE_TASK_KEY] = {
        "id": task_id,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": str(log_path),
        "cmd": cmd,
        "process": process,
        "output_root": output_root,
        "inputs": inputs,
    }


def stop_autotune_task() -> None:
    task = st.session_state.get(AUTOTUNE_TASK_KEY)
    if not task:
        return
    proc = task.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    task["status"] = "cancelled"
    st.session_state[AUTOTUNE_TASK_KEY] = task


# 侧边栏 - 配置
st.sidebar.header("配置")

# 模式选择
mode = st.sidebar.radio(
    "模式",
    ["查看结果", "运行评估", "自动筛选"],
    index=0,
)

if mode == "查看结果":
    st.header("📊 查看评估结果")

    result_files = get_existing_results()

    if not result_files:
        st.warning("还没有评估结果。请先运行评估或将结果文件放到 results/ 目录下。")
    else:
        selected_file = st.selectbox(
            "选择结果文件",
            result_files,
            format_func=lambda x: Path(x).parent.name,
        )

        if selected_file:
            try:
                results = load_results(selected_file)
                df = results_to_dataframe(results)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("测试数量", len(results))
                with col2:
                    st.metric("最高 VMAF", f"{df['VMAF 平均'].max():.2f}")
                with col3:
                    if "SSIM 平均" in df.columns and df["SSIM 平均"].sum() > 0:
                        st.metric("最高 SSIM", f"{df['SSIM 平均'].max():.4f}")
                    else:
                        st.metric("最高 SSIM", "N/A")
                with col4:
                    st.metric("最小文件", f"{df['文件大小 (MB)'].min():.2f} MB")

                st.subheader("🎯 四象限图：质量 vs 压缩比")
                st.markdown(
                    """
                > **如何看图**：左上角是最佳区域（高质量 + 小文件），右下角是最差区域。
                > 每个点代表一个测试配置，点旁边的标签显示参数值。
                """
                )
                st.plotly_chart(plot_quadrant(df), use_container_width=True)

                st.subheader("📈 指标 vs 质量参数")
                available_metrics = []
                for metric_name, config in METRIC_OPTIONS.items():
                    if config["col"] in df.columns and df[config["col"]].sum() > 0:
                        available_metrics.append(metric_name)

                if available_metrics:
                    selected_metric = st.selectbox(
                        "选择要查看的指标",
                        available_metrics,
                        index=0,
                    )
                    st.plotly_chart(
                        plot_metric_vs_param(df, selected_metric),
                        use_container_width=True,
                    )
                else:
                    st.warning("没有可用的指标数据")

                st.subheader("📉 VMAF vs 文件大小")
                st.plotly_chart(plot_vmaf_vs_size(df), use_container_width=True)

                st.subheader("⚡ 压缩效率")
                st.plotly_chart(plot_compression_efficiency(df), use_container_width=True)

                st.subheader("📋 详细数据")
                format_dict = {
                    "VMAF 平均": "{:.2f}",
                    "VMAF 最小": "{:.2f}",
                    "VMAF 最大": "{:.2f}",
                    "PSNR-HVS 平均": "{:.2f}",
                    "PSNR-HVS 最小": "{:.2f}",
                    "PSNR-HVS 最大": "{:.2f}",
                    "SSIM 平均": "{:.4f}",
                    "SSIM 最小": "{:.4f}",
                    "SSIM 最大": "{:.4f}",
                    "MS-SSIM 平均": "{:.4f}",
                    "MS-SSIM 最小": "{:.4f}",
                    "MS-SSIM 最大": "{:.4f}",
                    "SNR 平均 (dB)": "{:.2f}",
                    "SNR 最小 (dB)": "{:.2f}",
                    "SNR 最大 (dB)": "{:.2f}",
                    "文件大小 (MB)": "{:.2f}",
                    "压缩比 (%)": "{:.2f}",
                    "编码耗时 (秒)": "{:.2f}",
                }
                format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
                st.dataframe(df.style.format(format_dict), use_container_width=True)

                st.subheader("🎯 最优推荐")
                high_quality = df[df["VMAF 平均"] >= 90]
                if not high_quality.empty:
                    best = high_quality.loc[high_quality["文件大小 (MB)"].idxmin()]
                    recommendation = (
                        f"推荐配置（VMAF ≥ 90 中最小文件）: "
                        f"**{best['编码器']}**, 参数值 **{best['参数值']}**, "
                        f"VMAF **{best['VMAF 平均']:.2f}**"
                    )
                    if "SSIM 平均" in best and best["SSIM 平均"] > 0:
                        recommendation += f", SSIM **{best['SSIM 平均']:.4f}**"
                    if "SNR 平均 (dB)" in best and best["SNR 平均 (dB)"] > 0:
                        recommendation += f", SNR **{best['SNR 平均 (dB)']:.2f} dB**"
                    recommendation += f", 大小 **{best['文件大小 (MB)']:.2f} MB**"
                    st.success(recommendation)
                else:
                    st.info("没有 VMAF ≥ 90 的配置，请尝试更高的质量参数")

            except Exception as e:  # noqa: BLE001
                st.error(f"加载结果失败: {e}")

elif mode == "运行评估":
    st.header("🚀 运行评估")

    video_path = st.text_input(
        "视频文件路径",
        placeholder="/path/to/video.mp4",
    )

    selected_encoders = st.multiselect(
        "选择编码器",
        list(ENCODERS.keys()),
        default=["hevc"],
        format_func=lambda x: ENCODERS[x].name,
    )

    st.subheader("参数配置")

    encoder_params = {}
    for encoder_key in selected_encoders:
        config = ENCODERS[encoder_key]
        st.markdown(f"**{config.name}** (`-{config.param_name}`)")

        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.number_input(
                "起始值",
                min_value=0,
                max_value=100,
                value=config.param_range[0],
                key=f"{encoder_key}_start",
            )
        with col2:
            end = st.number_input(
                "结束值",
                min_value=0,
                max_value=100,
                value=config.param_range[1],
                key=f"{encoder_key}_end",
            )
        with col3:
            step = st.number_input(
                "步长",
                min_value=1,
                max_value=20,
                value=5,
                key=f"{encoder_key}_step",
            )

        encoder_params[encoder_key] = (int(start), int(end), int(step))

    if st.button("开始评估", type="primary", disabled=not video_path or not selected_encoders):
        if not os.path.exists(video_path):
            st.error(f"文件不存在: {video_path}")
        else:
            input_name = Path(video_path).stem
            output_dir = os.path.join("results", input_name)
            os.makedirs(os.path.join(output_dir, "encoded"), exist_ok=True)

            results = []
            total_tasks = sum(
                len(range(start, end + 1, step))
                for start, end, step in encoder_params.values()
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            current_task = 0

            for encoder_key in selected_encoders:
                start, end, step = encoder_params[encoder_key]
                config = ENCODERS[encoder_key]

                for param_value in range(start, end + 1, step):
                    current_task += 1
                    progress = current_task / total_tasks
                    progress_bar.progress(progress)
                    status_text.text(
                        f"正在处理: {config.name}, {config.param_name}={param_value} "
                        f"({current_task}/{total_tasks})"
                    )

                    try:
                        result = run_single_benchmark(
                            video_path, output_dir, encoder_key, param_value
                        )
                        results.append(result)
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"编码失败 ({encoder_key}, {param_value}): {e}")

            if results:
                results_path = os.path.join(output_dir, "results.json")
                save_results(results, results_path)
                progress_bar.progress(1.0)
                status_text.text("评估完成！")
                st.success(f"评估完成！结果已保存到 {results_path}")
                st.info("切换到「查看结果」模式查看详细图表")
            else:
                st.error("没有成功的评估结果")

else:
    st.header("🤖 自动筛选")
    st.caption("两阶段粗到细：先短片粗扫锁区间，再全片精扫。")

    task = poll_autotune_task()

    input_mode = st.radio("视频来源", ["目录扫描", "手工输入"], horizontal=True)
    if input_mode == "目录扫描":
        source_dir = st.text_input("视频目录", value=".")
        discovered_videos = discover_videos(source_dir)
        st.caption(f"发现 {len(discovered_videos)} 个视频文件")
        selected_inputs = discovered_videos
    else:
        manual_inputs = st.text_area(
            "视频路径（每行一个）",
            placeholder="/path/to/sample1.mp4\n/path/to/sample2.mp4",
            height=120,
        )
        selected_inputs = [line.strip() for line in manual_inputs.splitlines() if line.strip()]

    if selected_inputs:
        st.write("待处理视频:")
        for path in selected_inputs:
            st.code(path)

    col1, col2 = st.columns(2)
    with col1:
        target_vmaf = st.number_input("目标 VMAF", min_value=50.0, max_value=100.0, value=95.0)
        coarse_duration = st.number_input("粗扫时长（秒）", min_value=1, max_value=30, value=10)
        coarse_scale = st.number_input("粗扫宽度", min_value=160, max_value=3840, value=1280)
    with col2:
        autotune_encoders = st.multiselect(
            "编码器",
            list(ENCODERS.keys()),
            default=["hevc", "av1"],
            format_func=lambda x: ENCODERS[x].name,
        )
        jobs = st.number_input("并发任务数", min_value=1, max_value=8, value=1)
        strict_mode = st.checkbox("严格模式", value=False)
        vmaf_threads = st.number_input(
            "VMAF 线程数",
            min_value=1,
            max_value=max(1, os.cpu_count() or 1),
            value=max(1, os.cpu_count() or 1),
        )
        vmaf_io_mode = st.selectbox(
            "VMAF I/O 模式",
            ["auto", "libvmaf", "fifo", "file"],
            index=0,
        )

    output_root = st.text_input("输出目录", value="results_autotune")

    task_running = bool(task and task.get("status") == "running")
    start_disabled = task_running or not selected_inputs or not autotune_encoders
    if st.button("启动自动筛选", type="primary", disabled=start_disabled):
        start_autotune_task(
            inputs=selected_inputs,
            encoders=autotune_encoders,
            target_vmaf=float(target_vmaf),
            coarse_duration=int(coarse_duration),
            coarse_scale=int(coarse_scale),
            output_root=output_root,
            jobs=int(jobs),
            strict_mode=strict_mode,
            vmaf_threads=int(vmaf_threads),
            vmaf_io_mode=vmaf_io_mode,
        )
        st.rerun()

    if task:
        st.subheader("任务状态")
        status = task.get("status")
        st.write(f"- 任务 ID: `{task.get('id')}`")
        st.write(f"- 状态: `{status}`")
        st.write(f"- 日志: `{task.get('log_path')}`")

        if status == "running":
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("取消任务"):
                    stop_autotune_task()
                    st.rerun()
            with col_b:
                st.button("立即刷新", on_click=st.rerun)

        log_tail = tail_file(task.get("log_path", ""), lines=100)
        st.text_area("任务日志（最近 100 行）", log_tail, height=280)

        if status in {"success", "failed", "cancelled"}:
            if status == "success":
                st.success("任务已完成。")
            elif status == "failed":
                st.error(f"任务失败（exit={task.get('returncode')}）。")
            else:
                st.warning("任务已取消。")

            summary_path = task.get("summary_path")
            report_path = task.get("report_path")
            if report_path:
                st.write(f"报告: `{report_path}`")
            if summary_path:
                render_autotune_summary(summary_path)
            else:
                st.info("日志中尚未解析到 summary 路径。")

        if status == "running":
            st.info("任务进行中，页面每 2 秒自动刷新。")
            time.sleep(2)
            st.rerun()

    st.subheader("历史结果")
    existing_summaries = get_existing_autotune_summaries(output_root)
    if existing_summaries:
        selected_summary = st.selectbox(
            "选择历史 summary",
            existing_summaries,
            format_func=lambda p: str(Path(p).parent.name),
        )
        if selected_summary:
            render_autotune_summary(selected_summary)
    else:
        st.caption("当前输出目录还没有历史 summary。")


st.markdown("---")
st.markdown("💡 **提示**: 大批量筛选建议直接用 `python main.py autotune ...`。")
