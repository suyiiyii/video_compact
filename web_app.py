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
from pathlib import Path

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


# 侧边栏 - 配置
st.sidebar.header("配置")

# 模式选择
mode = st.sidebar.radio(
    "模式",
    ["查看结果", "运行评估"],
    index=0,
)

if mode == "查看结果":
    st.header("📊 查看评估结果")
    
    # 获取已有结果
    result_files = get_existing_results()
    
    if not result_files:
        st.warning("还没有评估结果。请先运行评估或将结果文件放到 results/ 目录下。")
    else:
        # 选择结果文件
        selected_file = st.selectbox(
            "选择结果文件",
            result_files,
            format_func=lambda x: Path(x).parent.name,
        )
        
        if selected_file:
            # 加载结果
            try:
                results = load_results(selected_file)
                df = results_to_dataframe(results)
                
                # 显示基本信息
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
                
                # 四象限图（最重要，放在最前面）
                st.subheader("🎯 四象限图：质量 vs 压缩比")
                st.markdown("""
                > **如何看图**：左上角是最佳区域（高质量 + 小文件），右下角是最差区域。
                > 每个点代表一个测试配置，点旁边的标签显示参数值。
                """)
                st.plotly_chart(plot_quadrant(df), use_container_width=True)
                
                # 指标选择器曲线图
                st.subheader("📈 指标 vs 质量参数")
                
                # 检查哪些指标有数据
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
                    st.plotly_chart(plot_metric_vs_param(df, selected_metric), use_container_width=True)
                else:
                    st.warning("没有可用的指标数据")
                
                st.subheader("📉 VMAF vs 文件大小")
                st.plotly_chart(plot_vmaf_vs_size(df), use_container_width=True)
                
                st.subheader("⚡ 压缩效率")
                st.plotly_chart(plot_compression_efficiency(df), use_container_width=True)
                
                # 数据表格
                st.subheader("📋 详细数据")
                
                # 构建格式化字典
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
                # 只保留存在的列
                format_dict = {k: v for k, v in format_dict.items() if k in df.columns}
                
                st.dataframe(
                    df.style.format(format_dict),
                    use_container_width=True,
                )
                
                # 最优推荐
                st.subheader("🎯 最优推荐")
                
                # 找到 VMAF > 90 且文件最小的配置
                high_quality = df[df["VMAF 平均"] >= 90]
                if not high_quality.empty:
                    best = high_quality.loc[high_quality["文件大小 (MB)"].idxmin()]
                    
                    # 构建推荐信息
                    recommendation = (
                        f"推荐配置（VMAF ≥ 90 中最小文件）: "
                        f"**{best['编码器']}**, 参数值 **{best['参数值']}**, "
                        f"VMAF **{best['VMAF 平均']:.2f}**"
                    )
                    
                    # 如果有其他指标，也显示
                    if "SSIM 平均" in best and best["SSIM 平均"] > 0:
                        recommendation += f", SSIM **{best['SSIM 平均']:.4f}**"
                    if "SNR 平均 (dB)" in best and best["SNR 平均 (dB)"] > 0:
                        recommendation += f", SNR **{best['SNR 平均 (dB)']:.2f} dB**"
                    
                    recommendation += f", 大小 **{best['文件大小 (MB)']:.2f} MB**"
                    
                    st.success(recommendation)
                else:
                    st.info("没有 VMAF ≥ 90 的配置，请尝试更高的质量参数")
                
            except Exception as e:
                st.error(f"加载结果失败: {e}")

else:  # 运行评估模式
    st.header("🚀 运行评估")
    
    # 输入视频
    video_path = st.text_input(
        "视频文件路径",
        placeholder="/path/to/video.mp4",
    )
    
    # 编码器选择
    selected_encoders = st.multiselect(
        "选择编码器",
        list(ENCODERS.keys()),
        default=["hevc"],
        format_func=lambda x: ENCODERS[x].name,
    )
    
    # 参数配置
    st.subheader("参数配置")
    
    encoder_params = {}
    for encoder_key in selected_encoders:
        config = ENCODERS[encoder_key]
        st.markdown(f"**{config.name}** (`-{config.param_name}`)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.number_input(
                f"起始值",
                min_value=0,
                max_value=100,
                value=config.param_range[0],
                key=f"{encoder_key}_start",
            )
        with col2:
            end = st.number_input(
                f"结束值",
                min_value=0,
                max_value=100,
                value=config.param_range[1],
                key=f"{encoder_key}_end",
            )
        with col3:
            step = st.number_input(
                f"步长",
                min_value=1,
                max_value=20,
                value=5,
                key=f"{encoder_key}_step",
            )
        
        encoder_params[encoder_key] = (int(start), int(end), int(step))
    
    # 运行按钮
    if st.button("开始评估", type="primary", disabled=not video_path or not selected_encoders):
        if not os.path.exists(video_path):
            st.error(f"文件不存在: {video_path}")
        else:
            # 创建输出目录
            input_name = Path(video_path).stem
            output_dir = os.path.join("results", input_name)
            os.makedirs(os.path.join(output_dir, "encoded"), exist_ok=True)
            
            results = []
            
            # 计算总任务数
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
                    except Exception as e:
                        st.warning(f"编码失败 ({encoder_key}, {param_value}): {e}")
            
            # 保存结果
            if results:
                results_path = os.path.join(output_dir, "results.json")
                save_results(results, results_path)
                
                progress_bar.progress(1.0)
                status_text.text("评估完成！")
                
                st.success(f"评估完成！结果已保存到 {results_path}")
                st.info("切换到「查看结果」模式查看详细图表")
            else:
                st.error("没有成功的评估结果")


# 页脚
st.markdown("---")
st.markdown(
    "💡 **提示**: 使用命令行运行评估更快：`uv run python benchmark.py video.mp4`"
)
