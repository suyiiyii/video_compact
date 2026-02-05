#!/usr/bin/env python3
"""
视频质量评估工具 - 核心编码和评估模块

支持的编码器:
- HEVC (libx265): 使用 -crf 参数
- AV1 (SVT-AV1): 使用 -crf 参数

支持的质量指标:
- VMAF: Video Multi-method Assessment Fusion
- PSNR-HVS: PSNR Human Visual System
- SSIM: Structural Similarity Index
- MS-SSIM: Multi-Scale Structural Similarity Index
- SNR: Signal-to-Noise Ratio
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
@dataclass
class EncoderConfig:
    """编码器配置"""
    name: str
    codec: str
    param_name: str
    param_range: tuple[int, int]
    extra_args: list[str]


# 编码器配置
ENCODERS = {
    "hevc": EncoderConfig(
        name="HEVC (libx265)",
        codec="libx265",
        param_name="crf",
        param_range=(20, 35),
        extra_args=["-preset", "medium", "-tag:v", "hvc1"],
    ),
    "av1": EncoderConfig(
        name="AV1 (SVT-AV1)",
        codec="libsvtav1",
        param_name="crf",
        param_range=(25, 50),
        extra_args=["-preset", "6"],
    ),
}


@dataclass
class BenchmarkResult:
    """单次评估结果"""
    encoder: str
    param_name: str
    param_value: int
    input_file: str
    output_file: str
    input_size_mb: float
    output_size_mb: float
    compression_ratio: float
    encode_time_seconds: float
    # VMAF 指标
    vmaf_mean: float
    vmaf_min: float
    vmaf_max: float
    vmaf_harmonic_mean: float
    # PSNR-HVS 指标
    psnr_hvs_mean: float = 0.0
    psnr_hvs_min: float = 0.0
    psnr_hvs_max: float = 0.0
    psnr_hvs_harmonic_mean: float = 0.0
    # SSIM 指标
    ssim_mean: float = 0.0
    ssim_min: float = 0.0
    ssim_max: float = 0.0
    ssim_harmonic_mean: float = 0.0
    # MS-SSIM 指标
    ms_ssim_mean: float = 0.0
    ms_ssim_min: float = 0.0
    ms_ssim_max: float = 0.0
    ms_ssim_harmonic_mean: float = 0.0
    # SNR 指标
    snr_mean: float = 0.0
    snr_min: float = 0.0
    snr_max: float = 0.0
    snr_harmonic_mean: float = 0.0


def get_video_info(video_path: str) -> dict:
    """获取视频信息"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def encode_video(
    input_path: str,
    output_path: str,
    encoder_key: str,
    param_value: int,
) -> float:
    """
    使用指定编码器和参数编码视频
    
    返回: 编码耗时（秒）
    """
    config = ENCODERS[encoder_key]
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", config.codec,
        f"-{config.param_name}", str(param_value),
        *config.extra_args,
        output_path
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    encode_time = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"编码失败: {result.stderr}")
    
    return encode_time


def calculate_vmaf(reference_path: str, distorted_path: str, output_json: str) -> dict:
    """
    使用命名管道计算 VMAF 及相关指标（避免生成大文件）
    
    计算的指标:
    - VMAF
    - PSNR-HVS
    - SSIM (float_ssim)
    - MS-SSIM (float_ms_ssim)
    
    返回: VMAF 结果字典
    """
    # 创建临时命名管道
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_pipe = os.path.join(tmpdir, "ref.y4m")
        dist_pipe = os.path.join(tmpdir, "dist.y4m")
        
        os.mkfifo(ref_pipe)
        os.mkfifo(dist_pipe)
        
        # 启动 ffmpeg 进程输出到管道
        ref_proc = subprocess.Popen(
            ["ffmpeg", "-i", reference_path, "-pix_fmt", "yuv420p", 
             "-f", "yuv4mpegpipe", ref_pipe, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        dist_proc = subprocess.Popen(
            ["ffmpeg", "-i", distorted_path, "-pix_fmt", "yuv420p",
             "-f", "yuv4mpegpipe", dist_pipe, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # 运行 vmaf，添加额外特征计算
        vmaf_result = subprocess.run(
            [
                "vmaf", "-r", ref_pipe, "-d", dist_pipe,
                "--json", "-o", output_json, "--threads", "8",
                "--feature", "psnr_hvs",
                "--feature", "float_ssim",
                "--feature", "float_ms_ssim",
            ],
            capture_output=True,
            text=True,
        )
        
        # 等待 ffmpeg 进程结束
        ref_proc.wait()
        dist_proc.wait()
        
        if vmaf_result.returncode != 0:
            raise RuntimeError(f"VMAF 计算失败: {vmaf_result.stderr}")
    
    # 读取结果
    with open(output_json, "r") as f:
        result = json.load(f)
    
    return result


def extract_vmaf_scores(vmaf_result: dict) -> dict:
    """从 VMAF 结果中提取所有指标分数"""
    pooled_metrics = vmaf_result.get("pooled_metrics", {})
    
    def extract_metric(metric_name: str, prefix: str) -> dict:
        """提取单个指标的统计值"""
        metric_data = pooled_metrics.get(metric_name, {})
        return {
            f"{prefix}_mean": metric_data.get("mean", 0),
            f"{prefix}_min": metric_data.get("min", 0),
            f"{prefix}_max": metric_data.get("max", 0),
            f"{prefix}_harmonic_mean": metric_data.get("harmonic_mean", 0),
        }
    
    scores = {}
    # VMAF
    scores.update(extract_metric("vmaf", "vmaf"))
    # PSNR-HVS
    scores.update(extract_metric("psnr_hvs", "psnr_hvs"))
    # SSIM (vmaf 输出为 float_ssim)
    scores.update(extract_metric("float_ssim", "ssim"))
    # MS-SSIM (vmaf 输出为 float_ms_ssim)
    scores.update(extract_metric("float_ms_ssim", "ms_ssim"))
    
    return scores


def get_video_resolution(video_path: str) -> tuple[int, int]:
    """获取视频分辨率"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"无法获取视频分辨率: {result.stderr}")
    
    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    return stream.get("width", 0), stream.get("height", 0)


def calculate_snr(reference_path: str, distorted_path: str) -> dict:
    """
    计算 SNR (Signal-to-Noise Ratio)
    
    使用灰度帧逐帧计算 SNR，然后汇总统计值
    
    返回: SNR 统计字典 (mean, min, max, harmonic_mean)
    """
    width, height = get_video_resolution(reference_path)
    if width == 0 or height == 0:
        raise RuntimeError("无法获取视频分辨率")
    
    # 使用 ffmpeg 提取灰度帧数据
    ref_cmd = [
        "ffmpeg", "-i", reference_path,
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-"
    ]
    dist_cmd = [
        "ffmpeg", "-i", distorted_path,
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-"
    ]
    
    ref_proc = subprocess.Popen(
        ref_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    dist_proc = subprocess.Popen(
        dist_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    
    frame_size = width * height
    snr_values = []
    
    while True:
        ref_data = ref_proc.stdout.read(frame_size)
        dist_data = dist_proc.stdout.read(frame_size)
        
        if len(ref_data) < frame_size or len(dist_data) < frame_size:
            break
        
        # 转换为 numpy 数组
        ref_frame = np.frombuffer(ref_data, dtype=np.uint8).astype(np.float64)
        dist_frame = np.frombuffer(dist_data, dtype=np.uint8).astype(np.float64)
        
        # 计算信号功率和噪声功率
        signal_power = np.mean(ref_frame ** 2)
        noise = ref_frame - dist_frame
        noise_power = np.mean(noise ** 2)
        
        # 避免除以零
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 100.0  # 无噪声，设为高值
        
        snr_values.append(snr_db)
    
    ref_proc.wait()
    dist_proc.wait()
    
    if not snr_values:
        raise RuntimeError("无法计算 SNR：没有有效帧")
    
    snr_array = np.array(snr_values)
    
    # 计算调和平均值（过滤无穷值）
    finite_snr = snr_array[np.isfinite(snr_array)]
    if len(finite_snr) > 0 and np.all(finite_snr > 0):
        harmonic_mean = len(finite_snr) / np.sum(1.0 / finite_snr)
    else:
        harmonic_mean = np.mean(finite_snr) if len(finite_snr) > 0 else 0
    
    return {
        "snr_mean": float(np.mean(snr_array)),
        "snr_min": float(np.min(snr_array)),
        "snr_max": float(np.max(snr_array)),
        "snr_harmonic_mean": float(harmonic_mean),
    }


def validate_metrics(scores: dict, required_prefixes: list[str]) -> None:
    """
    验证所有必需的指标是否已计算
    
    Args:
        scores: 指标字典
        required_prefixes: 必需的指标前缀列表
    
    Raises:
        RuntimeError: 如果任何必需指标缺失或为零
    """
    for prefix in required_prefixes:
        mean_key = f"{prefix}_mean"
        if mean_key not in scores:
            raise RuntimeError(f"指标缺失: {prefix} (未找到 {mean_key})")
        # 检查是否为有效值（除了 SSIM/MS-SSIM 可能为 1.0 表示完全相同）
        if scores[mean_key] == 0 and prefix not in ["ssim", "ms_ssim"]:
            raise RuntimeError(f"指标无效: {prefix}_mean = 0")


def run_single_benchmark(
    input_path: str,
    output_dir: str,
    encoder_key: str,
    param_value: int,
    strict_mode: bool = True,
) -> BenchmarkResult:
    """
    运行单次编码评估
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        encoder_key: 编码器键
        param_value: 参数值
        strict_mode: 是否启用严格模式（所有指标必须可计算）
    
    Returns:
        BenchmarkResult 对象
    
    Raises:
        RuntimeError: 如果 strict_mode=True 且任何指标计算失败
    """
    config = ENCODERS[encoder_key]
    input_name = Path(input_path).stem
    
    # 输出文件路径
    output_filename = f"{input_name}_{encoder_key}_{config.param_name}{param_value}.mp4"
    output_path = os.path.join(output_dir, "encoded", output_filename)
    vmaf_json = os.path.join(output_dir, f"vmaf_{encoder_key}_{config.param_name}{param_value}.json")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"  编码中: {config.name}, {config.param_name}={param_value}")
    
    # 编码
    encode_time = encode_video(input_path, output_path, encoder_key, param_value)
    
    # 获取文件大小
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"  计算质量指标 (VMAF/PSNR-HVS/SSIM/MS-SSIM)...")
    
    # 计算 VMAF 及相关指标
    vmaf_result = calculate_vmaf(input_path, output_path, vmaf_json)
    all_scores = extract_vmaf_scores(vmaf_result)
    
    print(f"  计算 SNR...")
    
    # 计算 SNR
    snr_scores = calculate_snr(input_path, output_path)
    all_scores.update(snr_scores)
    
    # 严格模式：验证所有指标
    if strict_mode:
        required_metrics = ["vmaf", "psnr_hvs", "ssim", "ms_ssim", "snr"]
        validate_metrics(all_scores, required_metrics)
        print(f"  ✓ 所有指标计算完成")
    
    return BenchmarkResult(
        encoder=encoder_key,
        param_name=config.param_name,
        param_value=param_value,
        input_file=input_path,
        output_file=output_path,
        input_size_mb=round(input_size, 2),
        output_size_mb=round(output_size, 2),
        compression_ratio=round(output_size / input_size * 100, 2),
        encode_time_seconds=round(encode_time, 2),
        **{k: round(v, 4) for k, v in all_scores.items()},
    )


def run_benchmark(
    input_path: str,
    output_dir: str,
    encoders: list[str],
    param_step: int = 5,
    strict_mode: bool = True,
) -> list[BenchmarkResult]:
    """
    批量运行编码评估
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        encoders: 要测试的编码器列表
        param_step: 参数步长
        strict_mode: 是否启用严格模式（所有指标必须可计算）
    
    Returns:
        评估结果列表
    """
    results = []
    
    for encoder_key in encoders:
        if encoder_key not in ENCODERS:
            print(f"警告: 未知编码器 {encoder_key}，跳过")
            continue
        
        config = ENCODERS[encoder_key]
        print(f"\n测试编码器: {config.name}")
        print(f"参数范围: {config.param_name} = {config.param_range[0]} ~ {config.param_range[1]}")
        
        # 遍历参数范围
        for param_value in range(config.param_range[0], config.param_range[1] + 1, param_step):
            try:
                result = run_single_benchmark(
                    input_path, output_dir, encoder_key, param_value, strict_mode
                )
                results.append(result)
                print(f"    VMAF: {result.vmaf_mean:.2f}, SSIM: {result.ssim_mean:.4f}, "
                      f"SNR: {result.snr_mean:.2f} dB, 大小: {result.output_size_mb:.2f} MB")
            except Exception as e:
                print(f"    错误: {e}")
                if strict_mode:
                    raise  # 严格模式下，任何错误都停止
    
    return results


def save_results(results: list[BenchmarkResult], output_path: str):
    """保存结果到 JSON 文件"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


def load_results(input_path: str) -> list[BenchmarkResult]:
    """从 JSON 文件加载结果"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    return [BenchmarkResult(**r) for r in data["results"]]


def main():
    parser = argparse.ArgumentParser(description="视频质量评估工具")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument(
        "--encoders", "-e",
        nargs="+",
        default=["hevc", "av1"],
        choices=list(ENCODERS.keys()),
        help="要测试的编码器",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="输出目录",
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=5,
        help="参数步长",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="禁用严格模式（允许部分指标计算失败）",
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return 1
    
    # 创建输出目录
    input_name = Path(args.input).stem
    output_dir = os.path.join(args.output, input_name)
    os.makedirs(output_dir, exist_ok=True)
    
    strict_mode = not args.no_strict
    
    print(f"输入视频: {args.input}")
    print(f"输出目录: {output_dir}")
    print(f"编码器: {', '.join(args.encoders)}")
    print(f"参数步长: {args.step}")
    print(f"严格模式: {'是' if strict_mode else '否'}")
    
    # 运行评估
    try:
        results = run_benchmark(
            args.input,
            output_dir,
            args.encoders,
            args.step,
            strict_mode,
        )
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("严格模式下，所有指标必须可计算。使用 --no-strict 禁用严格模式。")
        return 1
    
    # 保存结果
    results_path = os.path.join(output_dir, "results.json")
    save_results(results, results_path)
    
    # 打印汇总
    print("\n" + "=" * 100)
    print("评估完成，结果汇总:")
    print("=" * 100)
    print(f"{'编码器':<8} {'参数':<6} {'VMAF':<8} {'SSIM':<8} {'MS-SSIM':<10} {'PSNR-HVS':<10} {'SNR(dB)':<10} {'大小(MB)':<10} {'压缩比':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r.encoder:<8} {r.param_value:<6} {r.vmaf_mean:<8.2f} {r.ssim_mean:<8.4f} "
              f"{r.ms_ssim_mean:<10.4f} {r.psnr_hvs_mean:<10.2f} {r.snr_mean:<10.2f} "
              f"{r.output_size_mb:<10.2f} {r.compression_ratio:<8.2f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())
