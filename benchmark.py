#!/usr/bin/env python3
"""
视频质量评估工具 - 核心编码和评估模块

支持的编码器:
- HEVC (VideoToolbox): 使用 -q:v 参数
- AV1 (SVT-AV1): 使用 -crf 参数
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal


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
        name="HEVC (VideoToolbox)",
        codec="hevc_videotoolbox",
        param_name="q:v",
        param_range=(40, 70),
        extra_args=["-tag:v", "hvc1"],
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
    vmaf_mean: float
    vmaf_min: float
    vmaf_max: float
    vmaf_harmonic_mean: float


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
    使用命名管道计算 VMAF 分数（避免生成大文件）
    
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
        
        # 运行 vmaf
        vmaf_result = subprocess.run(
            ["vmaf", "-r", ref_pipe, "-d", dist_pipe,
             "--json", "-o", output_json, "--threads", "8"],
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
    """从 VMAF 结果中提取分数"""
    pooled = vmaf_result.get("pooled_metrics", {}).get("vmaf", {})
    return {
        "vmaf_mean": pooled.get("mean", 0),
        "vmaf_min": pooled.get("min", 0),
        "vmaf_max": pooled.get("max", 0),
        "vmaf_harmonic_mean": pooled.get("harmonic_mean", 0),
    }


def run_single_benchmark(
    input_path: str,
    output_dir: str,
    encoder_key: str,
    param_value: int,
) -> BenchmarkResult:
    """运行单次编码评估"""
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
    
    print(f"  计算 VMAF 分数...")
    
    # 计算 VMAF
    vmaf_result = calculate_vmaf(input_path, output_path, vmaf_json)
    vmaf_scores = extract_vmaf_scores(vmaf_result)
    
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
        **{k: round(v, 2) for k, v in vmaf_scores.items()},
    )


def run_benchmark(
    input_path: str,
    output_dir: str,
    encoders: list[str],
    param_step: int = 5,
) -> list[BenchmarkResult]:
    """
    批量运行编码评估
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        encoders: 要测试的编码器列表
        param_step: 参数步长
    
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
                    input_path, output_dir, encoder_key, param_value
                )
                results.append(result)
                print(f"    VMAF: {result.vmaf_mean:.2f}, 大小: {result.output_size_mb:.2f} MB")
            except Exception as e:
                print(f"    错误: {e}")
    
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
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return 1
    
    # 创建输出目录
    input_name = Path(args.input).stem
    output_dir = os.path.join(args.output, input_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入视频: {args.input}")
    print(f"输出目录: {output_dir}")
    print(f"编码器: {', '.join(args.encoders)}")
    print(f"参数步长: {args.step}")
    
    # 运行评估
    results = run_benchmark(
        args.input,
        output_dir,
        args.encoders,
        args.step,
    )
    
    # 保存结果
    results_path = os.path.join(output_dir, "results.json")
    save_results(results, results_path)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("评估完成，结果汇总:")
    print("=" * 60)
    print(f"{'编码器':<10} {'参数':<10} {'VMAF':<10} {'大小(MB)':<12} {'压缩比':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.encoder:<10} {r.param_value:<10} {r.vmaf_mean:<10.2f} {r.output_size_mb:<12.2f} {r.compression_ratio:<10.2f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())
