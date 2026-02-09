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

from __future__ import annotations

import argparse
import functools
import json
import os
import selectors
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VMAF_IO_MODES = {"auto", "libvmaf", "fifo", "file"}

FFMPEG_BIN = os.getenv("VIDEO_COMPACT_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
FFPROBE_BIN = os.getenv("VIDEO_COMPACT_FFPROBE_BIN", "ffprobe").strip() or "ffprobe"


def _available_cpu_count() -> int:
    """获取当前进程可用 CPU 数量。"""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _resolve_default_vmaf_threads() -> int:
    raw = os.getenv("VIDEO_COMPACT_VMAF_THREADS")
    if raw is None:
        return max(1, _available_cpu_count())
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, _available_cpu_count())


DEFAULT_VMAF_THREADS = _resolve_default_vmaf_threads()
DEFAULT_VMAF_IO_MODE = os.getenv("VIDEO_COMPACT_VMAF_IO_MODE", "auto").strip().lower()
if DEFAULT_VMAF_IO_MODE not in VMAF_IO_MODES:
    DEFAULT_VMAF_IO_MODE = "auto"
DEFAULT_ENCODE_TIMEOUT_SECONDS = int(
    os.getenv("VIDEO_COMPACT_ENCODE_TIMEOUT_SECONDS", "7200")
)
DEFAULT_VMAF_TIMEOUT_SECONDS = int(
    os.getenv("VIDEO_COMPACT_VMAF_TIMEOUT_SECONDS", "1800")
)
DEFAULT_PREP_TIMEOUT_SECONDS = int(
    os.getenv("VIDEO_COMPACT_PREP_TIMEOUT_SECONDS", "1800")
)
DEFAULT_SNR_TIMEOUT_SECONDS = int(os.getenv("VIDEO_COMPACT_SNR_TIMEOUT_SECONDS", "1800"))


class CommandExecutionError(RuntimeError):
    """统一命令执行异常，带上下文和 stderr 摘要。"""


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
    # 可选诊断信息
    warnings: list[str] | None = field(default=None)
    error: str | None = None


def _tail(text: str, limit: int = 1600) -> str:
    """截断长日志，避免错误信息过长。"""
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_command(
    cmd: list[str],
    *,
    timeout_seconds: int | None,
    context: str,
) -> subprocess.CompletedProcess[str]:
    """统一执行外部命令，并在异常时给出可诊断上下文。"""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandExecutionError(
            f"{context} 超时（>{timeout_seconds}s）：{' '.join(cmd)}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise CommandExecutionError(f"{context} 执行失败：{exc}") from exc

    if result.returncode != 0:
        stderr_tail = _tail(result.stderr or "<empty stderr>")
        raise CommandExecutionError(
            f"{context} 失败（exit={result.returncode}）：{stderr_tail}"
        )
    return result


def _ffmpeg_common_prefix() -> list[str]:
    return [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"]


def get_video_info(video_path: str) -> dict[str, Any]:
    """获取视频信息"""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = run_command(cmd, timeout_seconds=60, context="读取视频信息")
    return json.loads(result.stdout)


def encode_video(
    input_path: str,
    output_path: str,
    encoder_key: str,
    param_value: int,
    *,
    timeout_seconds: int = DEFAULT_ENCODE_TIMEOUT_SECONDS,
) -> float:
    """
    使用指定编码器和参数编码视频。

    返回: 编码耗时（秒）
    """
    config = ENCODERS[encoder_key]

    cmd = [
        *_ffmpeg_common_prefix(),
        "-i",
        input_path,
        "-c:v",
        config.codec,
        f"-{config.param_name}",
        str(param_value),
        *config.extra_args,
        output_path,
    ]

    started = time.monotonic()
    run_command(
        cmd,
        timeout_seconds=timeout_seconds,
        context=f"编码失败（encoder={encoder_key}, {config.param_name}={param_value}）",
    )
    return time.monotonic() - started


def _convert_to_y4m(
    input_path: str,
    output_path: str,
    *,
    timeout_seconds: int,
    context: str,
) -> None:
    cmd = [
        *_ffmpeg_common_prefix(),
        "-i",
        input_path,
        "-pix_fmt",
        "yuv420p",
        "-f",
        "yuv4mpegpipe",
        output_path,
    ]
    run_command(cmd, timeout_seconds=timeout_seconds, context=context)


def _normalize_vmaf_io_mode(io_mode: str | None) -> str:
    mode = (io_mode or DEFAULT_VMAF_IO_MODE).strip().lower()
    if mode not in VMAF_IO_MODES:
        raise ValueError(f"未知 VMAF I/O 模式: {mode}")
    return mode


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_vmaf_command(
    reference_path: str,
    distorted_path: str,
    output_json: str,
    *,
    vmaf_threads: int,
) -> list[str]:
    return [
        "vmaf",
        "-r",
        reference_path,
        "-d",
        distorted_path,
        "--json",
        "-o",
        output_json,
        "--threads",
        str(max(1, vmaf_threads)),
        "--feature",
        "psnr_hvs",
        "--feature",
        "float_ssim",
        "--feature",
        "float_ms_ssim",
    ]


def _escape_ffmpeg_filter_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return escaped


@functools.lru_cache(maxsize=8)
def _supports_ffmpeg_libvmaf(ffmpeg_bin: str) -> bool:
    try:
        result = run_command(
            [ffmpeg_bin, "-hide_banner", "-filters"],
            timeout_seconds=30,
            context=f"探测 ffmpeg libvmaf 支持（{ffmpeg_bin}）",
        )
    except CommandExecutionError:
        return False
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return "libvmaf" in text


def _collect_process_output(proc: subprocess.Popen[Any]) -> tuple[str, str]:
    try:
        stdout, stderr = proc.communicate(timeout=0.1)
    except subprocess.TimeoutExpired:
        _kill_process(proc)
        stdout, stderr = proc.communicate()
    return (stdout or "", stderr or "")


def _run_vmaf_via_ffmpeg_libvmaf(
    reference_path: str,
    distorted_path: str,
    output_json: str,
    *,
    vmaf_threads: int,
    vmaf_timeout_seconds: int,
) -> None:
    if not _supports_ffmpeg_libvmaf(FFMPEG_BIN):
        raise CommandExecutionError(
            f"当前 ffmpeg 不支持 libvmaf 过滤器: {FFMPEG_BIN} "
            "(请切换到带 --enable-libvmaf 的 ffmpeg)"
        )

    log_path = _escape_ffmpeg_filter_value(output_json)
    thread_value = max(1, vmaf_threads)
    libvmaf_expr = (
        f"libvmaf=log_fmt=json:log_path='{log_path}':n_threads={thread_value}:"
        "feature=name=psnr_hvs|name=float_ssim|name=float_ms_ssim"
    )
    filter_graph = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS[dist];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]{libvmaf_expr}"
    )
    cmd = [
        *_ffmpeg_common_prefix(),
        "-i",
        distorted_path,
        "-i",
        reference_path,
        "-lavfi",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    run_command(
        cmd,
        timeout_seconds=vmaf_timeout_seconds,
        context="FFmpeg libvmaf 计算",
    )


def _run_vmaf_via_files(
    reference_path: str,
    distorted_path: str,
    output_json: str,
    *,
    vmaf_threads: int,
    vmaf_timeout_seconds: int,
    prep_timeout_seconds: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="video_compact_vmaf_") as tmpdir:
        ref_y4m = os.path.join(tmpdir, "ref.y4m")
        dist_y4m = os.path.join(tmpdir, "dist.y4m")

        _convert_to_y4m(
            reference_path,
            ref_y4m,
            timeout_seconds=prep_timeout_seconds,
            context="参考视频转 Y4M",
        )
        _convert_to_y4m(
            distorted_path,
            dist_y4m,
            timeout_seconds=prep_timeout_seconds,
            context="失真视频转 Y4M",
        )

        run_command(
            _build_vmaf_command(
                ref_y4m,
                dist_y4m,
                output_json,
                vmaf_threads=vmaf_threads,
            ),
            timeout_seconds=vmaf_timeout_seconds,
            context="VMAF 计算",
        )


def _run_vmaf_via_fifo(
    reference_path: str,
    distorted_path: str,
    output_json: str,
    *,
    vmaf_threads: int,
    vmaf_timeout_seconds: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="video_compact_vmaf_fifo_") as tmpdir:
        ref_fifo = os.path.join(tmpdir, "ref.y4m.fifo")
        dist_fifo = os.path.join(tmpdir, "dist.y4m.fifo")
        os.mkfifo(ref_fifo)
        os.mkfifo(dist_fifo)

        vmaf_cmd = _build_vmaf_command(
            ref_fifo,
            dist_fifo,
            output_json,
            vmaf_threads=vmaf_threads,
        )
        ref_cmd = [
            *_ffmpeg_common_prefix(),
            "-i",
            reference_path,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "yuv4mpegpipe",
            ref_fifo,
        ]
        dist_cmd = [
            *_ffmpeg_common_prefix(),
            "-i",
            distorted_path,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "yuv4mpegpipe",
            dist_fifo,
        ]

        proc_items: list[tuple[str, list[str], subprocess.Popen[Any]]] = []
        started = time.monotonic()

        try:
            vmaf_proc = subprocess.Popen(
                vmaf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_items.append(("vmaf", vmaf_cmd, vmaf_proc))

            ref_proc = subprocess.Popen(
                ref_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_items.append(("ffmpeg_ref", ref_cmd, ref_proc))

            dist_proc = subprocess.Popen(
                dist_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc_items.append(("ffmpeg_dist", dist_cmd, dist_proc))
        except Exception as exc:  # noqa: BLE001
            for _, _, proc in proc_items:
                _kill_process(proc)
            raise CommandExecutionError(f"VMAF FIFO 模式启动失败: {exc}") from exc

        while True:
            if time.monotonic() - started > vmaf_timeout_seconds:
                for _, _, proc in proc_items:
                    _kill_process(proc)
                raise CommandExecutionError(f"VMAF FIFO 计算超时（>{vmaf_timeout_seconds}s）")

            status_map = {name: proc.poll() for name, _, proc in proc_items}
            if status_map["vmaf"] is not None:
                if status_map["vmaf"] != 0:
                    break
                # vmaf 退出后，允许编码进程在短时间内自行收尾。
                grace_deadline = time.monotonic() + 3
                while time.monotonic() < grace_deadline:
                    if status_map["ffmpeg_ref"] is not None and status_map["ffmpeg_dist"] is not None:
                        break
                    time.sleep(0.1)
                    status_map = {name: proc.poll() for name, _, proc in proc_items}
                break

            if (
                status_map["ffmpeg_ref"] is not None and status_map["ffmpeg_ref"] != 0
            ) or (
                status_map["ffmpeg_dist"] is not None and status_map["ffmpeg_dist"] != 0
            ):
                break

            time.sleep(0.2)

        errors: list[str] = []
        for name, cmd, proc in proc_items:
            if proc.poll() is None:
                _kill_process(proc)
            stdout, stderr = _collect_process_output(proc)
            if proc.returncode != 0:
                detail = _tail(stderr or stdout or "<empty stdout/stderr>")
                errors.append(
                    f"{name} 失败（exit={proc.returncode}）：{detail}\n命令: {' '.join(cmd)}"
                )

        if errors:
            raise CommandExecutionError("VMAF FIFO 计算失败:\n" + "\n".join(errors))


def calculate_vmaf(
    reference_path: str,
    distorted_path: str,
    output_json: str,
    *,
    vmaf_timeout_seconds: int = DEFAULT_VMAF_TIMEOUT_SECONDS,
    prep_timeout_seconds: int = DEFAULT_PREP_TIMEOUT_SECONDS,
    vmaf_threads: int = DEFAULT_VMAF_THREADS,
    io_mode: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """
    以受控流程计算 VMAF 及相关指标。

    I/O 模式:
    - libvmaf: 直接使用 FFmpeg libvmaf（无中间 y4m）
    - fifo: vmaf CLI + 命名管道
    - file: vmaf CLI + 临时 y4m 文件
    - auto: libvmaf -> fifo -> file 自动回退

    计算指标:
    - VMAF
    - PSNR-HVS
    - SSIM (float_ssim)
    - MS-SSIM (float_ms_ssim)
    """
    _ensure_parent_dir(output_json)
    mode = _normalize_vmaf_io_mode(io_mode)

    if mode in {"auto", "libvmaf"}:
        try:
            _run_vmaf_via_ffmpeg_libvmaf(
                reference_path,
                distorted_path,
                output_json,
                vmaf_threads=vmaf_threads,
                vmaf_timeout_seconds=vmaf_timeout_seconds,
            )
            with open(output_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except CommandExecutionError as exc:
            if mode == "libvmaf":
                raise
            if warnings is not None:
                warnings.append(f"VMAF libvmaf 模式失败，尝试 FIFO 模式: {exc}")

    if mode in {"auto", "fifo"}:
        try:
            _run_vmaf_via_fifo(
                reference_path,
                distorted_path,
                output_json,
                vmaf_threads=vmaf_threads,
                vmaf_timeout_seconds=vmaf_timeout_seconds,
            )
        except CommandExecutionError as exc:
            if mode == "fifo":
                raise
            if warnings is not None:
                warnings.append(f"VMAF FIFO 模式失败，尝试文件模式: {exc}")
            _run_vmaf_via_files(
                reference_path,
                distorted_path,
                output_json,
                vmaf_threads=vmaf_threads,
                vmaf_timeout_seconds=vmaf_timeout_seconds,
                prep_timeout_seconds=prep_timeout_seconds,
            )
    else:
        _run_vmaf_via_files(
            reference_path,
            distorted_path,
            output_json,
            vmaf_threads=vmaf_threads,
            vmaf_timeout_seconds=vmaf_timeout_seconds,
            prep_timeout_seconds=prep_timeout_seconds,
        )

    with open(output_json, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def extract_vmaf_scores(vmaf_result: dict[str, Any]) -> dict[str, float]:
    """从 VMAF 结果中提取所有指标分数"""
    pooled_metrics = vmaf_result.get("pooled_metrics", {})

    def extract_metric(metric_name: str, prefix: str) -> dict[str, float]:
        metric_data = pooled_metrics.get(metric_name, {})
        return {
            f"{prefix}_mean": metric_data.get("mean", 0.0),
            f"{prefix}_min": metric_data.get("min", 0.0),
            f"{prefix}_max": metric_data.get("max", 0.0),
            f"{prefix}_harmonic_mean": metric_data.get("harmonic_mean", 0.0),
        }

    scores: dict[str, float] = {}
    scores.update(extract_metric("vmaf", "vmaf"))
    scores.update(extract_metric("psnr_hvs", "psnr_hvs"))
    scores.update(extract_metric("float_ssim", "ssim"))
    scores.update(extract_metric("float_ms_ssim", "ms_ssim"))
    return scores


def get_video_resolution(video_path: str) -> tuple[int, int]:
    """获取视频分辨率"""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]
    result = run_command(cmd, timeout_seconds=60, context="获取视频分辨率")
    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    return int(stream.get("width", 0)), int(stream.get("height", 0))


def _kill_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _snr_zeros() -> dict[str, float]:
    return {
        "snr_mean": 0.0,
        "snr_min": 0.0,
        "snr_max": 0.0,
        "snr_harmonic_mean": 0.0,
    }


def calculate_snr(
    reference_path: str,
    distorted_path: str,
    *,
    timeout_seconds: int = DEFAULT_SNR_TIMEOUT_SECONDS,
) -> dict[str, float]:
    """
    计算 SNR (Signal-to-Noise Ratio)。

    使用灰度帧逐帧计算 SNR，并增加整体超时控制避免任务挂死。
    """
    width, height = get_video_resolution(reference_path)
    if width == 0 or height == 0:
        raise RuntimeError("无法获取视频分辨率")

    ref_cmd = [
        *_ffmpeg_common_prefix(),
        "-i",
        reference_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    dist_cmd = [
        *_ffmpeg_common_prefix(),
        "-i",
        distorted_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
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

    if ref_proc.stdout is None or dist_proc.stdout is None:
        _kill_process(ref_proc)
        _kill_process(dist_proc)
        raise RuntimeError("SNR 计算初始化失败：无法打开视频流")

    frame_size = width * height
    chunk_size = max(8192, min(frame_size, 1024 * 1024))
    snr_values: list[float] = []
    buffers = {"ref": bytearray(), "dist": bytearray()}
    finished = {"ref": False, "dist": False}
    started = time.monotonic()

    selector = selectors.DefaultSelector()
    ref_fd = ref_proc.stdout.fileno()
    dist_fd = dist_proc.stdout.fileno()
    os.set_blocking(ref_fd, False)
    os.set_blocking(dist_fd, False)
    selector.register(ref_fd, selectors.EVENT_READ, data="ref")
    selector.register(dist_fd, selectors.EVENT_READ, data="dist")

    try:
        while True:
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"SNR 计算超时（>{timeout_seconds}s）")

            events = selector.select(timeout=0.5)
            for key, _ in events:
                stream_name = key.data
                chunk = os.read(key.fd, chunk_size)
                if not chunk:
                    finished[stream_name] = True
                    selector.unregister(key.fd)
                    continue
                buffers[stream_name].extend(chunk)

            while len(buffers["ref"]) >= frame_size and len(buffers["dist"]) >= frame_size:
                ref_data = bytes(buffers["ref"][:frame_size])
                dist_data = bytes(buffers["dist"][:frame_size])
                del buffers["ref"][:frame_size]
                del buffers["dist"][:frame_size]

                ref_frame = np.frombuffer(ref_data, dtype=np.uint8).astype(np.float64)
                dist_frame = np.frombuffer(dist_data, dtype=np.uint8).astype(np.float64)

                signal_power = np.mean(ref_frame ** 2)
                noise_power = np.mean((ref_frame - dist_frame) ** 2)
                snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
                snr_values.append(float(snr_db))

            if finished["ref"] and finished["dist"]:
                break
    finally:
        selector.close()
        _kill_process(ref_proc)
        _kill_process(dist_proc)

    if not snr_values:
        raise RuntimeError("无法计算 SNR：没有有效帧")

    snr_array = np.array(snr_values, dtype=np.float64)
    finite_snr = snr_array[np.isfinite(snr_array)]
    if len(finite_snr) > 0 and np.all(finite_snr > 0):
        harmonic_mean = len(finite_snr) / np.sum(1.0 / finite_snr)
    else:
        harmonic_mean = np.mean(finite_snr) if len(finite_snr) > 0 else 0.0

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
    *,
    vmaf_threads: int = DEFAULT_VMAF_THREADS,
    vmaf_io_mode: str | None = None,
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
    warnings: list[str] = []

    output_filename = f"{input_name}_{encoder_key}_{config.param_name}{param_value}.mp4"
    output_path = os.path.join(output_dir, "encoded", output_filename)
    vmaf_json = os.path.join(
        output_dir, f"vmaf_{encoder_key}_{config.param_name}{param_value}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  编码中: {config.name}, {config.param_name}={param_value}")
    encode_time = encode_video(input_path, output_path, encoder_key, param_value)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    print("  计算质量指标 (VMAF/PSNR-HVS/SSIM/MS-SSIM)...")
    vmaf_result = calculate_vmaf(
        input_path,
        output_path,
        vmaf_json,
        vmaf_threads=vmaf_threads,
        io_mode=vmaf_io_mode,
        warnings=warnings,
    )
    all_scores = extract_vmaf_scores(vmaf_result)

    print("  计算 SNR...")
    try:
        snr_scores = calculate_snr(input_path, output_path)
    except Exception as exc:  # noqa: BLE001
        if strict_mode:
            raise RuntimeError(f"SNR 计算失败: {exc}") from exc
        warning = f"SNR 计算失败，已按 no-strict 降级处理: {exc}"
        warnings.append(warning)
        snr_scores = _snr_zeros()
    all_scores.update(snr_scores)

    if strict_mode:
        required_metrics = ["vmaf", "psnr_hvs", "ssim", "ms_ssim", "snr"]
        validate_metrics(all_scores, required_metrics)
        print("  ✓ 所有指标计算完成")
    elif warnings:
        print(f"  ⚠ {warnings[-1]}")

    compression_ratio = round(output_size / input_size * 100, 2) if input_size > 0 else 0.0
    return BenchmarkResult(
        encoder=encoder_key,
        param_name=config.param_name,
        param_value=param_value,
        input_file=input_path,
        output_file=output_path,
        input_size_mb=round(input_size, 2),
        output_size_mb=round(output_size, 2),
        compression_ratio=compression_ratio,
        encode_time_seconds=round(encode_time, 2),
        warnings=warnings or None,
        error=None,
        **{k: round(v, 4) for k, v in all_scores.items()},
    )


def run_benchmark(
    input_path: str,
    output_dir: str,
    encoders: list[str],
    param_step: int = 5,
    strict_mode: bool = True,
    *,
    vmaf_threads: int = DEFAULT_VMAF_THREADS,
    vmaf_io_mode: str | None = None,
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
                    input_path,
                    output_dir,
                    encoder_key,
                    param_value,
                    strict_mode,
                    vmaf_threads=vmaf_threads,
                    vmaf_io_mode=vmaf_io_mode,
                )
                results.append(result)
                summary = (
                    f"    VMAF: {result.vmaf_mean:.2f}, SSIM: {result.ssim_mean:.4f}, "
                    f"SNR: {result.snr_mean:.2f} dB, 大小: {result.output_size_mb:.2f} MB"
                )
                if result.warnings:
                    summary += f", 警告: {result.warnings[-1]}"
                print(summary)
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


def load_results(input_path: str) -> list[BenchmarkResult]:
    """从 JSON 文件加载结果"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    loaded: list[BenchmarkResult] = []
    for result in data.get("results", []):
        if "warnings" not in result:
            result["warnings"] = None
        if "error" not in result:
            result["error"] = None
        loaded.append(BenchmarkResult(**result))
    return loaded


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
    parser.add_argument(
        "--vmaf-threads",
        type=int,
        default=DEFAULT_VMAF_THREADS,
        help=f"VMAF 线程数（默认可用核数: {DEFAULT_VMAF_THREADS}）",
    )
    parser.add_argument(
        "--vmaf-io-mode",
        default=DEFAULT_VMAF_IO_MODE,
        choices=sorted(VMAF_IO_MODES),
        help="VMAF 输入模式：auto(优先libvmaf)、libvmaf(FFmpeg滤镜)、fifo(仅管道)、file(落盘Y4M)",
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
    print(f"VMAF线程: {max(1, args.vmaf_threads)}")
    print(f"VMAF I/O模式: {_normalize_vmaf_io_mode(args.vmaf_io_mode)}")
    
    # 运行评估
    try:
        results = run_benchmark(
            args.input,
            output_dir,
            args.encoders,
            args.step,
            strict_mode,
            vmaf_threads=max(1, args.vmaf_threads),
            vmaf_io_mode=args.vmaf_io_mode,
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
