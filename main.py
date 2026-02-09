#!/usr/bin/env python3
"""统一命令行入口：评估与可视化启动。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SUPPORTED_ENCODERS = ("hevc", "av1")
VMAF_IO_MODE_CHOICES = ("auto", "libvmaf", "fifo", "file")


def _default_vmaf_threads() -> int:
    raw = os.getenv("VIDEO_COMPACT_VMAF_THREADS")
    if raw is None:
        return max(1, os.cpu_count() or 1)
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, os.cpu_count() or 1)


DEFAULT_VMAF_THREADS = _default_vmaf_threads()
DEFAULT_VMAF_IO_MODE = os.getenv("VIDEO_COMPACT_VMAF_IO_MODE", "auto").strip().lower()
if DEFAULT_VMAF_IO_MODE not in VMAF_IO_MODE_CHOICES:
    DEFAULT_VMAF_IO_MODE = "auto"


def build_parser() -> argparse.ArgumentParser:
    """构建顶层参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="video-compact",
        description="视频质量评估工具统一入口",
        epilog=(
            "示例:\n"
            "  python main.py benchmark video.mp4 -e hevc av1 -s 5\n"
            "  python main.py autotune sample1.mp4 sample2.mp4 --target-vmaf 95\n"
            "  python main.py web --host 0.0.0.0 --port 8501"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="启动视频评估",
        description="运行编码与质量评估（与 benchmark.py 参数语义对齐）",
    )
    benchmark_parser.add_argument("input", help="输入视频文件路径")
    benchmark_parser.add_argument(
        "--encoders",
        "-e",
        nargs="+",
        default=["hevc", "av1"],
        choices=list(SUPPORTED_ENCODERS),
        help="要测试的编码器",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="输出根目录（结果将写入 <output>/<输入文件名>/）",
    )
    benchmark_parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=5,
        help="参数步长",
    )
    benchmark_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="禁用严格模式（允许部分指标计算失败）",
    )
    benchmark_parser.add_argument(
        "--vmaf-threads",
        type=int,
        default=DEFAULT_VMAF_THREADS,
        help=f"VMAF 线程数（默认可用核数: {DEFAULT_VMAF_THREADS}）",
    )
    benchmark_parser.add_argument(
        "--vmaf-io-mode",
        default=DEFAULT_VMAF_IO_MODE,
        choices=VMAF_IO_MODE_CHOICES,
        help="VMAF 输入模式：auto(优先libvmaf)、libvmaf(FFmpeg滤镜)、fifo(仅管道)、file(落盘Y4M)",
    )
    benchmark_parser.set_defaults(handler=cmd_benchmark)

    web_parser = subparsers.add_parser(
        "web",
        help="启动可视化页面",
        description="通过 Streamlit 启动 web_app.py",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（映射到 --server.address）",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="监听端口（映射到 --server.port）",
    )
    web_parser.add_argument(
        "--server-headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用无头模式（默认 true）",
    )
    web_parser.set_defaults(handler=cmd_web)

    autotune_parser = subparsers.add_parser(
        "autotune",
        help="自动筛选推荐参数",
        description="执行两阶段粗到细筛选，输出每视频每编码器推荐参数",
    )
    autotune_parser.add_argument(
        "inputs",
        nargs="+",
        help="输入视频路径（可多个）",
    )
    autotune_parser.add_argument(
        "--encoders",
        "-e",
        nargs="+",
        default=["hevc", "av1"],
        choices=list(SUPPORTED_ENCODERS),
        help="要筛选的编码器",
    )
    autotune_parser.add_argument(
        "--target-vmaf",
        type=float,
        default=95.0,
        help="目标 VMAF 阈值（默认 95）",
    )
    autotune_parser.add_argument(
        "--coarse-duration",
        type=int,
        default=10,
        help="粗扫片段时长（秒，默认 10）",
    )
    autotune_parser.add_argument(
        "--coarse-scale",
        type=int,
        default=1280,
        help="粗扫缩放宽度（默认 1280）",
    )
    autotune_parser.add_argument(
        "--output",
        "-o",
        default="results_autotune",
        help="输出目录（默认 results_autotune）",
    )
    autotune_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="并发任务数（默认 1）",
    )
    autotune_parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否开启严格模式（默认 false）",
    )
    autotune_parser.add_argument(
        "--vmaf-threads",
        type=int,
        default=DEFAULT_VMAF_THREADS,
        help=f"VMAF 线程数（默认可用核数: {DEFAULT_VMAF_THREADS}）",
    )
    autotune_parser.add_argument(
        "--vmaf-io-mode",
        default=DEFAULT_VMAF_IO_MODE,
        choices=VMAF_IO_MODE_CHOICES,
        help="VMAF 输入模式：auto(优先libvmaf)、libvmaf(FFmpeg滤镜)、fifo(仅管道)、file(落盘Y4M)",
    )
    autotune_parser.set_defaults(handler=cmd_autotune)

    return parser


def cmd_benchmark(args: argparse.Namespace) -> int:
    """桥接 benchmark.py 的核心评估逻辑。"""
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        return 1

    try:
        from benchmark import run_benchmark, save_results
    except ModuleNotFoundError as exc:
        print(f"错误: 缺少依赖，无法启动评估: {exc}")
        print("请先安装项目依赖（例如 `uv sync` 或 `pip install -e .`）。")
        return 1

    input_name = Path(args.input).stem
    output_dir = os.path.join(args.output, input_name)
    os.makedirs(output_dir, exist_ok=True)

    strict_mode = not args.no_strict

    print(f"输入视频: {args.input}")
    print(f"输出目录: {output_dir}")
    print(f"编码器: {', '.join(args.encoders)}")
    print(f"参数步长: {args.step}")
    print(f"严格模式: {'是' if strict_mode else '否'}")
    print(f"VMAF 线程: {max(1, args.vmaf_threads)}")
    print(f"VMAF I/O 模式: {args.vmaf_io_mode}")

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
    except RuntimeError as exc:
        print(f"\n错误: {exc}")
        print("严格模式下，所有指标必须可计算。使用 --no-strict 禁用严格模式。")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n错误: 评估执行失败: {exc}")
        return 1

    results_path = os.path.join(output_dir, "results.json")
    save_results(results, results_path)

    print("\n" + "=" * 100)
    print("评估完成，结果汇总:")
    print("=" * 100)
    print(
        f"{'编码器':<8} {'参数':<6} {'VMAF':<8} {'SSIM':<8} {'MS-SSIM':<10} "
        f"{'PSNR-HVS':<10} {'SNR(dB)':<10} {'大小(MB)':<10} {'压缩比':<8}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.encoder:<8} {result.param_value:<6} {result.vmaf_mean:<8.2f} "
            f"{result.ssim_mean:<8.4f} {result.ms_ssim_mean:<10.4f} "
            f"{result.psnr_hvs_mean:<10.2f} {result.snr_mean:<10.2f} "
            f"{result.output_size_mb:<10.2f} {result.compression_ratio:<8.2f}%"
        )

    return 0


def cmd_web(args: argparse.Namespace) -> int:
    """启动 Streamlit 可视化页面。"""
    web_app = Path(__file__).with_name("web_app.py")
    if not web_app.exists():
        print(f"错误: 未找到 Web 入口文件 {web_app}")
        return 1

    cmd = [
        "streamlit",
        "run",
        str(web_app),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.server_headless else "false",
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("错误: 未找到 streamlit 命令，请先安装依赖（例如 `uv sync`）。")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"错误: 启动 Web 失败: {exc}")
        return 1

    return result.returncode


def _print_autotune_progress(event: dict) -> None:
    phase = event.get("phase", "unknown")
    video = event.get("video", "-")
    encoder = event.get("encoder")
    crf = event.get("crf")
    status = event.get("status")
    message = event.get("message", "")

    parts = [f"[{phase}]"]
    if status:
        parts.append(f"[{status}]")
    parts.append(video)
    if encoder:
        parts.append(encoder)
    if crf is not None:
        parts.append(f"crf={crf}")
    if message:
        parts.append(f"- {message}")
    print(" ".join(parts))


def cmd_autotune(args: argparse.Namespace) -> int:
    """自动筛选推荐参数。"""
    missing_inputs = [p for p in args.inputs if not os.path.exists(p)]
    if missing_inputs:
        print("错误: 以下输入文件不存在:")
        for path in missing_inputs:
            print(f"  - {path}")
        return 1
    if args.jobs < 1:
        print("错误: --jobs 必须 >= 1")
        return 1
    if args.coarse_duration < 1:
        print("错误: --coarse-duration 必须 >= 1")
        return 1
    if args.coarse_scale < 16:
        print("错误: --coarse-scale 必须 >= 16")
        return 1

    try:
        from autotune import print_summary, run_autotune
    except ModuleNotFoundError as exc:
        print(f"错误: 缺少依赖，无法启动 autotune: {exc}")
        print("请先安装项目依赖（例如 `uv sync` 或 `pip install -e .`）。")
        return 1

    print(f"输入视频数量: {len(args.inputs)}")
    print(f"编码器: {', '.join(args.encoders)}")
    print(f"目标 VMAF: {args.target_vmaf}")
    print(f"粗扫时长: {args.coarse_duration}s")
    print(f"粗扫缩放宽度: {args.coarse_scale}")
    print(f"并发数: {args.jobs}")
    print(f"严格模式: {'是' if args.strict else '否'}")
    print(f"VMAF 线程: {max(1, args.vmaf_threads)}")
    print(f"VMAF I/O 模式: {args.vmaf_io_mode}")
    print(f"输出目录: {args.output}")

    try:
        summary = run_autotune(
            inputs=args.inputs,
            output_root=args.output,
            encoders=args.encoders,
            target_vmaf=args.target_vmaf,
            coarse_duration=args.coarse_duration,
            coarse_scale=args.coarse_scale,
            strict_mode=args.strict,
            vmaf_threads=max(1, args.vmaf_threads),
            vmaf_io_mode=args.vmaf_io_mode,
            jobs=args.jobs,
            progress_cb=_print_autotune_progress,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n错误: autotune 执行失败: {exc}")
        return 1

    print_summary(summary)
    if summary["stats"]["successful_recommendations"] == 0:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
