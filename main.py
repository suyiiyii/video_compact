#!/usr/bin/env python3
"""统一命令行入口：评估与可视化启动。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SUPPORTED_ENCODERS = ("hevc", "av1")


def build_parser() -> argparse.ArgumentParser:
    """构建顶层参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="video-compact",
        description="视频质量评估工具统一入口",
        epilog=(
            "示例:\n"
            "  python main.py benchmark video.mp4 -e hevc av1 -s 5\n"
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

    try:
        results = run_benchmark(
            args.input,
            output_dir,
            args.encoders,
            args.step,
            strict_mode,
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


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
