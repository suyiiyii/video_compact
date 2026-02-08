#!/usr/bin/env python3
"""自动化参数筛选流程（粗扫 -> 精扫）。"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from benchmark import (
    ENCODERS,
    BenchmarkResult,
    CommandExecutionError,
    run_command,
    run_single_benchmark,
)

COARSE_GRID: dict[str, list[int]] = {
    "hevc": [22, 26, 30, 34],
    "av1": [30, 36, 42, 48],
}

FINE_SPAN: dict[str, int] = {
    "hevc": 2,
    "av1": 3,
}


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class StageTask:
    """单个编码候选任务。"""

    stage: str
    video: str
    input_path: str
    output_dir: str
    encoder: str
    crf: int
    strict_mode: bool


def _emit_progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    callback(event)


def build_fine_grid(encoder: str, coarse_best_crf: int) -> list[int]:
    """基于粗扫结果构建精扫网格。"""
    if encoder not in ENCODERS:
        raise ValueError(f"未知编码器: {encoder}")
    span = FINE_SPAN.get(encoder, 2)
    min_crf, max_crf = ENCODERS[encoder].param_range
    values = range(coarse_best_crf - span, coarse_best_crf + span + 1)
    return sorted({v for v in values if min_crf <= v <= max_crf})


def _to_candidate(stage: str, result: BenchmarkResult) -> dict[str, Any]:
    return {
        "stage": stage,
        "encoder": result.encoder,
        "crf": result.param_value,
        "status": "ok",
        "vmaf_mean": result.vmaf_mean,
        "output_size_mb": result.output_size_mb,
        "compression_ratio": result.compression_ratio,
        "encode_time_seconds": result.encode_time_seconds,
        "warnings": result.warnings,
        "output_file": result.output_file,
    }


def rank_candidates(
    candidates: list[dict[str, Any]],
    target_vmaf: float,
) -> list[dict[str, Any]]:
    """对候选结果排序并标注 rank。"""
    ok_candidates = [c for c in candidates if c.get("status") == "ok"]

    def sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        vmaf_mean = float(item["vmaf_mean"])
        size_mb = float(item["output_size_mb"])
        crf = float(item["crf"])
        if vmaf_mean >= target_vmaf:
            return (0.0, size_mb, -vmaf_mean, crf)
        return (1.0, -vmaf_mean, size_mb, crf)

    ranked: list[dict[str, Any]] = []
    for idx, item in enumerate(sorted(ok_candidates, key=sort_key), start=1):
        candidate = dict(item)
        candidate["rank"] = idx
        candidate["meets_target"] = candidate["vmaf_mean"] >= target_vmaf
        ranked.append(candidate)
    return ranked


def select_best_candidate(
    candidates: list[dict[str, Any]],
    target_vmaf: float,
) -> tuple[dict[str, Any] | None, bool]:
    """
    甜点选择规则:
    1) 先选 VMAF>=target 中体积最小
    2) 若都不满足阈值，回退到 VMAF 最大
    """
    ranked = rank_candidates(candidates, target_vmaf)
    if not ranked:
        return None, True
    best = ranked[0]
    threshold_unmet = not bool(best.get("meets_target", False))
    return best, threshold_unmet


def _evaluate_task(task: StageTask) -> dict[str, Any]:
    try:
        result = run_single_benchmark(
            task.input_path,
            task.output_dir,
            task.encoder,
            task.crf,
            strict_mode=task.strict_mode,
        )
        return _to_candidate(task.stage, result)
    except Exception as exc:  # noqa: BLE001
        return {
            "stage": task.stage,
            "encoder": task.encoder,
            "crf": task.crf,
            "status": "failed",
            "error": str(exc),
        }


def _run_stage(
    *,
    stage: str,
    video_path: str,
    input_path: str,
    output_dir: str,
    encoder: str,
    crf_values: list[int],
    strict_mode: bool,
    jobs: int,
    progress_cb: ProgressCallback | None,
) -> list[dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        StageTask(
            stage=stage,
            video=video_path,
            input_path=input_path,
            output_dir=output_dir,
            encoder=encoder,
            crf=crf,
            strict_mode=strict_mode,
        )
        for crf in crf_values
    ]

    _emit_progress(
        progress_cb,
        {
            "phase": stage,
            "video": video_path,
            "encoder": encoder,
            "message": f"开始 {stage}，CRF={crf_values}",
        },
    )

    results: list[dict[str, Any]] = []
    if jobs <= 1:
        for task in tasks:
            _emit_progress(
                progress_cb,
                {
                    "phase": stage,
                    "video": video_path,
                    "encoder": encoder,
                    "crf": task.crf,
                    "message": "评估中",
                },
            )
            results.append(_evaluate_task(task))
        return results

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_task = {executor.submit(_evaluate_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            _emit_progress(
                progress_cb,
                {
                    "phase": stage,
                    "video": video_path,
                    "encoder": encoder,
                    "crf": task.crf,
                    "status": result.get("status"),
                    "message": "完成" if result.get("status") == "ok" else "失败",
                    "error": result.get("error"),
                },
            )

    return results


def _prepare_coarse_clip(
    input_path: str,
    clip_path: str,
    *,
    duration_seconds: int,
    scale_width: int,
) -> None:
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-t",
        str(duration_seconds),
        "-vf",
        f"scale={scale_width}:-2:flags=lanczos",
        "-pix_fmt",
        "yuv420p",
        "-an",
        clip_path,
    ]
    run_command(
        cmd,
        timeout_seconds=max(120, duration_seconds * 30),
        context=f"生成粗扫片段失败: {input_path}",
    )


def _safe_name(input_path: str, index: int) -> str:
    stem = Path(input_path).stem
    sanitized = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return f"{index:02d}_{sanitized}"


def _build_encoder_ranges(videos: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    ranges: dict[str, dict[str, float] | None] = {}
    for encoder in ENCODERS:
        crfs: list[float] = []
        for video in videos:
            rec = video.get("encoders", {}).get(encoder, {}).get("recommendation")
            if rec and rec.get("crf") is not None:
                crfs.append(float(rec["crf"]))
        if not crfs:
            ranges[encoder] = None
            continue
        ranges[encoder] = {
            "min_crf": min(crfs),
            "max_crf": max(crfs),
        }
    return ranges


def generate_markdown_report(summary: dict[str, Any], output_path: str) -> None:
    lines: list[str] = []
    lines.append("# AutoTune 报告")
    lines.append("")
    lines.append(f"- 运行时间: {summary['created_at']}")
    lines.append(f"- 目标 VMAF: {summary['target_vmaf']}")
    lines.append(f"- 视频数: {summary['stats']['videos_total']}")
    lines.append(
        f"- 成功推荐数: {summary['stats']['successful_recommendations']}/"
        f"{summary['stats']['recommendations_total']}"
    )
    lines.append("")
    lines.append("## 结果总览")
    lines.append("")
    for video in summary["videos"]:
        lines.append(f"### {video['input']}")
        if video.get("errors"):
            for err in video["errors"]:
                lines.append(f"- 错误: {err}")
        for encoder, encoder_data in video.get("encoders", {}).items():
            recommendation = encoder_data.get("recommendation")
            if not recommendation:
                lines.append(f"- `{encoder}`: 无可用推荐")
                continue
            suffix = "（未达到 VMAF 阈值，已回退）" if recommendation["threshold_unmet"] else ""
            lines.append(
                f"- `{encoder}`: CRF `{recommendation['crf']}`, "
                f"VMAF `{recommendation['vmaf_mean']:.2f}`, "
                f"大小 `{recommendation['output_size_mb']:.2f} MB` {suffix}"
            )
        lines.append("")

    lines.append("## 推荐区间")
    lines.append("")
    for encoder, data in summary.get("encoder_recommendation_ranges", {}).items():
        if not data:
            lines.append(f"- `{encoder}`: 无法生成区间")
            continue
        lines.append(f"- `{encoder}`: CRF {data['min_crf']:.0f} ~ {data['max_crf']:.0f}")
    lines.append("")

    lines.append("## 失败样本")
    lines.append("")
    has_failure = False
    for video in summary["videos"]:
        for encoder, encoder_data in video.get("encoders", {}).items():
            for stage_name in ("coarse", "fine"):
                stage = encoder_data.get(stage_name, {})
                for candidate in stage.get("candidates", []):
                    if candidate.get("status") == "failed":
                        has_failure = True
                        lines.append(
                            f"- `{video['input']}` / `{encoder}` / `{stage_name}` "
                            f"CRF `{candidate.get('crf')}`: {candidate.get('error')}"
                        )
    if not has_failure:
        lines.append("- 无")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def run_autotune(
    *,
    inputs: list[str],
    output_root: str,
    encoders: list[str],
    target_vmaf: float = 95.0,
    coarse_duration: int = 10,
    coarse_scale: int = 1280,
    strict_mode: bool = False,
    jobs: int = 1,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    """执行自动筛选流程并产出 summary/report。"""
    if not inputs:
        raise ValueError("inputs 不能为空")
    if not encoders:
        raise ValueError("encoders 不能为空")
    for encoder in encoders:
        if encoder not in ENCODERS:
            raise ValueError(f"不支持的编码器: {encoder}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_vmaf": target_vmaf,
        "output_dir": str(run_dir),
        "config": {
            "encoders": encoders,
            "coarse_duration": coarse_duration,
            "coarse_scale": coarse_scale,
            "strict_mode": strict_mode,
            "jobs": jobs,
            "coarse_grid": {k: v for k, v in COARSE_GRID.items() if k in encoders},
            "fine_span": {k: v for k, v in FINE_SPAN.items() if k in encoders},
        },
        "videos": [],
    }

    total_recommendations = 0
    successful_recommendations = 0
    videos_succeeded = 0

    for idx, input_path in enumerate(inputs, start=1):
        video_entry: dict[str, Any] = {
            "input": input_path,
            "encoders": {},
            "errors": [],
        }
        summary["videos"].append(video_entry)

        if not os.path.exists(input_path):
            video_entry["errors"].append("输入文件不存在")
            _emit_progress(
                progress_cb,
                {
                    "phase": "validate",
                    "video": input_path,
                    "status": "failed",
                    "message": "输入文件不存在",
                },
            )
            continue

        sample_dir = run_dir / _safe_name(input_path, idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        coarse_clip_path = str(sample_dir / "coarse_clip.mp4")

        try:
            _emit_progress(
                progress_cb,
                {
                    "phase": "coarse_prepare",
                    "video": input_path,
                    "message": f"生成粗扫片段 ({coarse_duration}s/{coarse_scale}w)",
                },
            )
            _prepare_coarse_clip(
                input_path,
                coarse_clip_path,
                duration_seconds=coarse_duration,
                scale_width=coarse_scale,
            )
        except (CommandExecutionError, Exception) as exc:  # noqa: BLE001
            video_entry["errors"].append(f"粗扫片段生成失败: {exc}")
            continue

        video_has_recommendation = False
        for encoder in encoders:
            encoder_dir = sample_dir / encoder
            coarse_dir = str(encoder_dir / "coarse")
            fine_dir = str(encoder_dir / "fine")

            coarse_candidates = _run_stage(
                stage="coarse",
                video_path=input_path,
                input_path=coarse_clip_path,
                output_dir=coarse_dir,
                encoder=encoder,
                crf_values=COARSE_GRID[encoder],
                strict_mode=strict_mode,
                jobs=jobs,
                progress_cb=progress_cb,
            )
            coarse_ranked = rank_candidates(coarse_candidates, target_vmaf)
            coarse_best, coarse_unmet = select_best_candidate(coarse_candidates, target_vmaf)

            fine_candidates: list[dict[str, Any]] = []
            fine_ranked: list[dict[str, Any]] = []
            fine_best: dict[str, Any] | None = None
            fine_unmet = True
            fine_grid: list[int] = []
            if coarse_best:
                fine_grid = build_fine_grid(encoder, int(coarse_best["crf"]))
                fine_candidates = _run_stage(
                    stage="fine",
                    video_path=input_path,
                    input_path=input_path,
                    output_dir=fine_dir,
                    encoder=encoder,
                    crf_values=fine_grid,
                    strict_mode=strict_mode,
                    jobs=jobs,
                    progress_cb=progress_cb,
                )
                fine_ranked = rank_candidates(fine_candidates, target_vmaf)
                fine_best, fine_unmet = select_best_candidate(fine_candidates, target_vmaf)

            final_choice = fine_best or coarse_best
            threshold_unmet = fine_unmet if fine_best else coarse_unmet
            source_stage = "fine" if fine_best else "coarse"
            recommendation = None
            total_recommendations += 1
            if final_choice:
                recommendation = {
                    "encoder": encoder,
                    "crf": final_choice["crf"],
                    "vmaf_mean": final_choice["vmaf_mean"],
                    "output_size_mb": final_choice["output_size_mb"],
                    "compression_ratio": final_choice["compression_ratio"],
                    "threshold_unmet": threshold_unmet,
                    "source_stage": source_stage,
                }
                successful_recommendations += 1
                video_has_recommendation = True

            video_entry["encoders"][encoder] = {
                "coarse": {
                    "grid": COARSE_GRID[encoder],
                    "candidates": coarse_candidates,
                    "ranked": coarse_ranked,
                    "best": coarse_best,
                    "threshold_unmet": coarse_unmet,
                },
                "fine": {
                    "grid": fine_grid,
                    "candidates": fine_candidates,
                    "ranked": fine_ranked,
                    "best": fine_best,
                    "threshold_unmet": fine_unmet,
                },
                "recommendation": recommendation,
            }

            _emit_progress(
                progress_cb,
                {
                    "phase": "recommendation",
                    "video": input_path,
                    "encoder": encoder,
                    "message": "推荐已生成" if recommendation else "推荐失败",
                    "recommendation": recommendation,
                },
            )

        if video_has_recommendation:
            videos_succeeded += 1

    summary["encoder_recommendation_ranges"] = _build_encoder_ranges(summary["videos"])
    summary["stats"] = {
        "videos_total": len(inputs),
        "videos_succeeded": videos_succeeded,
        "recommendations_total": total_recommendations,
        "successful_recommendations": successful_recommendations,
    }

    summary_path = run_dir / "autotune_summary.json"
    report_path = run_dir / "autotune_report.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    generate_markdown_report(summary, str(report_path))

    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    return summary


def format_recommendation_line(encoder_data: dict[str, Any]) -> str:
    recommendation = encoder_data.get("recommendation")
    if not recommendation:
        return "无可用推荐"
    suffix = " (未达到阈值，已回退)" if recommendation["threshold_unmet"] else ""
    return (
        f"CRF={recommendation['crf']}, "
        f"VMAF={recommendation['vmaf_mean']:.2f}, "
        f"大小={recommendation['output_size_mb']:.2f}MB{suffix}"
    )


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 100)
    print("AutoTune 结果")
    print("=" * 100)
    for video in summary["videos"]:
        print(f"\n视频: {video['input']}")
        if video.get("errors"):
            for error in video["errors"]:
                print(f"  错误: {error}")
            continue
        for encoder, encoder_data in video.get("encoders", {}).items():
            print(f"  {encoder}: {format_recommendation_line(encoder_data)}")

    print("\n推荐区间:")
    for encoder, value in summary.get("encoder_recommendation_ranges", {}).items():
        if not value:
            print(f"  {encoder}: 无")
        else:
            print(f"  {encoder}: CRF {value['min_crf']:.0f} ~ {value['max_crf']:.0f}")

    print(f"\nAUTOTUNE_SUMMARY_PATH: {summary['summary_path']}")
    print(f"AUTOTUNE_REPORT_PATH: {summary['report_path']}")
