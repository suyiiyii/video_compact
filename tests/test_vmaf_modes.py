import json
from pathlib import Path

import pytest

import benchmark


def _write_fake_vmaf_json(path: str, mean: float) -> None:
    payload = {
        "pooled_metrics": {
            "vmaf": {"mean": mean},
            "psnr_hvs": {"mean": 40.0},
            "float_ssim": {"mean": 0.99},
            "float_ms_ssim": {"mean": 0.99},
        }
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def test_calculate_vmaf_file_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    called: dict[str, int] = {"libvmaf": 0, "file": 0, "fifo": 0}

    def fake_file(
        reference_path: str,
        distorted_path: str,
        output_json: str,
        *,
        vmaf_threads: int,
        vmaf_timeout_seconds: int,
        prep_timeout_seconds: int,
    ) -> None:
        called["file"] += 1
        assert reference_path == "ref.mp4"
        assert distorted_path == "dist.mp4"
        assert vmaf_threads == 12
        _write_fake_vmaf_json(output_json, 95.2)

    def fake_fifo(*args, **kwargs):  # noqa: ANN002, ANN003
        called["fifo"] += 1
        raise AssertionError("file 模式不应调用 fifo")

    def fake_libvmaf(*args, **kwargs):  # noqa: ANN002, ANN003
        called["libvmaf"] += 1
        raise AssertionError("file 模式不应调用 libvmaf")

    monkeypatch.setattr(benchmark, "_run_vmaf_via_ffmpeg_libvmaf", fake_libvmaf)
    monkeypatch.setattr(benchmark, "_run_vmaf_via_files", fake_file)
    monkeypatch.setattr(benchmark, "_run_vmaf_via_fifo", fake_fifo)

    output_json = tmp_path / "vmaf.json"
    result = benchmark.calculate_vmaf(
        "ref.mp4",
        "dist.mp4",
        str(output_json),
        io_mode="file",
        vmaf_threads=12,
    )

    assert called["libvmaf"] == 0
    assert called["file"] == 1
    assert called["fifo"] == 0
    assert result["pooled_metrics"]["vmaf"]["mean"] == 95.2


def test_calculate_vmaf_auto_fallback_to_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def fake_libvmaf(*args, **kwargs):  # noqa: ANN002, ANN003
        raise benchmark.CommandExecutionError("libvmaf failed")

    def fake_fifo(*args, **kwargs):  # noqa: ANN002, ANN003
        raise benchmark.CommandExecutionError("fifo failed")

    def fake_file(
        reference_path: str,
        distorted_path: str,
        output_json: str,
        *,
        vmaf_threads: int,
        vmaf_timeout_seconds: int,
        prep_timeout_seconds: int,
    ) -> None:
        _write_fake_vmaf_json(output_json, 94.8)

    monkeypatch.setattr(benchmark, "_run_vmaf_via_ffmpeg_libvmaf", fake_libvmaf)
    monkeypatch.setattr(benchmark, "_run_vmaf_via_fifo", fake_fifo)
    monkeypatch.setattr(benchmark, "_run_vmaf_via_files", fake_file)

    output_json = tmp_path / "vmaf.json"
    warnings: list[str] = []
    result = benchmark.calculate_vmaf(
        "ref.mp4",
        "dist.mp4",
        str(output_json),
        io_mode="auto",
        warnings=warnings,
    )

    assert result["pooled_metrics"]["vmaf"]["mean"] == 94.8
    assert len(warnings) == 2
    assert "libvmaf 模式失败" in warnings[0]
    assert "FIFO 模式失败" in warnings[1]


def test_calculate_vmaf_libvmaf_mode_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_libvmaf(*args, **kwargs):  # noqa: ANN002, ANN003
        raise benchmark.CommandExecutionError("libvmaf failed")

    monkeypatch.setattr(benchmark, "_run_vmaf_via_ffmpeg_libvmaf", fake_libvmaf)

    with pytest.raises(benchmark.CommandExecutionError):
        benchmark.calculate_vmaf(
            "ref.mp4",
            "dist.mp4",
            str(tmp_path / "vmaf.json"),
            io_mode="libvmaf",
        )


def test_calculate_vmaf_fifo_mode_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_fifo(*args, **kwargs):  # noqa: ANN002, ANN003
        raise benchmark.CommandExecutionError("fifo failed")

    monkeypatch.setattr(benchmark, "_run_vmaf_via_fifo", fake_fifo)

    with pytest.raises(benchmark.CommandExecutionError):
        benchmark.calculate_vmaf("ref.mp4", "dist.mp4", str(tmp_path / "vmaf.json"), io_mode="fifo")


def test_normalize_vmaf_io_mode():
    assert benchmark._normalize_vmaf_io_mode("AUTO") == "auto"
    assert benchmark._normalize_vmaf_io_mode(" libvmaf ") == "libvmaf"
    assert benchmark._normalize_vmaf_io_mode(" fifo ") == "fifo"
    with pytest.raises(ValueError):
        benchmark._normalize_vmaf_io_mode("unknown")
