from autotune import build_fine_grid, rank_candidates, select_best_candidate


def test_select_best_candidate_prefers_smallest_size_when_target_met():
    candidates = [
        {"status": "ok", "crf": 22, "vmaf_mean": 95.1, "output_size_mb": 10.0, "compression_ratio": 10.0},
        {"status": "ok", "crf": 24, "vmaf_mean": 95.0, "output_size_mb": 8.0, "compression_ratio": 8.0},
        {"status": "ok", "crf": 26, "vmaf_mean": 94.9, "output_size_mb": 6.0, "compression_ratio": 6.0},
    ]
    best, threshold_unmet = select_best_candidate(candidates, 95.0)
    assert best is not None
    assert best["crf"] == 24
    assert threshold_unmet is False


def test_select_best_candidate_fallback_to_highest_vmaf_when_target_unmet():
    candidates = [
        {"status": "ok", "crf": 30, "vmaf_mean": 90.1, "output_size_mb": 3.0, "compression_ratio": 3.0},
        {"status": "ok", "crf": 32, "vmaf_mean": 92.5, "output_size_mb": 4.0, "compression_ratio": 4.0},
        {"status": "ok", "crf": 34, "vmaf_mean": 89.0, "output_size_mb": 2.0, "compression_ratio": 2.0},
    ]
    best, threshold_unmet = select_best_candidate(candidates, 95.0)
    assert best is not None
    assert best["crf"] == 32
    assert threshold_unmet is True


def test_rank_candidates_assigns_rank():
    candidates = [
        {"status": "ok", "crf": 22, "vmaf_mean": 95.2, "output_size_mb": 9.0, "compression_ratio": 9.0},
        {"status": "ok", "crf": 24, "vmaf_mean": 95.2, "output_size_mb": 8.0, "compression_ratio": 8.0},
    ]
    ranked = rank_candidates(candidates, 95.0)
    assert ranked[0]["rank"] == 1
    assert ranked[0]["crf"] == 24
    assert ranked[1]["rank"] == 2


def test_build_fine_grid_hevc_with_boundaries():
    # hevc range is [20, 35], fine span is ±2
    grid = build_fine_grid("hevc", 20)
    assert grid == [20, 21, 22]


def test_build_fine_grid_av1_with_boundaries():
    # av1 range is [25, 50], fine span is ±3
    grid = build_fine_grid("av1", 49)
    assert grid == [46, 47, 48, 49, 50]
