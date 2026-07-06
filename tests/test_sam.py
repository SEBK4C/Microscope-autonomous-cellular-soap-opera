"""SAM integration tests — the model-agnostic glue, no torch/ultralytics needed."""

import numpy as np
import pytest

from soapscope.config import SegmentConfig
from soapscope.vision.segment import (
    SamSegmenter, make_segmenter, masks_to_segresult,
)


def _mask(H, W, y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_masks_to_segresult_filters_by_area():
    H, W = 40, 40
    big = _mask(H, W, 5, 15, 5, 15)      # area 100
    tiny = _mask(H, W, 30, 33, 30, 33)   # area 9
    res = masks_to_segresult([big, tiny], (H, W), min_area=20, max_area=10000)
    assert len(res.detections) == 1
    assert res.detections[0].area == 100
    assert res.labels.max() == 1
    cy, cx = res.detections[0].centroid
    assert 9.0 <= cy <= 10.0 and 9.0 <= cx <= 10.0


def test_masks_to_segresult_small_painted_on_top():
    H, W = 30, 30
    big = _mask(H, W, 5, 25, 5, 25)
    small = _mask(H, W, 10, 14, 10, 14)
    res = masks_to_segresult([big, small], (H, W), min_area=1, max_area=10000)
    assert len(res.detections) == 2
    # Smaller mask is painted last, so its interior pixel keeps its own label,
    # not the big mask's — the two detections are distinguishable in the map.
    assert res.labels[12, 12] != res.labels[6, 6]


def test_masks_to_segresult_empty():
    res = masks_to_segresult([], (16, 16), min_area=1, max_area=100)
    assert res.detections == []
    assert res.labels.shape == (16, 16)
    assert res.labels.max() == 0


def test_make_segmenter_routes_sam_variants():
    for b in ("sam", "sam2", "sam3"):
        seg = make_segmenter(SegmentConfig(backend=b))
        assert isinstance(seg, SamSegmenter)
    # Lazy: constructing a SamSegmenter must NOT import torch/ultralytics.
    assert make_segmenter(SegmentConfig(backend="sam3"))._backend is None


def test_cli_demo_and_run_wire_sam_backend():
    # The brief's marquee — "SAM follows the microbes" — must be reachable from
    # BOTH the demo (incl. moving stage) and run commands, without importing torch.
    from soapscope.cli import build_parser, _apply_sam_cfg
    from soapscope.config import PipelineConfig
    parser = build_parser()
    for argv in (["demo", "--moving", "--backend", "sam", "--sam-imgsz", "384"],
                 ["run", "--input", "x.mp4", "--backend", "sam",
                  "--sam-backend", "fastsam"]):
        args = parser.parse_args(argv)
        cfg = PipelineConfig()
        _apply_sam_cfg(cfg, args)
        assert cfg.segment.backend == "sam3"        # routes to SamSegmenter
    # imgsz flows through on the demo path
    args = parser.parse_args(["demo", "--backend", "sam", "--sam-imgsz", "384"])
    cfg = PipelineConfig()
    _apply_sam_cfg(cfg, args)
    assert cfg.segment.sam_imgsz == 384 and cfg.segment.sam_backend == "auto"
    # classical stays classical (no-op)
    args = parser.parse_args(["demo"])
    cfg = PipelineConfig()
    _apply_sam_cfg(cfg, args)
    assert cfg.segment.backend == "classical"


def test_sam_weight_resolver_size_guard(tmp_path, monkeypatch):
    # A GitHub-403 error saved as a ".pt" is a tiny file; the resolver must not
    # treat it as a real checkpoint (that was the iter-17 bug).
    import soapscope.vision.sam_backend as sb
    monkeypatch.setattr(sb, "_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sb, "_fetch_from_hf", lambda name: None)   # no network
    (tmp_path / "FastSAM-s.pt").write_text('{"message":"403"}')    # ~17-byte stub
    # The stub in models/ is rejected by the size guard -> falls through to the
    # bare name, NOT the stub path.
    assert sb.resolve_weight("FastSAM-s.pt") == "FastSAM-s.pt"
    # A genuine large local file IS accepted as-is.
    big = tmp_path / "real.pt"
    big.write_bytes(b"\0" * 200_000)
    assert sb.resolve_weight(str(big)) == str(big)


def test_sam_missing_backend_raises_actionable_error():
    # segment_anything is never a core dependency -> deterministic failure path.
    seg = SamSegmenter(SegmentConfig(backend="sam3", sam_backend="segment_anything",
                                     sam_model="does-not-exist.pth"))
    with pytest.raises(ImportError) as ei:
        seg.segment(np.zeros((16, 16, 3), np.uint8))
    msg = str(ei.value)
    assert "SAM backend" in msg and "pip install" in msg
