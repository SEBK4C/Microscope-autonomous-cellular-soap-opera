"""VLM narrator tests — no transformers/torch needed (fallback + mock model)."""

import numpy as np

import soapscope.drama.vlm_backend as vlm_backend
from soapscope.config import DramaConfig
from soapscope.drama.captioner import VLMCaptioner, make_captioner
from soapscope.drama.characters import CharacterRegistry
from soapscope.vision.features import Beat, FrameFeatures
from soapscope.vision.track import Track


def _feats(i, kind="CHASE", subs=(1, 2)):
    return FrameFeatures(i, [Beat(kind, list(subs), 5.0)], {})


def _raise(cfg):
    raise ImportError("no vlm here")


class _FakeVLM:
    def describe(self, image):
        return "glowing green"


def test_make_captioner_vlm_routes():
    assert isinstance(make_captioner(DramaConfig(backend="vlm")), VLMCaptioner)


def test_vlm_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(vlm_backend, "load_vlm", _raise)
    cap = VLMCaptioner(DramaConfig(backend="vlm", caption_every=1, seed=1))
    frame = np.zeros((40, 40, 3), np.uint8)
    t = Track(id=1, cy=20, cx=20, bbox=(10, 10, 30, 30), hits=3)
    ev = cap.update(0, _feats(0), CharacterRegistry(1), frame=frame, tracks=[t])
    assert ev.headline and cap.used_vlm == 0 and cap.used_fallback >= 1


def test_vlm_grounds_caption_in_appearance(monkeypatch):
    monkeypatch.setattr(vlm_backend, "load_vlm", lambda cfg: _FakeVLM())
    reg = CharacterRegistry(1)
    cap = VLMCaptioner(DramaConfig(backend="vlm", caption_every=1, seed=1))
    frame = np.zeros((40, 40, 3), np.uint8)
    t = Track(id=1, cy=20, cx=20, bbox=(8, 8, 32, 32), hits=3)
    ev = cap.update(0, _feats(0, "CHASE"), reg, frame=frame, tracks=[t])
    joined = " ".join(ev.lines)
    assert cap.used_vlm == 1
    assert "glowing green" in joined          # grounded in the thumbnail's look
    assert reg.get(1).name in joined          # and attributed to the character


def test_vlm_style_format():
    cap = VLMCaptioner(DramaConfig(backend="vlm", seed=1))
    line = cap._style("glowing green", Beat("CHASE", [1, 2], 5.0), CharacterRegistry(1))
    assert "glowing green" in line and "gives chase" in line


def test_vlm_crop_bbox_then_whole_frame_fallback():
    cap = VLMCaptioner(DramaConfig(backend="vlm", seed=1))
    frame = np.zeros((40, 50, 3), np.uint8)
    beat = Beat("CHASE", [1, 2], 5.0)
    good = cap._crop_star(beat, frame, [Track(id=1, cy=20, cx=25, bbox=(10, 12, 30, 38), hits=3)])
    assert good is not None and good.size[0] > 0
    # A world-coord bbox off the sensor -> whole-frame fallback (PIL size is (W, H)).
    whole = cap._crop_star(beat, frame, [Track(id=1, cy=200, cx=300, bbox=(190, 290, 210, 310), hits=3)])
    assert whole.size == (50, 40)
