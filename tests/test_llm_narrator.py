"""LLM narrator tests — no transformers/torch needed (fallback + mock model)."""

import soapscope.drama.llm_backend as llm_backend
from soapscope.config import DramaConfig
from soapscope.drama.captioner import LLMCaptioner, make_captioner
from soapscope.drama.characters import CharacterRegistry
from soapscope.vision.features import Beat, FrameFeatures


def _feats(i, kind="CHASE", subs=(1, 2)):
    return FrameFeatures(frame_idx=i, beats=[Beat(kind, list(subs), 5.0)], per_track={})


def _raise(cfg):
    raise ImportError("no transformers here")


class _FakeLLM:
    def generate(self, system, user):
        return 'Caption: "A scandal unfolds on the slide!" #drama #microbes'


def test_make_captioner_llm_routes():
    assert isinstance(make_captioner(DramaConfig(backend="llm")), LLMCaptioner)


def test_llm_falls_back_to_template(monkeypatch):
    monkeypatch.setattr(llm_backend, "load_llm", _raise)
    cap = LLMCaptioner(DramaConfig(backend="llm", caption_every=1, seed=1))
    reg = CharacterRegistry(1)
    ev = cap.update(0, _feats(0), reg)
    assert ev.headline                      # produced *something*
    assert cap.used_llm == 0 and cap.used_fallback >= 1
    assert cap.transcript                   # recorded


def test_llm_uses_model_and_cleans_output(monkeypatch):
    monkeypatch.setattr(llm_backend, "load_llm", lambda cfg: _FakeLLM())
    cap = LLMCaptioner(DramaConfig(backend="llm", caption_every=1, seed=1))
    reg = CharacterRegistry(1)
    ev = cap.update(0, _feats(0, "ENCOUNTER"), reg)
    joined = " ".join(ev.lines)
    assert cap.used_llm == 1
    assert "A scandal unfolds on the slide!" in joined
    assert "#" not in joined and "Caption:" not in joined


def test_caption_cadence_holds_between_ticks(monkeypatch):
    monkeypatch.setattr(llm_backend, "load_llm", _raise)   # deterministic template path
    cap = LLMCaptioner(DramaConfig(backend="llm", caption_every=6, seed=1))
    reg = CharacterRegistry(1)
    ev0 = cap.update(0, _feats(0), reg)
    ev3 = cap.update(3, _feats(3), reg)
    assert ev3 is ev0                       # held between ticks
    ev6 = cap.update(6, _feats(6), reg)
    assert ev6.frame_idx == 6               # fresh caption on the tick


def test_clean_strips_prefix_hashtags_and_truncation():
    c = LLMCaptioner._clean
    assert c('Caption: "Hello there, darling!" #a #b') == "Hello there, darling!"
    assert c("The microbes waltz dramatically. Then they wander off into") \
        == "The microbes waltz dramatically."
    assert c("Just one line\nsecond line ignored") == "Just one line"
