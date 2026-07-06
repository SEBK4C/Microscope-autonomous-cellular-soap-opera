from soapscope.config import DramaConfig
from soapscope.drama.captioner import TemplateCaptioner
from soapscope.drama.characters import CharacterRegistry
from soapscope.vision.features import Beat, FrameFeatures


def _feat(frame_idx, beats):
    return FrameFeatures(frame_idx=frame_idx, beats=beats, per_track={})


def test_characters_are_stable_and_unique():
    reg = CharacterRegistry(seed=3)
    a1 = reg.get(1)
    a2 = reg.get(1)
    b = reg.get(2)
    assert a1 is a2                       # same track -> same character
    assert a1.name != b.name             # different tracks -> different names


def test_captioner_narrates_beats_and_evolves_rivalry():
    cap = TemplateCaptioner(DramaConfig(caption_every=1, seed=1))
    reg = CharacterRegistry(seed=1)
    headlines = []
    for i in range(6):
        ev = cap.update(i, _feat(i, [Beat("CHASE", [1, 2], score=5.0)]), reg)
        headlines.append(ev.headline)
    # Repeated chases between the same two build a rivalry tag over time.
    assert cap.rivalry[(1, 2)] >= 3
    joined = " ".join(" ".join(e.lines) for e in cap.transcript)
    assert "rivalry" in joined.lower() or "feud" in joined.lower() \
        or "confrontation" in joined.lower()


def test_captioner_variety():
    cap = TemplateCaptioner(DramaConfig(caption_every=1, seed=5))
    reg = CharacterRegistry(seed=5)
    kinds = ["ENTER", "CHASE", "ENCOUNTER", "FLEE", "DIVIDE", "WANDER"]
    outs = []
    for i, k in enumerate(kinds * 2):
        subs = [1, 2] if k in ("CHASE", "ENCOUNTER", "FLEE", "DIVIDE") else [1]
        ev = cap.update(i, _feat(i, [Beat(k, subs, score=5.0)]), reg)
        outs.append(ev.headline)
    assert len(set(outs)) >= len(kinds)   # meaningfully varied narration
