from soapscope.config import PipelineConfig
from soapscope.drama.characters import CharacterRegistry
from soapscope.drama.season import SeasonMemory, Showrunner, _ordinal
from soapscope.pipeline import run_synthetic
from soapscope.vision.features import Beat, FrameFeatures


def _feats(beats):
    return FrameFeatures(0, beats, {})


def test_ordinal():
    got = [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)]
    assert got == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]


def test_season_persist_and_reload(tmp_path):
    p = str(tmp_path / "s.json")
    m = SeasonMemory(p)
    m.note_appearance("Contessa X", "the diva", 1)
    m.bump_relationship("Contessa X", "Baron Y", "feuds", 1)
    m.add_event(1, "X and Y feuded", 5.0)
    m.finish_episode()
    m2 = SeasonMemory(p)
    assert m2.episodes_done == 1
    assert m2.state["characters"]["Contessa X"]["feuds"]["Baron Y"] == 1


def test_recap_none_on_first_episode(tmp_path):
    assert Showrunner(SeasonMemory(str(tmp_path / "s.json"))).recap() is None


def test_recap_references_prior_events(tmp_path):
    p = str(tmp_path / "s.json")
    m = SeasonMemory(p)
    m.add_event(1, "Baron Y gave a dramatic speech", 9.0)
    m.finish_episode()
    recap = Showrunner(SeasonMemory(p)).recap()
    assert recap and recap.startswith("Previously") and "Baron Y gave a dramatic speech" in recap


def test_cliffhanger_mentions_hot_thread(tmp_path):
    reg = CharacterRegistry(1)
    sr = Showrunner(SeasonMemory(str(tmp_path / "s.json")), seed=1, observe_every=1)
    for i in range(4):
        sr.observe(_feats([Beat("CHASE", [1, 2], 5.0)]), reg, i)
    cliff = sr.cliffhanger()
    assert "NEXT TIME" in cliff
    assert reg.get(1).name in cliff or reg.get(2).name in cliff


def test_pipeline_recaps_second_episode(tmp_path):
    p = str(tmp_path / "s.json")

    def episode():
        c = PipelineConfig()
        c.n_frames = 30
        c.drama.season_path = p
        c.drama.seed = 7
        return run_synthetic(c, collect_frames=False)

    r1 = episode()
    assert any(e.kind == "CLIFFHANGER" for e in r1.transcript)
    assert not any(e.kind == "RECAP" for e in r1.transcript)   # first episode: no recap
    r2 = episode()
    assert any(e.kind == "RECAP" for e in r2.transcript)       # second episode recaps ep 1
