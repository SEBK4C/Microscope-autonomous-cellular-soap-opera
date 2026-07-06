from soapscope.config import PipelineConfig, StageConfig, WorldConfig
from soapscope.pipeline import Pipeline
from soapscope.stage.moving import MovingStageController
from soapscope.video.synthetic import SyntheticWorld
from soapscope.vision.features import Beat, FrameFeatures
from soapscope.vision.track import Track


def test_world_scale_enlarges_world_keeps_sensor():
    w1 = SyntheticWorld(WorldConfig(world_scale=1.0, height=100, width=120, seed=1))
    assert (w1.H, w1.W) == (100, 120)                 # scale 1 => unchanged
    w2 = SyntheticWorld(WorldConfig(world_scale=2.0, height=100, width=120, seed=1))
    assert (w2.H, w2.W) == (200, 240)                 # world doubled
    assert (w2.sensor_h, w2.sensor_w) == (100, 120)   # sensor unchanged
    frame, _gt = next(w2.frames(1))
    assert frame.shape[:2] == (200, 240)              # frames are slide-sized


def test_moving_stage_pans_toward_star_and_stays_in_bounds():
    m = MovingStageController(StageConfig(deadzone=5, max_step=20),
                              world_h=400, world_w=400, sensor_h=100, sensor_w=100)
    star = Track(id=1, cy=350.0, cx=350.0, hits=5)
    feats = FrameFeatures(0, [Beat("WANDER", [1], 1.0)], {})
    y_start, x_start = m.cy, m.cx
    for _ in range(25):
        m.step([star], feats)
    assert m.cy > y_start and m.cx > x_start          # panned toward the star
    y0, x0 = m.crop_origin()
    assert 0 <= y0 <= 300 and 0 <= x0 <= 300          # sensor stays inside the slide


def test_run_moving_end_to_end():
    cfg = PipelineConfig()
    cfg.n_frames = 20
    cfg.world.world_scale = 1.6
    cfg.world.n_start = 10
    cfg.world.max_microbes = 16
    cfg.world.seed = 2
    world = SyntheticWorld(cfg.world)
    res = Pipeline(cfg).run_moving(world.frames(cfg.n_frames), collect_frames=True)
    assert len(res.frames) == 20
    assert res.frames[0].shape[:2] == (cfg.world.height, cfg.world.width)   # sensor-sized
    assert res.moving is not None and res.moving["stage_travel"] >= 0.0
    assert res.records[0].stage_cmd["cmd"] == "move_abs"
    assert res.transcript
