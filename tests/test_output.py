"""Tests for mp4 export and the self-contained episode page."""

import numpy as np
import pytest

from soapscope.render.episode_page import build_episode_html, save_episode_html


def test_build_episode_html_has_media_transcript_and_stats():
    rows = [(0, "RECAP", "Previously, on As the Slide Turns: the feud deepened."),
            (6, "CHASE", "Count Vesper pursues Madame Dmitri across the slide.")]
    doc = build_episode_html("As the Slide Turns", "70 frames · score 0.86",
                             "data:video/mp4;base64,AAAA", True, rows,
                             [("recall", "93%"), ("MOTA", "0.74")])
    assert "As the Slide Turns" in doc
    assert "data:video/mp4;base64,AAAA" in doc and "<video" in doc
    assert "the feud deepened" in doc and "pursues Madame Dmitri" in doc
    assert "93%" in doc and "0.74" in doc
    assert "http://" not in doc and "https://" not in doc      # self-contained


def test_build_episode_html_escapes_and_gif_branch():
    doc = build_episode_html("t", "s", "data:image/gif;base64,AA", False,
                             [(0, "CHASE", "<script>evil()</script> & co")], [])
    assert "<img" in doc                                       # gif branch
    assert "&lt;script&gt;" in doc and "<script>evil" not in doc  # escaped, not injected


def test_save_episode_html_writes_self_contained_file(tmp_path):
    p = tmp_path / "episode.html"
    save_episode_html(p, b"\x00\x01\x02", "video/mp4", "Show", "sub",
                      [(0, "IDLE", "The pond is calm.")], [("fps", "16")])
    text = p.read_text(encoding="utf-8")
    assert "data:video/mp4;base64," in text and "The pond is calm." in text


def test_save_mp4_roundtrip_or_skip(tmp_path):
    iio = pytest.importorskip("imageio.v3")     # skip if imageio not installed (core CI)
    from soapscope.video.io import save_mp4
    frames = [np.full((32, 48, 3), i * 8, np.uint8) for i in range(10)]
    out = save_mp4(frames, tmp_path / "clip.mp4", fps=10)
    assert out.exists() and out.stat().st_size > 0
    back = iio.imread(out)
    assert back.shape[0] == 10                  # 10 frames read back
