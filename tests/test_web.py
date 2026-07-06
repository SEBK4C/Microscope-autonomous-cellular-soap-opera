"""Live web viewer tests — dashboard HTML, shared state, and server routing."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from soapscope.web.viewer import LiveShow, _make_handler, dashboard_html


def test_dashboard_html_has_stream_and_ticker():
    doc = dashboard_html()
    assert "As the Slide Turns" in doc
    assert "/stream.mjpg" in doc          # the live video
    assert "/state" in doc                # the caption ticker poll
    assert "http://" not in doc and "https://" not in doc   # self-contained


def test_liveshow_is_thread_safe_container():
    s = LiveShow()
    s.update(b"JPGBYTES", {"caption": "hi", "kind": "CHASE", "cast": 3,
                           "episode": 2, "frame": 5, "star": 1})
    assert s.jpeg() == b"JPGBYTES"
    st = s.state()
    assert st["caption"] == "hi" and st["cast"] == 3
    st["caption"] = "mutated"             # returned dict is a copy
    assert s.state()["caption"] == "hi"


def test_server_serves_state_and_dashboard():
    show = LiveShow()
    show.update(b"\xff\xd8jpg", {"caption": "Live test line", "kind": "IDLE",
                                 "cast": 4, "episode": 1, "frame": 0, "star": None})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(show, fps=12))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        st = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/state", timeout=3).read())
        assert st["caption"] == "Live test line" and st["cast"] == 4
        home = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/", timeout=3).read().decode()
        assert "As the Slide Turns" in home and "/stream.mjpg" in home
    finally:
        show.running = False
        httpd.shutdown()
