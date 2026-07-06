"""Live web viewer for "As the Slide Turns".

Runs the pipeline in a background thread over an endless synthetic feed, encodes
each annotated frame to JPEG, and serves:

- ``/``            the broadcast dashboard (below)
- ``/stream.mjpg`` a multipart MJPEG stream (the live video, near-real-time)
- ``/state``       JSON of the current caption / cast / episode (a caption ticker)

Pure standard-library HTTP (``http.server``) so there's no web-framework
dependency. Swap the synthetic feed for a webcam source and it streams real
microscopy the same way.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from ..config import PipelineConfig


class LiveShow:
    """Thread-safe latest-frame + latest-state, shared producer → HTTP handlers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = b""
        self._state = {"caption": "Standby…", "kind": "", "cast": 0,
                       "episode": 1, "frame": 0, "star": None}
        self.running = True

    def update(self, jpeg: bytes, state: dict) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._state = state

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)


def _producer(show: LiveShow, cfg: PipelineConfig) -> None:
    """Endlessly stream episodes into ``show`` (a fresh world/cast each episode)."""
    from ..pipeline import Pipeline
    from ..video.synthetic import SyntheticWorld

    episode = 1
    delay = 1.0 / max(1, cfg.render.fps)
    while show.running:
        try:
            world = SyntheticWorld(cfg.world)
            pipe = Pipeline(cfg)
            for i, (annotated, caption, tracks, step) in enumerate(
                    pipe.stream(world.frames(cfg.n_frames))):
                if not show.running:
                    return
                buf = io.BytesIO()
                Image.fromarray(np.asarray(annotated, np.uint8)).save(
                    buf, format="JPEG", quality=82)
                show.update(buf.getvalue(), {
                    "caption": caption.headline if caption else "…",
                    "kind": caption.kind if caption else "",
                    "cast": len(tracks), "episode": episode, "frame": i,
                    "star": step.star_id if step else None,
                })
                time.sleep(delay)
            episode += 1
        except Exception:  # noqa: BLE001 - never let the producer thread die
            time.sleep(0.5)


def dashboard_html() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>As the Slide Turns — LIVE</title>
<style>
:root{--bg:#0b0e12;--panel:#141a21;--edge:#26303b;--ink:#e9eef3;--muted:#8b98a5;
 --gold:#ffd666;--gold-dim:#b9964a;--rose:#ef476f;--teal:#2ec4b6;
 --serif:Georgia,"Times New Roman",serif;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}
.wrap{max-width:760px;margin:0 auto;padding:34px 18px 56px}
header{text-align:center;padding:20px 0 22px;
 background:radial-gradient(120% 150% at 50% -45%,#17222e,transparent 68%)}
.live{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11px;
 letter-spacing:.24em;color:var(--rose);text-transform:uppercase}
.live .dot{width:9px;height:9px;border-radius:50%;background:var(--rose);
 box-shadow:0 0 10px var(--rose);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.live .dot{animation:none}}
h1{font-family:var(--serif);font-weight:600;letter-spacing:.14em;text-transform:uppercase;
 font-size:clamp(26px,6vw,42px);margin:.3em 0 .05em;color:var(--gold);text-shadow:0 1px 0 #000}
.screen{margin:12px 0 0;border:1px solid var(--edge);border-radius:14px;background:#000;padding:9px;
 box-shadow:0 24px 60px rgba(0,0,0,.55)}
.screen img{display:block;width:100%;border-radius:7px}
.caption{margin-top:10px;border-left:3px solid var(--gold);background:linear-gradient(90deg,rgba(255,214,102,.08),transparent);
 padding:12px 14px;border-radius:0 8px 8px 0;min-height:56px;font-size:17px;font-family:var(--serif)}
.meta{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
.chip{flex:1 1 90px;background:var(--panel);border:1px solid var(--edge);border-radius:10px;padding:9px 12px}
.chip .k{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.chip .v{font-size:19px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
footer{margin-top:26px;text-align:center;color:var(--muted);font-size:12.5px}
</style></head>
<body><div class="wrap">
<header>
 <div class="live"><span class="dot"></span> On air · live</div>
 <h1>As the Slide Turns</h1>
</header>
<div class="screen"><img src="/stream.mjpg" alt="live microscope feed"></div>
<div class="caption" id="cap">Standby…</div>
<div class="meta">
 <div class="chip"><div class="k">Episode</div><div class="v" id="ep">1</div></div>
 <div class="chip"><div class="k">Cast on slide</div><div class="v" id="cast">0</div></div>
 <div class="chip"><div class="k">Now</div><div class="v" id="kind" style="font-size:14px">—</div></div>
</div>
<footer>SoapScope — the autonomous cellular soap opera · streaming from the standard library</footer>
</div>
<script>
async function tick(){
 try{
  const s = await (await fetch('/state',{cache:'no-store'})).json();
  document.getElementById('cap').textContent = s.caption || '…';
  document.getElementById('ep').textContent = s.episode;
  document.getElementById('cast').textContent = s.cast;
  document.getElementById('kind').textContent = (s.kind||'—').toLowerCase();
 }catch(e){}
}
setInterval(tick, 450); tick();
</script>
</body></html>"""


def _make_handler(show: LiveShow, fps: int):
    html = dashboard_html().encode("utf-8")
    delay = 1.0 / max(1, fps)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):      # keep the console quiet
            pass

        def _send(self, code, ctype, body, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            elif path == "/state":
                self._send(200, "application/json",
                           json.dumps(show.state()).encode("utf-8"),
                           {"Cache-Control": "no-store"})
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while show.running:
                        jpg = show.jpeg()
                        if jpg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n"
                                             .encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def serve(cfg: PipelineConfig, host: str = "127.0.0.1", port: int = 8000):
    """Start the producer + HTTP server. Blocks until interrupted."""
    show = LiveShow()
    threading.Thread(target=_producer, args=(show, cfg), daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), _make_handler(show, cfg.render.fps))
    print(f"[serve] As the Slide Turns — LIVE at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        show.running = False
        httpd.shutdown()
    return httpd
