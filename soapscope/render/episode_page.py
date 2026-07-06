"""A self-contained, shareable episode page.

Bundles the rendered episode (mp4 or gif, embedded as a data URI) and the full
narration transcript into ONE HTML file with no external assets — so an episode
can be watched and read in any browser, offline, forever.

Deliberately single-theme: styled as a dark "on-air" broadcast (dark-field
microscope meets a daytime-TV title card), which is the show's actual world.
Uses system fonts only, so the file stays self-contained.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_CSS = """
:root{
  --bg:#0b0e12; --panel:#141a21; --panel2:#1b232c; --edge:#26303b;
  --ink:#e9eef3; --muted:#8b98a5; --gold:#ffd666; --gold-dim:#b9964a;
  --rose:#ef476f; --teal:#2ec4b6;
  --serif:Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);
   color:var(--ink);font-family:var(--sans);line-height:1.55;
   -webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:40px 20px 64px}
header{text-align:center;padding:26px 0 26px;border-bottom:1px solid var(--edge);
  background:radial-gradient(120% 150% at 50% -45%, #17222e 0%, transparent 68%)}
.onair{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);
  font-size:11px;letter-spacing:.22em;color:var(--rose);text-transform:uppercase}
.onair::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--rose);
  box-shadow:0 0 10px var(--rose)}
h1{font-family:var(--serif);font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  font-size:clamp(28px,6vw,46px);margin:.35em 0 .1em;color:var(--gold);
  text-wrap:balance;text-shadow:0 1px 0 #000}
.sub{color:var(--muted);font-size:14px;font-family:var(--mono);letter-spacing:.03em}
.screen{margin:26px 0 8px;border:1px solid var(--edge);border-radius:14px;
  background:#000;padding:10px;box-shadow:0 24px 60px rgba(0,0,0,.55),
  inset 0 0 0 1px rgba(255,255,255,.03)}
.screen video,.screen img{display:block;width:100%;border-radius:7px}
.scan{position:relative}
.scan::after{content:"";position:absolute;inset:10px;border-radius:7px;pointer-events:none;
  background:repeating-linear-gradient(rgba(255,255,255,.035) 0 1px, transparent 1px 3px);
  mix-blend-mode:overlay}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 6px}
.stat{flex:1 1 96px;background:var(--panel);border:1px solid var(--edge);border-radius:10px;
  padding:11px 13px}
.stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted)}
.stat .v{font-size:20px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
h2{font-family:var(--serif);font-weight:600;letter-spacing:.05em;font-size:19px;
  margin:34px 0 12px;color:var(--ink)}
h2 .n{color:var(--muted);font-family:var(--mono);font-size:13px;letter-spacing:0}
.script{border-top:1px solid var(--edge)}
.line{display:grid;grid-template-columns:52px auto;gap:14px;padding:11px 4px;
  border-bottom:1px solid var(--edge);align-items:baseline}
.t{font-family:var(--mono);font-size:12px;color:var(--gold-dim);
  font-variant-numeric:tabular-nums}
.body .chip{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;padding:2px 7px;border-radius:999px;margin-right:9px;
  border:1px solid var(--edge);color:var(--muted);vertical-align:1px}
.chip.teal{color:var(--teal);border-color:#215a55}
.chip.rose{color:var(--rose);border-color:#5f2038}
.chip.gold{color:var(--gold);border-color:#5b4a1e}
.line.card{background:linear-gradient(90deg,rgba(255,214,102,.07),transparent);
  border-left:2px solid var(--gold);padding-left:10px;margin-left:-12px}
.line.card .txt{font-family:var(--serif);font-size:16px}
.txt{color:var(--ink)}
footer{margin-top:40px;text-align:center;color:var(--muted);font-size:12.5px;
  border-top:1px solid var(--edge);padding-top:20px}
footer a{color:var(--gold-dim)}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
"""

_CARD_KINDS = {"RECAP", "CLIFFHANGER"}
_TEAL_KINDS = {"CHASE", "FLEE", "ENCOUNTER", "SPEED_BURST"}
_ROSE_KINDS = {"DIVIDE", "EXIT"}


def _chip_class(kind: str) -> str:
    if kind in _CARD_KINDS:
        return "gold"
    if kind in _TEAL_KINDS:
        return "teal"
    if kind in _ROSE_KINDS:
        return "rose"
    return ""


def build_episode_html(title: str, subtitle: str, media_data_uri: str,
                       is_video: bool, transcript: Sequence[Tuple[int, str, str]],
                       stats: Sequence[Tuple[str, str]]) -> str:
    if is_video:
        media = (f'<video src="{media_data_uri}" controls autoplay loop muted '
                 f'playsinline></video>')
    else:
        media = f'<img src="{media_data_uri}" alt="episode">'

    stat_html = "".join(
        f'<div class="stat"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>' for k, v in stats)

    rows = []
    for frame, kind, text in transcript:
        card = " card" if kind in _CARD_KINDS else ""
        chip = (f'<span class="chip {_chip_class(kind)}">{html.escape(kind)}</span>'
                if kind not in _CARD_KINDS else "")
        rows.append(
            f'<div class="line{card}"><div class="t">{frame:04d}</div>'
            f'<div class="body">{chip}<span class="txt">{html.escape(text)}</span></div></div>')
    script = "".join(rows)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <div class="onair">On air</div>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)}</div>
</header>
<div class="screen scan">{media}</div>
<div class="stats">{stat_html}</div>
<h2>Episode transcript <span class="n">({len(transcript)} beats)</span></h2>
<div class="script">{script}</div>
<footer>Generated by <b>SoapScope</b> — the autonomous cellular soap opera ·
a Karpathy-style autoresearch loop</footer>
</div></body></html>"""


def save_episode_html(path: str | Path, media_bytes: bytes, media_mime: str,
                      title: str, subtitle: str,
                      transcript: Sequence[Tuple[int, str, str]],
                      stats: Sequence[Tuple[str, str]]) -> Path:
    uri = f"data:{media_mime};base64," + base64.b64encode(media_bytes).decode("ascii")
    doc = build_episode_html(title, subtitle, uri, media_mime.startswith("video"),
                             transcript, stats)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
