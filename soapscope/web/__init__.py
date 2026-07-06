"""Live web viewer — watch an episode as it's produced (MJPEG stream + a live
caption ticker), served from the Python standard library (no web framework)."""

from .viewer import serve, dashboard_html, LiveShow

__all__ = ["serve", "dashboard_html", "LiveShow"]
