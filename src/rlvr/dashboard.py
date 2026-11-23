"""Compatibility shim. The read layer lives in runread.py."""
from .runread import *  # noqa: F401,F403
from .runread import collect_dashboard, render_html, sample_gpu, serve_dashboard

__all__ = ["collect_dashboard", "render_html", "sample_gpu", "serve_dashboard"]
