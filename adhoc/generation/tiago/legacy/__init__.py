"""Legacy tiago helpers and experiments.

Canonical TIAGo mobile renderer (shots, generation configs, pilot GIFs):
``render_config`` / ``render_config_with_trajectory`` from ``render_mobile_config``.
"""
from .render_mobile_config import _make_env, render_config, render_config_with_trajectory

__all__ = ["render_config", "render_config_with_trajectory", "_make_env"]
