# app/config.py
"""
Central config loader. All modules import CFG instead of hardcoding values.

Usage:
    from app.config import CFG
    top_k    = CFG["retrieval"]["top_k"]
    model    = CFG["models"]["embedding"]

Or with convenience accessors:
    from app.config import retrieval, generation, agent
    bm25_div = retrieval.bm25_div
"""

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def _load() -> dict:
    with _CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)

CFG: dict = _load()


class _Section:
    """Dot-access wrapper for a config section."""
    def __init__(self, d: dict):
        self._d = d
    def __getattr__(self, key):
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(f"Config key '{key}' not found in section")
    def __getitem__(self, key):
        return self._d[key]

models     = _Section(CFG["models"])
retrieval  = _Section(CFG["retrieval"])
chunking   = _Section(CFG["chunking"])
agent_cfg  = _Section(CFG["agent"])
generation = _Section(CFG["generation"])
evaluation = _Section(CFG["evaluation"])
paths      = _Section(CFG["paths"])