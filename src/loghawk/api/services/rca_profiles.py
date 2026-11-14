"""Server-owned RCA investigation presets bounded by administrator ceilings."""

from loghawk import config

DEFAULT_MODE = "performance"
LIMIT_KEYS = (
    "MAX_MODEL_TURNS", "MAX_DISCOVERY_CALLS", "MAX_EXECUTION_REQUESTS", "MAX_RAG_SEARCHES",
)
_PRESETS = {
    "quality": {
        "label": "Quality",
        "description": "Explore more evidence and alternative explanations.",
        "values": (15, 10, 10, 5),
    },
    "balanced": {
        "label": "Balanced",
        "description": "Moderate investigation depth and resource usage.",
        "values": (8, 4, 4, 2),
    },
    "performance": {
        "label": "Performance",
        "description": "Focus on supplied evidence and a few additional checks.",
        "values": (5, 2, 2, 1),
    },
}


def resolve_profile(mode: str | None = None) -> dict:
    """Resolve a named web preset, or preserve configured limits for older clients."""
    ceilings = {key: getattr(config, "LH_LLM_TOOL_" + key) for key in LIMIT_KEYS}
    if mode is None:
        return {"mode": "configured", "limits": ceilings}
    if not isinstance(mode, str) or mode not in _PRESETS:
        raise ValueError("RCA mode must be quality, balanced or performance")
    preset = dict(zip(LIMIT_KEYS, _PRESETS[mode]["values"]))
    return {"mode": mode, "limits": {key: min(preset[key], ceilings[key]) for key in LIMIT_KEYS}}


def available_profiles() -> dict:
    """Expose effective limits so the UI can show administrator reductions."""
    profiles = []
    for mode, preset in _PRESETS.items():
        resolved = resolve_profile(mode)
        preset_limits = dict(zip(LIMIT_KEYS, preset["values"]))
        profiles.append({
            **resolved, "label": preset["label"], "description": preset["description"],
            "preset_limits": preset_limits,
            "limited_by_configuration": resolved["limits"] != preset_limits,
        })
    return {"default_mode": DEFAULT_MODE, "profiles": profiles}
