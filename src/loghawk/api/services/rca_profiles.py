"""Server-owned RCA investigation presets bounded by administrator ceilings."""

from loghawk import config

DEFAULT_MODE = config.LH_RCA_DEFAULT_MODE
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


def available_models() -> dict:
    """Return the explicitly configured RCA model choices and their default."""
    models = list(config.LH_LLM_TOOL_MODEL_LIST)
    default_model = models[0] if models else config.LH_LLM_TOOL_MODEL
    return {"models": models, "default_model": default_model}


def resolve_model(model: str | None = None) -> str:
    """Resolve a request model, restricting selection to the configured allowlist."""
    choices = available_models()
    if model is None:
        return choices["default_model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("RCA model must be a non-empty configured model name.")
    selected = model.strip()
    if choices["models"] and selected in choices["models"]:
        return selected
    if not choices["models"] and selected == config.LH_LLM_TOOL_MODEL:
        return selected
    raise ValueError("Selected RCA model is not configured in LH_LLM_TOOL_MODEL_LIST.")


def available_reasoning_models() -> dict:
    """Return the configured reasoning-model choices and their default."""
    models = list(config.LH_LLM_REASONING_MODEL_LIST)
    default_model = models[0] if models else config.LH_LLM_REASONING_MODEL
    return {"reasoning_models": models, "default_reasoning_model": default_model}


def resolve_reasoning_model(model: str | None = None) -> str:
    """Resolve a reasoning model, restricting selection to its configured allowlist."""
    choices = available_reasoning_models()
    if model is None:
        return choices["default_reasoning_model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Reasoning model must be a non-empty configured model name.")
    selected = model.strip()
    if choices["reasoning_models"] and selected in choices["reasoning_models"]:
        return selected
    if not choices["reasoning_models"] and selected == config.LH_LLM_REASONING_MODEL:
        return selected
    raise ValueError("Selected reasoning model is not configured in LH_LLM_REASONING_MODEL_LIST.")


def available_profiles() -> dict:
    """Expose effective limits and both model choices so the UI can show server settings."""
    profiles = []
    for mode, preset in _PRESETS.items():
        resolved = resolve_profile(mode)
        preset_limits = dict(zip(LIMIT_KEYS, preset["values"]))
        profiles.append({
            **resolved, "label": preset["label"], "description": preset["description"],
            "preset_limits": preset_limits,
            "limited_by_configuration": resolved["limits"] != preset_limits,
        })
    return {"default_mode": DEFAULT_MODE, "profiles": profiles, **available_models(), **available_reasoning_models()}
