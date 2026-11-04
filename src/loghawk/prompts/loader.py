"""Load application-selected UTF-8 prompt resources; cached until restart."""
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from pathlib import PurePosixPath

@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Prompt path must be relative to loghawk.prompts")
    text = files("loghawk.prompts").joinpath(*path.parts).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty required prompt: {name}")
    return text

def prompt_snapshot(name: str) -> dict:
    text = load_prompt(name)
    return {"name": name, "sha256": sha256(text.encode("utf-8")).hexdigest(), "text": text}
