"""JSON-safe model evidence compaction; callers must redact before using this module."""

from copy import deepcopy
import json


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _preview(value, items, chars, depth=0):
    if depth > 16:
        return {"truncated": True, "notice": "Nested content omitted."}
    if isinstance(value, str) and len(value) > chars:
        return {"truncated": True, "original_chars": len(value), "text_preview": value[:chars]}
    if isinstance(value, list):
        selected = [_preview(item, items, chars, depth + 1) for item in value[:items]]
        if len(value) > items:
            return {"truncated": True, "total_items": len(value),
                    "included_items": len(selected), "items": selected}
        return selected
    if isinstance(value, dict):
        selected = {key: _preview(item, items, chars, depth + 1)
                    for key, item in list(value.items())[:items]}
        if len(value) > items:
            return {"truncated": True, "total_fields": len(value), "fields": selected}
        return selected
    return value


def bounded_json(value, max_chars):
    """Return complete JSON with explicit omissions, never a cut JSON fragment."""
    text = _dump(value)
    if len(text) <= max_chars:
        return text
    items, chars = 32, 512
    while items >= 1:
        preview = {"truncated": True, "notice": "Partial evidence; omitted content is not evidence of absence.",
                   "data": _preview(value, items, chars)}
        text = _dump(preview)
        if len(text) <= max_chars:
            return text
        if chars > 64:
            chars //= 2
        else:
            items //= 2
    text = _dump({"truncated": True, "notice": "Evidence omitted: result budget too small."})
    if len(text) > max_chars:
        raise ValueError("Result character budget is too small to describe evidence omissions.")
    return text


def compact_evidence(evidence, max_chars):
    """Factor repeated Prometheus labels; otherwise use a generic bounded preview."""
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except ValueError:
            # Older records may already contain a cut JSON document. Never imply completeness.
            evidence = {"truncated": True,
                        "notice": "Legacy or non-JSON evidence; completeness cannot be established.",
                        "text": evidence}
    packed = deepcopy(evidence)
    result = packed.get("result") if isinstance(packed, dict) else None
    data = result.get("data") if isinstance(result, dict) else None
    series = data.get("result") if isinstance(data, dict) else None
    if (isinstance(series, list) and series
            and data.get("resultType") in {"vector", "matrix"}
            and all(isinstance(item, dict) and isinstance(item.get("metric"), dict)
                    for item in series)):
        common = dict(series[0]["metric"])
        for item in series[1:]:
            common = {key: value for key, value in common.items()
                      if key in item["metric"] and item["metric"][key] == value}
        entries = [{**{key: value for key, value in item.items() if key != "metric"},
                    "labels": {key: value for key, value in item["metric"].items() if key not in common}}
                   for item in series]
        del data["result"]
        data.update(common_labels=common, series=entries, total_series=len(series),
                    included_series=len(entries), omitted_series=0, truncated=False)
        while len(_dump(packed)) > max_chars and entries:
            entries.pop()
            data.update(included_series=len(entries), omitted_series=len(series) - len(entries),
                        truncated=True)
        if data["truncated"]:
            data["notice"] = "Some series omitted for the model; consult saved collector evidence."
            while len(_dump(packed)) > max_chars and entries:
                entries.pop()
                data.update(included_series=len(entries), omitted_series=len(series) - len(entries))
    return bounded_json(packed, max_chars)
