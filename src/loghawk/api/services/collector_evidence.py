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


def evidence_metadata(original: str, prepared: str) -> dict:
    """Describe observed omissions without claiming source completeness."""
    def parse(text):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None
    source, output = parse(original), parse(prepared)
    markers = []
    def walk(value, path="$", depth=0):
        if depth > 20:
            return
        if isinstance(value, dict):
            keys = ("truncated", "truncated_by_script", "total_series", "included_series", "omitted_series",
                    "total_items", "included_items", "total_fields", "original_chars", "source_limit", "requested_limit")
            found = {key: value[key] for key in keys if key in value}
            if found:
                markers.append({"path": path, **found})
            for key, item in value.items():
                walk(item, path + "." + key, depth + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]", depth + 1)
    walk(output)
    result = source.get("result", {}) if isinstance(source, dict) else {}
    source_markers = []
    if isinstance(result, dict):
        data = result.get("data", {})
        for container in (result, data if isinstance(data, dict) else {}):
            for key in ("truncated", "truncated_by_script", "source_limit", "requested_limit", "limit", "returned_series"):
                if key in container:
                    source_markers.append({key: container[key]})
    omitted = any(m.get("truncated") is True or
                  (isinstance(m.get("omitted_series"), (int, float)) and m["omitted_series"] > 0)
                  for m in markers)
    return {"redacted": True, "changed": source != output, "model_omissions": omitted,
            "reason": "Bounded preview or record omission" if omitted else
                      ("Representation compacted without reported omissions" if source != output else "Unchanged evidence"),
            "model_markers": markers, "source_reported_limits": source_markers or None,
            "source_completeness": "Not established; limits apply before saved evidence",
            "runner_output": result.get("output_metadata") if isinstance(result, dict) else None,
            "character_counts": "Unicode characters after redaction, not tokens or bytes"}
