"""JSON-safe model evidence views; callers must redact before using this module.

Saved results are never mutated. Character limits apply to the final serialized
view, including evidence aliases. Sampling describes omissions, not completeness.
"""

from copy import deepcopy
import json
import math


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _omitted(value, max_chars, reason="Evidence omitted: result budget too small."):
    result = {"truncated": True, "evidence_omitted": True, "notice": reason}
    if isinstance(value, dict) and value.get("evidence_ref") is not None:
        result = {"evidence_ref": value["evidence_ref"], **result}
    if isinstance(value, dict):
        data = value.get("result", {}).get("data", {}) if isinstance(value.get("result"), dict) else {}
        if isinstance(data, dict) and isinstance(data.get("total_series"), int):
            result.update(total_series=data["total_series"], included_series=0, omitted_series=data["total_series"])
        if isinstance(value.get("chunks"), list):
            result["omitted_chunks"] = len(value["chunks"]) + value.get("omitted_chunks", 0)
    text = _dump(result)
    if len(text) <= max_chars:
        return text
    # Even a small allowance can carry an honest omission and a short alias.
    minimal = {"evidence_omitted": True, "reason": "budget"}
    if isinstance(value, dict) and value.get("evidence_ref") is not None:
        minimal["evidence_ref"] = value["evidence_ref"]
    text = _dump(minimal)
    if len(text) > max_chars:
        raise ValueError("Result character budget is too small to retain the evidence reference and omission notice.")
    return text


def _preview(value, items, chars, depth=0):
    if depth > 16:
        return {"truncated": True, "notice": "Nested content omitted."}
    if isinstance(value, str) and len(value) > chars:
        return {"truncated": True, "original_chars": len(value),
                "omitted_chars": len(value) - chars, "text_preview": value[:chars]}
    if isinstance(value, list):
        selected = [_preview(item, items, chars, depth + 1) for item in value[:items]]
        if len(value) > items:
            return {"truncated": True, "total_items": len(value),
                    "included_items": len(selected), "omitted_items": len(value) - len(selected), "items": selected}
        return selected
    if isinstance(value, dict):
        # An alias is provenance, not arbitrary text to shorten or discard.
        keys = sorted(value, key=lambda key: key != "evidence_ref")
        selected = {key: value[key] if key == "evidence_ref" else
                    _preview(value[key], items, chars, depth + 1) for key in keys[:items]}
        if len(value) > items:
            return {"truncated": True, "total_fields": len(value),
                    "omitted_fields": len(value) - len(selected), "fields": selected}
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
        if isinstance(value, dict) and value.get("evidence_ref") is not None:
            preview["evidence_ref"] = value["evidence_ref"]
        text = _dump(preview)
        if len(text) <= max_chars:
            return text
        if chars > 64:
            chars //= 2
        else:
            items //= 2
    return _omitted(value, max_chars)


def _deduplicate_stdout(result):
    """Omit JSON stdout only when its complete contents are represented elsewhere."""
    if not isinstance(result, dict) or "data" not in result:
        return
    stdout = result.get("stdout")
    if not isinstance(stdout, str):
        return
    metadata = result.get("output_metadata")
    if isinstance(metadata, dict):
        stdout_metadata = metadata.get("stdout")
        if isinstance(stdout_metadata, dict) and stdout_metadata.get("truncated"):
            return
    try:
        payload = json.loads(stdout)
        def same(left, right):
            return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
        if isinstance(payload, dict) and "data" in payload:
            duplicate = (
                set(payload).issubset({"summary", "data"})
                and same(payload["data"], result["data"])
                and ("summary" not in payload or same(payload["summary"], result.get("summary")))
            )
        else:
            duplicate = same(payload, result["data"])
    except (ValueError, TypeError, RecursionError):
        return
    if duplicate:
        del result["stdout"]
        result["stdout_deduplicated"] = True


def _aliases(value, aliases):
    if isinstance(value, list):
        return [_aliases(item, aliases) for item in value]
    if isinstance(value, dict):
        return {key: aliases.get(item, item) if key == "evidence_ref" and isinstance(item, str)
                else _aliases(item, aliases) for key, item in value.items()}
    return value


def _normalize(evidence):
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except ValueError:
            evidence = {"truncated": True,
                        "notice": "Legacy or non-JSON evidence; completeness cannot be established.", "text": evidence}
    packed = deepcopy(evidence)
    # Older investigation results wrapped a collector envelope in another result.
    # Restrict unwrapping to that known envelope; arbitrary business data is left intact.
    while (isinstance(packed, dict) and set(packed).issubset({"evidence_ref", "result"})
           and isinstance(packed.get("result"), dict)
           and isinstance(packed["result"].get("result"), dict)
           and "collector" in packed["result"]):
        outer_ref = packed.get("evidence_ref")
        packed = packed["result"]
        if outer_ref is not None:
            packed["evidence_ref"] = outer_ref
    result = packed.get("result") if isinstance(packed, dict) else None
    _deduplicate_stdout(result)
    if isinstance(result, dict):
        omitted = []
        for key in ("task_id", "artifact_id"):
            if key in result:
                del result[key]; omitted.append(key)
        for key in ("run_id", "collector", "version"):
            if key in packed and key in result and str(packed[key]) == str(result[key]):
                del result[key]; omitted.append(key)
        if result.get("stderr") == "":
            del result["stderr"]; omitted.append("stderr")
        if result.get("error_type", "missing") is None:
            del result["error_type"]; omitted.append("error_type")
        # Per-stream counts are retained in the saved result and audit metadata.
        # Keep truncation/limits visible when the runner did omit output.
        metadata = result.pop("output_metadata", None)
        if isinstance(metadata, dict):
            limited = {key: value for key, value in metadata.items()
                       if isinstance(value, dict) and value.get("truncated")}
            if limited:
                result["runner_output_limits"] = limited
            omitted.append("output_metadata")
        elif metadata is not None:
            result["output_metadata"] = metadata
        if omitted:
            result["metadata_fields_omitted"] = omitted
    return packed


def deduplicate_knowledge_chunks(payload):
    """Deduplicate exact excerpts while retaining every source/alias association.

    Different full-content hashes are not merged even when their previews match.
    This helper accepts a combined chunks list for deduplication across searches.
    """
    result = deepcopy(payload)
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not isinstance(chunks, list):
        return result
    kept, seen, duplicates = [], {}, 0
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            kept.append(chunk); continue
        key = (chunk.get("content_sha256"), chunk["text"])
        if key not in seen:
            seen[key] = chunk; kept.append(chunk); continue
        duplicate = {key: chunk[key] for key in ("source", "evidence_ref") if key in chunk}
        provenance = seen[key].setdefault("duplicate_sources", [])
        if duplicate and duplicate not in provenance:
            provenance.append(duplicate)
        for source in chunk.get("duplicate_sources", []):
            if source not in provenance:
                provenance.append(source)
        duplicates += 1
    result["chunks"] = kept
    result["deduplicated_chunks"] = result.get("deduplicated_chunks", 0) + duplicates
    return result


def _knowledge_view(packed, max_chars):
    packed = deduplicate_knowledge_chunks(packed)
    chunks = packed["chunks"]
    original_count = len(chunks)
    previous_omitted = packed.get("omitted_chunks", 0)
    packed["total_chunks"] = original_count + previous_omitted
    packed["included_chunks"] = original_count
    packed["omitted_chunks"] = previous_omitted
    while len(_dump(packed)) > max_chars and len(chunks) > 1:
        chunks.pop()
        packed["included_chunks"] = len(chunks)
        packed["omitted_chunks"] = previous_omitted + original_count - len(chunks)
    if len(_dump(packed)) > max_chars and chunks and isinstance(chunks[0], dict):
        chunk = chunks[0]
        text = chunk.get("text")
        if isinstance(text, str):
            original_chars = max(chunk.get("original_chars", len(text)), len(text))
            for length in (1024, 512, 256, 128):
                if length >= len(text):
                    continue
                chunk["text"] = text[:length]
                chunk["original_chars"] = original_chars
                chunk["omitted_chars"] = original_chars - length
                chunk["text_truncated"] = True
                if len(_dump(packed)) <= max_chars:
                    break
    if packed.get("omitted_chunks"):
        packed["truncated"] = True
    text = _dump(packed)
    if len(text) <= max_chars:
        return text
    return _omitted(packed, max_chars, "RAG excerpts omitted: budget too small to retain text and source references.")


def _sample_points(points, count, numeric=True):
    """Select real samples; extrema/nonfinite representatives are never interpolated."""
    if len(points) <= count:
        return points, "all"
    chosen = {0, len(points) - 1}
    finite, exceptional = [], {}
    if numeric:
        for index, point in enumerate(points):
            try:
                if not isinstance(point, (list, tuple)) or len(point) != 2 or isinstance(point[1], bool):
                    raise ValueError("Uninterpreted sample shape")
                value = float(point[1])
                if math.isfinite(value):
                    finite.append((value, index))
                else:
                    exceptional.setdefault(str(point[1]), index)
            except (ValueError, TypeError, IndexError, KeyError):
                exceptional.setdefault("uninterpreted", index)
        if finite:
            chosen.update((min(finite)[1], max(finite)[1]))
        chosen.update(exceptional.values())
    # Required endpoints/extrema can exceed count; preserve them rather than
    # hide a nonfinite value. A higher-level budget can explicitly omit a series.
    for position in range(1, max(1, count - 1)):
        if len(chosen) >= count:
            break
        chosen.add(round(position * (len(points) - 1) / (count - 1)))
    method = "first_last_numeric_extrema_and_spaced_samples" if numeric else "first_last_and_spaced_histograms"
    return [points[index] for index in sorted(chosen)], method


def _prometheus_view(packed, result, data, max_chars):
    series = data.get("result")
    if not (isinstance(series, list) and series and all(
            isinstance(item, dict) and isinstance(item.get("metric"), dict) for item in series)):
        return None
    common = dict(series[0]["metric"])
    for item in series[1:]:
        common = {key: value for key, value in common.items()
                  if key in item["metric"] and item["metric"][key] == value}
    original_entries = []
    for index, item in enumerate(series, 1):
        entry = {key: deepcopy(value) for key, value in item.items() if key != "metric"}
        entry.update(series_index=index, labels={key: value for key, value in item["metric"].items() if key not in common})
        original_entries.append(entry)
    del data["result"]
    data.update(common_labels=common, series=deepcopy(original_entries), total_series=len(series),
                included_series=len(series), omitted_series=0, truncated=False)
    if len(_dump(packed)) <= max_chars:
        return _dump(packed)
    # Sample matrix points before dropping whole series. Rebuild each candidate
    # from its original points so repeated fitting does not relabel sampled data.
    if data.get("resultType") == "matrix":
        max_points = max((len(item.get(key, [])) for item in original_entries
                          for key in ("values", "histograms") if isinstance(item.get(key, []), list)), default=0)
        target = max(4, min(128, max_points // 2))
        while max_points > target:
            sampled = deepcopy(original_entries)
            for entry in sampled:
                for key in ("values", "histograms"):
                    points = entry.get(key)
                    if not isinstance(points, list) or len(points) <= target:
                        continue
                    selected, method = _sample_points(points, target, numeric=key == "values")
                    entry[key] = selected
                    entry[key + "_sampling"] = {
                        "total_points": len(points), "included_points": len(selected),
                        "omitted_points": len(points) - len(selected), "sampling_method": method,
                    }
            data["series"] = sampled
            data["truncated"] = True
            data["notice"] = "Sampled values retain original timestamps. Gaps cannot establish absence of incident activity."
            if len(_dump(packed)) <= max_chars:
                return _dump(packed)
            if target == 4:
                break
            target = max(4, target // 2)
    entries = data["series"]
    while len(_dump(packed)) > max_chars and len(entries) > 1:
        entries.pop()
        data.update(included_series=len(entries), omitted_series=len(series) - len(entries), truncated=True,
                    series_selection="source_order_prefix",
                    notice="Series and/or points omitted; a preview cannot establish absence. See saved evidence.")
    # Preserve exact retained label values. Missing labels are counted, never
    # shortened into values that might identify a different service or series.
    if len(_dump(packed)) > max_chars:
        candidates = [(container, key) for container in [common, entries[0]["labels"]]
                      for key in container if key != "__name__"]
        candidates.sort(key=lambda item: len(_dump({item[1]: item[0][item[1]]})), reverse=True)
        for container, key in candidates:
            del container[key]
            data["omitted_labels"] = data.get("omitted_labels", 0) + 1
            data["truncated"] = True
            data["labels_complete"] = False
            data["notice"] = "Labels, series and/or points omitted; retained series_index identifies the original series position."
            if len(_dump(packed)) <= max_chars:
                return _dump(packed)
    text = _dump(packed)
    if len(text) <= max_chars:
        return text
    # Never fall through to a generic preview that silently loses the last series.
    return _omitted(packed, max_chars, "Prometheus evidence omitted: no identified sample fits this budget.")


def prepare_model_evidence(evidence, max_chars, aliases=None):
    """Build one bounded view from original evidence; return the exact JSON sent.

    Alias translation happens before sizing. Callers should rebuild from saved
    evidence on each model turn rather than compact an already compacted view.
    """
    packed = _normalize(evidence)
    if aliases:
        packed = _aliases(packed, aliases)
    if isinstance(packed, dict) and isinstance(packed.get("chunks"), list):
        return _knowledge_view(packed, max_chars)
    result = packed.get("result") if isinstance(packed, dict) else None
    data = result.get("data") if isinstance(result, dict) else None
    if isinstance(data, dict) and data.get("resultType") in {"vector", "matrix"}:
        prepared = _prometheus_view(packed, result, data, max_chars)
        if prepared is not None:
            return prepared
    return bounded_json(packed, max_chars)


def compact_evidence(evidence, max_chars):
    """Compatibility entry point for callers without an evidence alias map."""
    return prepare_model_evidence(evidence, max_chars)


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
            keys = ("truncated", "text_truncated", "evidence_omitted", "truncated_by_script",
                    "total_series", "included_series", "omitted_series", "series_selection",
                    "total_points", "included_points", "omitted_points", "sampling_method",
                    "omitted_labels", "labels_complete", "total_items", "included_items", "omitted_items",
                    "total_fields", "omitted_fields", "original_chars", "omitted_chars",
                    "total_chunks", "included_chunks", "omitted_chunks", "deduplicated_chunks", "source_limit", "requested_limit",
                    "stdout_deduplicated", "metadata_fields_omitted")
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
    omitted = any(m.get("truncated") is True or m.get("evidence_omitted") is True or
                  any(isinstance(value, (int, float)) and value > 0 for key, value in m.items()
                      if key.startswith("omitted_")) for m in markers)
    return {"redacted": True, "changed": source != output, "model_omissions": omitted,
            "reason": "Bounded preview or evidence omission" if omitted else
                      ("Representation compacted without reported data omissions" if source != output else "Unchanged evidence"),
            "model_markers": markers, "source_reported_limits": source_markers or None,
            "source_completeness": "Not established; limits apply before saved evidence",
            "runner_output": result.get("output_metadata") if isinstance(result, dict) else None,
            "character_counts": "Unicode characters after the request's redaction policy, not tokens or bytes"}
