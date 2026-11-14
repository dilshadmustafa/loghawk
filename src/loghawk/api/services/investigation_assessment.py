"""Validated investigation assessments and deterministic evidence references."""
import hashlib
import json

from loghawk.builtin_tasks.contracts import validate_arguments

CLAIM = {
    "type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 1500,
                 "description": "One concise evidence-based claim; preserve material uncertainty and coverage limits."},
        "references": {"type": "array", "items": {"type": "string", "maxLength": 200},
                       "maxItems": 10,
                       "description": "At most 10 distinct supplied evidence labels supporting this claim. Cite primary records; do not repeat labels."},
    }, "required": ["text", "references"], "additionalProperties": False,
}
ASSESSMENT_SCHEMA = {
    "type": "object", "properties": {
        **{key: {"type": "array", "items": CLAIM, "maxItems": 10}
           for key in ("observations", "hypotheses", "contradictions")},
        **{key: {"type": "array", "items": {"type": "string", "maxLength": 1000}, "maxItems": 10}
           for key in ("missing_evidence", "next_steps")},
        "conclusion": CLAIM,
    },
    "required": ["observations", "hypotheses", "contradictions", "missing_evidence", "conclusion", "next_steps"],
    "additionalProperties": False,
}
SEARCH_SCHEMA = {
    "type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
    }, "required": ["query"], "additionalProperties": False,
}


def tools():
    return [
        {"type": "function", "function": {
            "name": "search_knowledge",
            "description": "Retrieve saved RAG document excerpts as background knowledge, not live observations.",
            "parameters": SEARCH_SCHEMA}},
        {"type": "function", "function": {
            "name": "finish_investigation",
            "description": "Finish with a concise assessment of roughly 500-700 tokens: 3-5 key observations where supported, hypotheses, contradictions, missing evidence, conclusion and next steps. Cite primary supplied evidence labels, at most 10 distinct labels per claim. Observations require references. Preserve uncertainty and coverage limits. Call alone; the validated assessment is the final response.",
            "parameters": ASSESSMENT_SCHEMA}},
    ]


def fingerprint(target, arguments):
    payload = [target.get("kind"), target.get("task_id", target["name"]),
               target["version"], target.get("artifact_id"), arguments]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def context_references(snapshot):
    """References are derived from immutable selected records, not user-supplied claims."""
    refs = {}
    for incident in snapshot.get("incidents", []):
        refs["incident:" + incident["id"]] = "imported incident"
    for finding in snapshot.get("findings", []):
        refs["finding:" + finding["id"]] = "user finding"
    for source in snapshot.get("evidence", []):
        for row in source.get("records", []):
            refs[f"source:{source['source_id']}:record:{row['record_number']}"] = "imported record"
        if source.get("collector_evidence"):
            refs["source:" + source["source_id"]] = "attached collector result"
    return refs


def assign_alias(reference, aliases):
    """Assign a stable, request-scoped model label to a canonical evidence reference."""
    if reference not in aliases:
        aliases[reference] = f"E{len(aliases) + 1}"
    return aliases[reference]


def assign_aliases(references, aliases=None):
    result = dict(aliases or {})
    for reference in references:
        assign_alias(reference, result)
    return result


def model_context(value, aliases, *, hide_source_ids=False):
    """Prepare evidence for a model without exposing canonical evidence IDs."""
    if isinstance(value, list):
        return [model_context(item, aliases, hide_source_ids=hide_source_ids) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if hide_source_ids and key == "source_id":
                continue
            if key == "evidence_ref" and isinstance(item, str):
                result[key] = aliases.get(item, "[unavailable]")
            else:
                result[key] = model_context(item, aliases, hide_source_ids=hide_source_ids)
        if isinstance(value.get("evidence_ref"), str) and value["evidence_ref"] in aliases and "id" in result:
            result["id"] = aliases[value["evidence_ref"]]
        return result
    return value


def model_assessment(value, aliases):
    """Translate canonical references in a saved assessment to model-facing labels."""
    result = json.loads(json.dumps(value))
    for key in ("observations", "hypotheses", "contradictions", "conclusion"):
        claims = [result[key]] if key == "conclusion" else result[key]
        for claim in claims:
            claim["references"] = [aliases.get(reference, "[unavailable]")
                                   for reference in claim["references"]]
    return result


def validate_model_assessment(value, aliases, references):
    """Check shape and labels before deduplicating exact repeats and enforcing limits."""
    # The normal schema caps distinct references after repeated labels are removed.
    # All other shape and global JSON-size constraints still apply before access.
    shape_schema = json.loads(json.dumps(ASSESSMENT_SCHEMA))
    for key in ("observations", "hypotheses", "contradictions", "conclusion"):
        schema = shape_schema["properties"][key]
        claim_schema = schema if key == "conclusion" else schema["items"]
        claim_schema["properties"]["references"].pop("maxItems")
    result = validate_arguments(shape_schema, value)
    alias_to_reference = {
        alias: reference for reference, alias in aliases.items()
        if reference in references
    }
    claims = [*result["observations"], *result["hypotheses"],
              *result["contradictions"], result["conclusion"]]
    # Check every supplied label, including those after the tenth entry.
    for claim in claims:
        if any(label not in alias_to_reference for label in claim["references"]):
            raise ValueError(
                "Unknown or unavailable evidence label(s). Cite only the supplied E-labels."
            )
    for claim in claims:
        labels = list(dict.fromkeys(claim["references"]))
        claim["references"] = [alias_to_reference[label] for label in labels]
    # More than ten unique references fails; no distinct reference is discarded.
    return validate(result, references)


def validate(value, references):
    result = validate_arguments(ASSESSMENT_SCHEMA, value)
    for key in ("observations", "hypotheses", "contradictions", "conclusion"):
        claims = [result[key]] if key == "conclusion" else result[key]
        for claim in claims:
            unknown = set(claim["references"]) - set(references)
            if unknown:
                raise ValueError("Unknown evidence reference(s): " + ", ".join(sorted(unknown)))
            if key == "observations" and not claim["references"]:
                raise ValueError("Every observation requires a saved evidence reference.")
    return result


def partial(reason):
    return {"observations": [], "hypotheses": [], "contradictions": [],
            "missing_evidence": [reason[:1000]],
            "conclusion": {"text": "Investigation incomplete; saved evidence does not establish a verified root cause.",
                           "references": []},
            "next_steps": ["Review saved evidence and the stop reason before starting another request."]}


def render(value):
    lines = []
    for key in ("observations", "hypotheses", "contradictions", "missing_evidence", "conclusion", "next_steps"):
        lines.append(key.replace("_", " ").title())
        entries = [value[key]] if key == "conclusion" else value[key]
        for item in entries:
            if isinstance(item, str):
                lines.append("- " + item)
            else:
                refs = ", ".join(item["references"])
                lines.append("- " + item["text"] + (f" [{refs}]" if refs else ""))
        if not entries:
            lines.append("- None reported.")
        lines.append("")
    return "\n".join(lines)
