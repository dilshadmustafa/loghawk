"""
LogHawk - Ollama-based schema identity mapping

Purpose:
    Analyze an unknown log schema using a local Ollama LLM and
    generate an identity-column mapping for Stage A.

The LLM receives:
    - Input column names
    - One representative input row

The LLM returns:
    - Candidate identity columns
    - Priority order
    - Recommended entity_id column
    - Reasoning

The generated mapping can then be used by deterministic
PySpark Stage A code.

Requirements:
    pip install requests pandas pyarrow

Ollama:
    ollama serve

Model:
    ollama pull llama3.2:3b
"""

import json
from typing import Any

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_TIMEOUT = 300

FALLBACK_ENTITY_ID = "unknown-entity"


# ============================================================
# Ollama communication
# ============================================================

def chat_with_ollama(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the model response.

    Parameters
    ----------
    prompt:
        User prompt to send to the local LLM.

    Returns
    -------
    str
        Raw text response from Ollama.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
You are a senior data engineer specializing in:

- Apache Spark
- PySpark
- log analytics
- observability
- AIOps
- schema discovery
- log normalization
- application and infrastructure telemetry

You are helping LogHawk analyze logs from many different sources.

Your job is to identify which input columns can represent the
logical entity that generated a log record.

Examples of entities include:

- service
- application
- application instance
- container
- Kubernetes pod
- deployment
- namespace
- host
- server
- database
- device
- process

You must carefully distinguish an ENTITY identifier from an
EVENT attribute.

For example:

Good entity candidates:
    service
    application
    app_name
    container
    pod
    hostname
    host
    database
    device

Normally NOT entity candidates:
    timestamp
    message
    log_level
    severity
    status_code
    exception
    stack_trace
    error_message

Your output must be valid JSON only.
Do not include Markdown.
Do not include ```json fences.
""",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama.\n"
            "Make sure Ollama is running with:\n"
            "    ollama serve"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT} seconds."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}"
        ) from exc

    result = response.json()

    if "message" not in result:
        raise RuntimeError(
            f"Unexpected Ollama response:\n{result}"
        )

    content = result["message"].get("content")

    if not content:
        raise RuntimeError(
            f"Ollama returned an empty response:\n{result}"
        )

    return content


# ============================================================
# Build the schema-discovery prompt
# ============================================================

def build_identity_mapping_prompt(
    column_names: list[str],
    sample_row: dict[str, Any],
) -> str:
    """
    Build the prompt sent to Llama for identity discovery.
    """

    return f"""
You are analyzing a previously unknown log source for LogHawk.

LogHawk needs a generic entity_id for anomaly detection.

The entity_id should represent the logical component, system,
application, infrastructure component, or device responsible
for generating the log event.

INPUT COLUMN NAMES
==================

{json.dumps(column_names, indent=2)}


ONE REPRESENTATIVE INPUT ROW
============================

{json.dumps(sample_row, indent=2, default=str)}


TASK
====

Analyze the schema and sample row.

Identify columns that can represent the entity that generated
the log event.

Possible examples include:

- service
- application
- application_id
- app_name
- container
- container_name
- pod
- pod_name
- namespace
- hostname
- host
- database
- device
- process
- component

Do NOT assume that the source contains a "service" column.

Do NOT assume that the source is Kubernetes.

Do NOT assume that the source is an application log.

The same LogHawk Stage A pipeline must support many different
types of log sources.


IMPORTANT RULES
===============

1. Only select columns that actually exist in the supplied
   column list.

2. Do not invent column names.

3. Do not use timestamp as an entity identifier.

4. Do not use message as an entity identifier.

5. Do not use severity or log level as an entity identifier.

6. Do not use status code as an entity identifier.

7. Do not use exception or stack trace as an entity identifier.

8. Prefer a stable logical identity over an ephemeral identity
   when both are available.

9. If application and pod are both available, application may
   generally represent the logical entity while pod represents
   an instance.

10. If only a host/server identity exists, that can be used.

11. If multiple useful identity columns exist, return them in
    priority order.

12. If no suitable identity column exists, return an empty list
    and use "unknown-entity" as the fallback.

13. Preserve the original column names exactly.

14. Do not generate PySpark code.

15. Return only JSON.


OUTPUT FORMAT
=============

Return exactly this JSON structure:

{{
    "identity_columns": [
        "column1",
        "column2"
    ],
    "priority_order": [
        "column1",
        "column2"
    ],
    "recommended_entity_column": "column1",
    "fallback_entity_id": "unknown-entity",
    "reason": "Short explanation of why these columns represent the entity."
}}

The "recommended_entity_column" should contain the single
preferred logical identity column.

If no suitable identity column exists:

{{
    "identity_columns": [],
    "priority_order": [],
    "recommended_entity_column": null,
    "fallback_entity_id": "unknown-entity",
    "reason": "No suitable entity identity column was found."
}}
"""


# ============================================================
# Validate the LLM mapping
# ============================================================

def validate_identity_mapping(
    mapping: dict[str, Any],
    column_names: list[str],
) -> dict[str, Any]:
    """
    Validate the mapping returned by the LLM.

    The LLM is allowed to suggest mappings, but it is not trusted
    to invent columns.

    Any identity column returned by the model must exist in the
    actual input schema.
    """

    available_columns = set(column_names)

    identity_columns = mapping.get(
        "identity_columns",
        [],
    )

    priority_order = mapping.get(
        "priority_order",
        [],
    )

    recommended_entity_column = mapping.get(
        "recommended_entity_column"
    )

    # --------------------------------------------------------
    # Validate identity_columns
    # --------------------------------------------------------

    valid_identity_columns = []

    for column in identity_columns:

        if column in available_columns:
            valid_identity_columns.append(column)

        else:
            print(
                f"WARNING: LLM suggested non-existent column "
                f"'{column}'. Ignoring it."
            )

    # --------------------------------------------------------
    # Validate priority_order
    # --------------------------------------------------------

    valid_priority_order = []

    for column in priority_order:

        if column in available_columns:

            if column not in valid_priority_order:
                valid_priority_order.append(column)

        else:
            print(
                f"WARNING: LLM suggested non-existent priority "
                f"column '{column}'. Ignoring it."
            )

    # --------------------------------------------------------
    # Keep priority_order consistent with identity_columns
    # --------------------------------------------------------

    for column in valid_identity_columns:

        if column not in valid_priority_order:
            valid_priority_order.append(column)

    # --------------------------------------------------------
    # Validate recommended entity column
    # --------------------------------------------------------

    if (
        recommended_entity_column
        not in available_columns
    ):
        recommended_entity_column = (
            valid_priority_order[0]
            if valid_priority_order
            else None
        )

    # --------------------------------------------------------
    # Build validated result
    # --------------------------------------------------------

    validated_mapping = {
        "identity_columns": valid_identity_columns,
        "priority_order": valid_priority_order,
        "recommended_entity_column": recommended_entity_column,
        "fallback_entity_id": FALLBACK_ENTITY_ID,
        "reason": mapping.get(
            "reason",
            "No explanation provided by the LLM."
        ),
    }

    return validated_mapping


# ============================================================
# Main identity mapping function
# ============================================================

def generate_identity_mapping(
    column_names: list[str],
    sample_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask Ollama/Llama to determine the entity identity mapping.

    Parameters
    ----------
    column_names:
        List of columns from the input log source.

    sample_row:
        One representative row from the input dataset.

    Returns
    -------
    dict
        Validated identity mapping.
    """

    prompt = build_identity_mapping_prompt(
        column_names=column_names,
        sample_row=sample_row,
    )

    print()
    print("=" * 70)
    print("Sending schema to Ollama")
    print("=" * 70)

    print(f"Model: {OLLAMA_MODEL}")
    print(f"Columns: {len(column_names)}")

    raw_response = chat_with_ollama(prompt)

    print()
    print("=" * 70)
    print("Raw Ollama response")
    print("=" * 70)

    print(raw_response)

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        mapping = json.loads(raw_response)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Ollama did not return valid JSON.\n\n"
            f"Raw response:\n{raw_response}"
        ) from exc

    # --------------------------------------------------------
    # Validate mapping
    # --------------------------------------------------------

    validated_mapping = validate_identity_mapping(
        mapping,
        column_names,
    )

    return validated_mapping


# ============================================================
# Pretty-print mapping
# ============================================================

def print_identity_mapping(
    mapping: dict[str, Any]
) -> None:

    print()
    print("=" * 70)
    print("LogHawk Identity Mapping")
    print("=" * 70)

    print()

    print("Identity columns:")
    for column in mapping["identity_columns"]:
        print(f"  - {column}")

    print()

    print("Priority order:")
    for index, column in enumerate(
        mapping["priority_order"],
        start=1,
    ):
        print(f"  {index}. {column}")

    print()

    print(
        "Recommended entity column:",
        mapping["recommended_entity_column"],
    )

    print(
        "Fallback:",
        mapping["fallback_entity_id"],
    )

    print()

    print("Reason:")
    print(mapping["reason"])

    print("=" * 70)


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example input schema
    #
    # In the real Stage A pipeline these will come from the
    # actual input DataFrame.
    # --------------------------------------------------------

    column_names = [
        "timestamp",
        "application",
        "container",
        "pod",
        "namespace",
        "level",
        "message",
        "status",
        "exception",
    ]

    # --------------------------------------------------------
    # One representative row from the input file
    # --------------------------------------------------------

    sample_row = {
        "timestamp": "2026-09-27T10:15:32",
        "application": "payment-api",
        "container": "payment-api",
        "pod": "payment-api-7f8d9c",
        "namespace": "production",
        "level": "ERROR",
        "message": "Database connection timeout",
        "status": 500,
        "exception": "TimeoutException",
    }

    # --------------------------------------------------------
    # Ask Llama to generate the mapping
    # --------------------------------------------------------

    mapping = generate_identity_mapping(
        column_names=column_names,
        sample_row=sample_row,
    )

    # --------------------------------------------------------
    # Display the validated mapping
    # --------------------------------------------------------

    print_identity_mapping(mapping)

    # --------------------------------------------------------
    # Save mapping for Stage A
    # --------------------------------------------------------

    with open(
        "identity_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            mapping,
            f,
            indent=4,
        )

    print()
    print(
        "Mapping saved to: identity_mapping.json"
    )