# sample output path s3a://loghawk-data/identitymapping/year=2026/month=09/day=23/identitymapping_sample_logs.json
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

The generated identity mapping is written to an S3/S3-compatible
location derived automatically from the corresponding raw input path.

Example:

    Raw input:

    s3a://loghawk-data/raw/year=2026/month=09/day=23/sample_logs.json

    Identity mapping:

    s3a://loghawk-data/identitymapping/year=2026/month=09/day=23/
        identitymapping_sample_logs.json

Requirements:

    pip install requests s3fs

Ollama:

    ollama serve

Model:

    ollama pull llama3.2:3b
"""

import json
import os
import re
from typing import Any

import requests
import s3fs
import loghawk.config as config

# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_TIMEOUT = 300

FALLBACK_ENTITY_ID = "unknown-entity"


# ============================================================
# S3 configuration
# ============================================================

# For RustFS / SeaweedFS / MinIO / other S3-compatible storage,
# configure the endpoint through an environment variable.
#
# Example:
#
# Windows PowerShell:
#
#   $env:S3_ENDPOINT_URL="http://localhost:9000"
#
# AWS:
#
#   Leave S3_ENDPOINT_URL unset.

S3_ENDPOINT_URL = os.getenv(
    "S3_ENDPOINT_URL",
    "http://localhost:9000",
)

S3_ACCESS_KEY_ID = config.LH_S3_ACCESS_KEY_ID

S3_SECRET_ACCESS_KEY = config.LH_S3_SECRET_ACCESS_KEY


# ============================================================
# Create S3 filesystem
# ============================================================

def create_s3_filesystem() -> s3fs.S3FileSystem:
    """
    Create an S3 filesystem connection.

    Works with AWS S3 and S3-compatible object stores such as
    RustFS, SeaweedFS and MinIO.

    Returns
    -------
    s3fs.S3FileSystem
    """

    return s3fs.S3FileSystem(
        key=S3_ACCESS_KEY_ID,
        secret=S3_SECRET_ACCESS_KEY,
        client_kwargs={
            "endpoint_url": S3_ENDPOINT_URL,
        },
    )


# ============================================================
# Convert s3a:// URI to s3:// URI
# ============================================================

def s3a_to_s3(path: str) -> str:
    """
    Convert an s3a:// URI to s3://.

    Spark uses s3a:// while Python s3fs uses s3://.
    """

    if path.startswith("s3a://"):
        return "s3://" + path[len("s3a://"):]

    if path.startswith("s3://"):
        return path

    raise ValueError(
        f"Unsupported S3 path: {path}\n"
        "Expected s3a:// or s3://"
    )


# ============================================================
# Derive identity-mapping output path
# ============================================================

def derive_identity_mapping_path(
    raw_input_path: str,
) -> str:
    """
    Derive the identity-mapping output path from the raw input path.

    Example:

        Input:
        s3a://loghawk-data/raw/year=2026/month=09/day=23/sample_logs.json

        Output:
        s3a://loghawk-data/identitymapping/year=2026/month=09/day=23/
            identitymapping_sample_logs.json
    """

    if not (
        raw_input_path.startswith("s3a://")
        or raw_input_path.startswith("s3://")
    ):
        raise ValueError(
            "Raw input path must start with s3a:// or s3://"
        )

    # --------------------------------------------------------
    # Keep the original URI scheme for the returned path.
    # --------------------------------------------------------

    scheme = "s3a://" if raw_input_path.startswith("s3a://") else "s3://"

    # Remove scheme for easier path manipulation.
    path_without_scheme = re.sub(
        r"^s3a?://",
        "",
        raw_input_path,
    )

    # --------------------------------------------------------
    # Split bucket and object key.
    # --------------------------------------------------------

    parts = path_without_scheme.split("/", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid S3 path: {raw_input_path}"
        )

    bucket = parts[0]
    key = parts[1]

    # --------------------------------------------------------
    # The raw input must contain /raw/
    # --------------------------------------------------------

    raw_marker = "/raw/"

    if raw_marker not in "/" + key:
        raise ValueError(
            "Raw input path must contain '/raw/'.\n"
            f"Received: {raw_input_path}"
        )

    # --------------------------------------------------------
    # Replace the first raw directory with identitymapping.
    #
    # Example:
    #
    # raw/year=2026/month=09/day=23/sample_logs.json
    #
    # becomes:
    #
    # identitymapping/year=2026/month=09/day=23/
    # identitymapping_sample_logs.json
    # --------------------------------------------------------

    key_parts = key.split("/")

    try:
        raw_index = key_parts.index("raw")

    except ValueError as exc:
        raise ValueError(
            f"Could not locate 'raw' directory in: "
            f"{raw_input_path}"
        ) from exc

    # Everything after raw contains:
    #
    # year=...
    # month=...
    # day=...
    # filename

    remaining_parts = key_parts[raw_index + 1:]

    if not remaining_parts:
        raise ValueError(
            f"No file found after /raw/ in: "
            f"{raw_input_path}"
        )

    input_filename = remaining_parts[-1]

    # Remove common compression extensions first.
    base_filename = input_filename

    if base_filename.endswith(".gz"):
        base_filename = base_filename[:-3]

    if base_filename.endswith(".gzip"):
        base_filename = base_filename[:-5]

    # --------------------------------------------------------
    # Remove existing extension.
    #
    # sample_logs.json
    #      ->
    # sample_logs
    #
    # sample_logs.parquet
    #      ->
    # sample_logs
    # --------------------------------------------------------

    base_filename = os.path.splitext(
        base_filename
    )[0]

    output_filename = (
        f"identitymapping_{base_filename}.json"
    )

    # --------------------------------------------------------
    # Build output directory.
    #
    # Preserve the partition directories:
    #
    # year=2026/month=09/day=23
    # --------------------------------------------------------

    partition_parts = remaining_parts[:-1]

    output_key_parts = [
        "identitymapping",
        *partition_parts,
        output_filename,
    ]

    output_key = "/".join(output_key_parts)

    return (
        f"{scheme}{bucket}/{output_key}"
    )


# ============================================================
# Ollama communication
# ============================================================

def chat_with_ollama(prompt: str) -> str:
    """
    Send a prompt to Ollama and return the model response.
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

Good entity candidates include:

    service
    application
    app_name
    application_id
    container
    container_name
    pod
    pod_name
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
            f"Ollama request timed out after "
            f"{OLLAMA_TIMEOUT} seconds."
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
# Build schema-discovery prompt
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
# Validate LLM mapping
# ============================================================

def validate_identity_mapping(
    mapping: dict[str, Any],
    column_names: list[str],
) -> dict[str, Any]:
    """
    Validate the mapping returned by the LLM.

    The LLM is allowed to suggest mappings, but it is not trusted
    to invent columns.
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
    # Validate identity columns
    # --------------------------------------------------------

    valid_identity_columns = []

    for column in identity_columns:

        if column in available_columns:

            if column not in valid_identity_columns:
                valid_identity_columns.append(column)

        else:

            print(
                f"WARNING: LLM suggested non-existent column "
                f"'{column}'. Ignoring it."
            )

    # --------------------------------------------------------
    # Validate priority order
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
    # Keep priority order consistent with identity columns
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

    return {
        "identity_columns": valid_identity_columns,
        "priority_order": valid_priority_order,
        "recommended_entity_column": (
            recommended_entity_column
        ),
        "fallback_entity_id": FALLBACK_ENTITY_ID,
        "reason": mapping.get(
            "reason",
            "No explanation provided by the LLM.",
        ),
    }


# ============================================================
# Generate identity mapping
# ============================================================

def generate_identity_mapping(
    column_names: list[str],
    sample_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask Ollama/Llama to determine the entity identity mapping.
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
    # Validate
    # --------------------------------------------------------

    return validate_identity_mapping(
        mapping,
        column_names,
    )


# ============================================================
# Save mapping to S3
# ============================================================

def save_identity_mapping_to_s3(
    mapping: dict[str, Any],
    output_path: str,
    raw_input_path: str,
    column_names: list[str],
    sample_row: dict[str, Any],
) -> None:
    """
    Save the identity mapping JSON to S3/S3-compatible storage.

    The output JSON also records the raw input path, which makes
    the mapping traceable to the source that generated it.
    """

    # --------------------------------------------------------
    # Add source metadata to the JSON.
    # --------------------------------------------------------

    output_document = {
        "source": {
            "raw_input_path": raw_input_path,
            "column_names": column_names,
            "sample_row": sample_row,
        },
        "identity_mapping": mapping,
    }

    # --------------------------------------------------------
    # Convert s3a:// to s3:// for s3fs.
    # --------------------------------------------------------

    filesystem_path = s3a_to_s3(output_path)

    print()
    print("=" * 70)
    print("Writing identity mapping")
    print("=" * 70)

    print(f"S3 output: {output_path}")

    fs = create_s3_filesystem()

    # --------------------------------------------------------
    # Write JSON directly to S3.
    # --------------------------------------------------------

    with fs.open(
        filesystem_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_document,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("Identity mapping successfully written.")
    print(f"Output: {output_path}")


# ============================================================
# Pretty print
# ============================================================

def print_identity_mapping(
    mapping: dict[str, Any],
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

        print(
            f"  {index}. {column}"
        )

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
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Raw input S3 path
    #
    # This is now the ONLY path that needs to be specified.
    #
    # The identity-mapping output path will be derived
    # automatically.
    # --------------------------------------------------------

    RAW_INPUT_PATH = (
        "s3a://loghawk-data/raw/"
        "year=2026/month=09/day=23/"
        "sample_logs.json"
    )

    # --------------------------------------------------------
    # Derive identity mapping output path.
    # --------------------------------------------------------

    IDENTITY_MAPPING_PATH = (
        derive_identity_mapping_path(
            RAW_INPUT_PATH
        )
    )

    print()
    print("=" * 70)
    print("LogHawk Identity Mapping")
    print("=" * 70)

    print()
    print("Raw input:")
    print(RAW_INPUT_PATH)

    print()
    print("Identity mapping output:")
    print(IDENTITY_MAPPING_PATH)

    # --------------------------------------------------------
    # Example input schema
    #
    # In the real Stage A pipeline these values should be
    # obtained from the raw input DataFrame.
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
    # One representative row
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
    # Ask Llama to generate identity mapping.
    # --------------------------------------------------------

    mapping = generate_identity_mapping(
        column_names=column_names,
        sample_row=sample_row,
    )

    # --------------------------------------------------------
    # Display mapping.
    # --------------------------------------------------------

    print_identity_mapping(mapping)

    # --------------------------------------------------------
    # Save mapping to S3.
    # --------------------------------------------------------

    save_identity_mapping_to_s3(
        mapping=mapping,
        output_path=IDENTITY_MAPPING_PATH,
        raw_input_path=RAW_INPUT_PATH,
        column_names=column_names,
        sample_row=sample_row,
    )

    print()
    print("=" * 70)
    print("Completed")
    print("=" * 70)