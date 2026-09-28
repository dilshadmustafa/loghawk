#read all files in a folder and generate identity mapping for each file using Ollama/Llama 3.2:3b.
"""
LogHawk - Folder-level Ollama-based schema identity mapping

Purpose
-------
Enumerate all raw log files in an S3 folder and generate one
identity-mapping JSON for each input file.

Example:

Raw input folder:

    s3://loghawk-data/raw/2026-09-28/

Containing:

    app1.json
    app2.json
    nginx.json

Identity mappings:

    s3://loghawk-data/identitymapping/2026-09-28/
        identitymapping_app1.json
        identitymapping_app2.json
        identitymapping_nginx.json

The identity mapping is generated using Ollama/Llama 3.2:3b.

The activity is designed to be:

    - folder based
    - file independent
    - idempotent
    - Temporal retry friendly

If a mapping JSON already exists for a raw file, that file is skipped.

Requirements
------------

    pip install requests s3fs

Ollama:

    ollama serve

Model:

    ollama pull llama3.2:3b
"""

import gzip
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

    Works with:

        AWS S3
        RustFS
        SeaweedFS
        MinIO
        Other S3-compatible storage
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
    Convert s3a:// to s3://.

    Spark normally uses s3a://.
    Python s3fs uses s3://.
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
# Normalize S3 folder
# ============================================================

def normalize_s3_folder(path: str) -> str:
    """
    Ensure an S3 folder ends with '/'.
    """

    if not (
        path.startswith("s3://")
        or path.startswith("s3a://")
    ):
        raise ValueError(
            f"Invalid S3 path: {path}"
        )

    if not path.endswith("/"):
        path += "/"

    return path


# ============================================================
# Derive identity-mapping folder
# ============================================================

def derive_identity_mapping_folder(
    raw_input_folder: str,
) -> str:
    """
    Convert:

        s3://loghawk-data/raw/2026-09-28/

    into:

        s3://loghawk-data/identitymapping/2026-09-28/
    """

    raw_input_folder = normalize_s3_folder(raw_input_folder)

    scheme = (
        "s3a://"
        if raw_input_folder.startswith("s3a://")
        else "s3://"
    )

    path_without_scheme = re.sub(
        r"^s3a?://",
        "",
        raw_input_folder,
    )

    parts = path_without_scheme.split("/", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid S3 path: {raw_input_folder}"
        )

    bucket = parts[0]
    key = parts[1]

    key_parts = [
        part
        for part in key.split("/")
        if part
    ]

    if "raw" not in key_parts:
        raise ValueError(
            "Raw input folder must contain '/raw/'.\n"
            f"Received: {raw_input_folder}"
        )

    raw_index = key_parts.index("raw")

    partition_parts = key_parts[raw_index + 1:]

    output_key_parts = [
        "identitymapping",
        *partition_parts,
    ]

    output_key = "/".join(output_key_parts) + "/"

    return f"{scheme}{bucket}/{output_key}"


# ============================================================
# Derive identity-mapping file
# ============================================================

def derive_identity_mapping_path(
    raw_input_path: str,
    identity_mapping_folder: str,
) -> str:
    """
    Derive the mapping file for one raw input file.

    Example:

        Raw:

        s3://loghawk-data/raw/2026-09-28/app1.json

        Mapping:

        s3://loghawk-data/identitymapping/2026-09-28/
            identitymapping_app1.json
    """

    raw_input_path = raw_input_path.rstrip("/")

    input_filename = raw_input_path.split("/")[-1]

    # Remove compression extension first.
    base_filename = input_filename

    if base_filename.endswith(".gz"):
        base_filename = base_filename[:-3]

    elif base_filename.endswith(".gzip"):
        base_filename = base_filename[:-6]

    # Remove normal extension.
    base_filename = os.path.splitext(
        base_filename
    )[0]

    output_filename = (
        f"identitymapping_{base_filename}.json"
    )

    identity_mapping_folder = normalize_s3_folder(
        identity_mapping_folder
    )

    return (
        identity_mapping_folder
        + output_filename
    )


# ============================================================
# Enumerate raw files
# ============================================================

def list_raw_files(
    raw_input_folder: str,
    fs: s3fs.S3FileSystem,
) -> list[str]:
    """
    Enumerate files directly under the raw input folder.

    Subdirectories are not recursively processed here.

    Returns s3:// paths.
    """

    raw_input_folder = normalize_s3_folder(
        raw_input_folder
    )

    filesystem_path = s3a_to_s3(
        raw_input_folder
    )

    print()
    print("=" * 70)
    print("Enumerating raw input files")
    print("=" * 70)

    print(f"Input folder: {raw_input_folder}")

    entries = fs.ls(
        filesystem_path,
        detail=True,
    )

    raw_files = []

    for entry in entries:

        if isinstance(entry, dict):

            entry_name = entry["name"]

            if entry.get("type") == "directory":
                continue

        else:

            entry_name = entry

        filename = entry_name.split("/")[-1]

        # Ignore directory-like entries.
        if not filename:
            continue

        # Ignore common Spark/control files.
        if filename.startswith("_"):
            continue

        if filename.startswith("."):
            continue

        # Only process files.
        if not (
            filename.endswith(".json")
            or filename.endswith(".json.gz")
            or filename.endswith(".jsonl")
            or filename.endswith(".jsonl.gz")
            or filename.endswith(".log")
            or filename.endswith(".log.gz")
        ):
            print(
                f"Skipping unsupported file: {filename}"
            )
            continue

        raw_files.append(
            "s3://" + entry_name
            if not entry_name.startswith("s3://")
            else entry_name
        )

    raw_files.sort()

    print()
    print(f"Raw files found: {len(raw_files)}")

    for index, path in enumerate(
        raw_files,
        start=1,
    ):
        print(
            f"  {index}. {path}"
        )

    return raw_files


# ============================================================
# Open text file from S3
# ============================================================

def open_s3_text_file(
    fs: s3fs.S3FileSystem,
    path: str,
):
    """
    Open a normal or gzip-compressed S3 file as text.
    """

    filesystem_path = s3a_to_s3(path)

    if path.endswith(".gz"):
        raw_file = fs.open(
            filesystem_path,
            "rb",
        )

        return gzip.open(
            raw_file,
            mode="rt",
            encoding="utf-8",
            errors="replace",
        )

    return fs.open(
        filesystem_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    )


# ============================================================
# Extract one representative JSON record
# ============================================================

def read_sample_record(
    fs: s3fs.S3FileSystem,
    input_path: str,
) -> dict[str, Any]:
    """
    Read one representative JSON record from a raw file.

    Supports the common LogHawk formats:

        JSON Lines:
            {"a": 1}
            {"a": 2}

        Single JSON object:
            {"a": 1, "b": 2}

    For JSON Lines, only the first non-empty record is read.
    This is intentional because the identity-mapping stage only
    needs schema/sample information.
    """

    print()
    print(
        f"Reading sample from: {input_path}"
    )

    with open_s3_text_file(
        fs,
        input_path,
    ) as f:

        # Read the first useful line.
        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                # JSON Lines / single object.
                if isinstance(record, dict):
                    return record

                # If the first line is a JSON array, use
                # its first object.
                if isinstance(record, list):

                    for item in record:

                        if isinstance(item, dict):
                            return item

                raise ValueError(
                    "JSON record is not an object."
                )

            except json.JSONDecodeError:

                # Continue reading in case the file is a
                # multi-line JSON object.
                break

        # ----------------------------------------------------
        # Fallback: read the complete file.
        #
        # This is intended for files containing one
        # pretty-printed JSON object/array.
        # ----------------------------------------------------

        f.seek(0)

        content = f.read()

        if not content.strip():
            raise ValueError(
                f"Input file is empty: {input_path}"
            )

        try:

            document = json.loads(content)

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Could not parse JSON from: "
                f"{input_path}\n"
                f"Error: {exc}"
            ) from exc

        if isinstance(document, dict):
            return document

        if isinstance(document, list):

            for item in document:

                if isinstance(item, dict):
                    return item

        raise ValueError(
            f"Could not find a JSON object in: "
            f"{input_path}"
        )


# ============================================================
# Ollama communication
# ============================================================

def chat_with_ollama(
    prompt: str,
) -> str:
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
    fs: s3fs.S3FileSystem,
) -> None:
    """
    Save one identity mapping JSON to S3/S3-compatible storage.
    """

    output_document = {
        "source": {
            "raw_input_path": raw_input_path,
            "column_names": column_names,
            "sample_row": sample_row,
        },

        "identity_mapping": mapping,
    }

    filesystem_path = s3a_to_s3(
        output_path
    )

    print()
    print("=" * 70)
    print("Writing identity mapping")
    print("=" * 70)

    print(f"S3 output: {output_path}")

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
    print(
        "Identity mapping successfully written."
    )


# ============================================================
# Check whether mapping exists
# ============================================================

def mapping_exists(
    fs: s3fs.S3FileSystem,
    mapping_path: str,
) -> bool:
    """
    Return True if the identity mapping already exists.
    """

    return fs.exists(
        s3a_to_s3(mapping_path)
    )


# ============================================================
# Process one raw file
# ============================================================

def process_one_file(
    raw_input_path: str,
    identity_mapping_path: str,
    fs: s3fs.S3FileSystem,
) -> str:
    """
    Generate identity mapping for one raw file.

    Existing mappings are skipped to make the operation
    idempotent and retry-friendly.
    """

    print()
    print("#" * 70)
    print("Processing raw file")
    print("#" * 70)

    print()
    print(
        f"Raw file:\n{raw_input_path}"
    )

    print()
    print(
        f"Mapping file:\n{identity_mapping_path}"
    )

    # --------------------------------------------------------
    # Idempotency check
    # --------------------------------------------------------

    if mapping_exists(
        fs,
        identity_mapping_path,
    ):

        print()
        print(
            "Mapping already exists."
        )

        print(
            "Skipping Ollama call."
        )

        return identity_mapping_path

    # --------------------------------------------------------
    # Read representative record
    # --------------------------------------------------------

    sample_row = read_sample_record(
        fs,
        raw_input_path,
    )

    if not sample_row:

        raise RuntimeError(
            f"No sample record found in: "
            f"{raw_input_path}"
        )

    column_names = list(
        sample_row.keys()
    )

    print()
    print("Input columns:")

    for column in column_names:
        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # Generate mapping
    # --------------------------------------------------------

    mapping = generate_identity_mapping(
        column_names=column_names,
        sample_row=sample_row,
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("Identity columns:")

    for column in mapping[
        "identity_columns"
    ]:
        print(
            f"  - {column}"
        )

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
        mapping[
            "recommended_entity_column"
        ],
    )

    print()
    print("Reason:")
    print(
        mapping["reason"]
    )

    # --------------------------------------------------------
    # Save mapping
    # --------------------------------------------------------

    save_identity_mapping_to_s3(
        mapping=mapping,
        output_path=identity_mapping_path,
        raw_input_path=raw_input_path,
        column_names=column_names,
        sample_row=sample_row,
        fs=fs,
    )

    return identity_mapping_path


# ============================================================
# Generate mappings for entire folder
# ============================================================

def generate_identity_mappings(
    raw_input_folder: str,
) -> list[str]:
    """
    Generate identity mappings for every raw file in a folder.

    This is the main function that should later be called by
    the Temporal identity-mapping activity.

    Processing model:

        raw folder
            |
            +-- file1
            +-- file2
            +-- file3
            |
            v
        mapping1
        mapping2
        mapping3

    The function returns only after every file has been
    successfully processed or already had a mapping.
    """

    raw_input_folder = normalize_s3_folder(
        raw_input_folder
    )

    identity_mapping_folder = (
        derive_identity_mapping_folder(
            raw_input_folder
        )
    )

    print()
    print("=" * 70)
    print("LogHawk Identity Mapping Stage")
    print("=" * 70)

    print()
    print(
        "Raw input folder:"
    )
    print(
        raw_input_folder
    )

    print()
    print(
        "Identity mapping folder:"
    )
    print(
        identity_mapping_folder
    )

    fs = create_s3_filesystem()

    # --------------------------------------------------------
    # Enumerate all raw files
    # --------------------------------------------------------

    raw_files = list_raw_files(
        raw_input_folder,
        fs,
    )

    if not raw_files:

        raise RuntimeError(
            "No supported raw log files found in:\n"
            f"{raw_input_folder}"
        )

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    generated_mappings = []

    failed_files = []

    for index, raw_file in enumerate(
        raw_files,
        start=1,
    ):

        print()
        print()
        print(
            "=" * 70
        )

        print(
            f"FILE {index} OF {len(raw_files)}"
        )

        print(
            "=" * 70
        )

        mapping_path = (
            derive_identity_mapping_path(
                raw_file,
                identity_mapping_folder,
            )
        )

        try:

            result = process_one_file(
                raw_input_path=raw_file,
                identity_mapping_path=mapping_path,
                fs=fs,
            )

            generated_mappings.append(
                result
            )

        except Exception as exc:

            print()
            print(
                "ERROR processing file:"
            )

            print(
                raw_file
            )

            print(
                f"Error: {exc}"
            )

            failed_files.append(
                (
                    raw_file,
                    str(exc),
                )
            )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("Identity Mapping Stage Completed")
    print("=" * 70)

    print()
    print(
        f"Total raw files:       {len(raw_files)}"
    )

    print(
        f"Successful mappings:   "
        f"{len(generated_mappings)}"
    )

    print(
        f"Failed mappings:       "
        f"{len(failed_files)}"
    )

    if failed_files:

        print()
        print(
            "Failed files:"
        )

        for raw_file, error in failed_files:

            print()
            print(
                f"  {raw_file}"
            )

            print(
                f"    {error}"
            )

        print()
        raise RuntimeError(
            f"Identity mapping failed for "
            f"{len(failed_files)} file(s). "
            "Stage A must not start."
        )

    print()
    print(
        "All identity mappings are available."
    )

    print(
        "Stage A can now proceed."
    )

    return generated_mappings


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # IMPORTANT
    #
    # Stage A / Temporal will pass a folder, not an individual
    # raw file.
    # ========================================================

    RAW_INPUT_FOLDER = (
        "s3://loghawk-data/raw/2026-09-28/"
    )

    # --------------------------------------------------------
    # Generate mappings for every file
    # --------------------------------------------------------

    mapping_paths = generate_identity_mappings(
        RAW_INPUT_FOLDER
    )

    # --------------------------------------------------------
    # Print final mapping list
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Generated Identity Mapping Files")
    print("=" * 70)

    for path in mapping_paths:

        print(
            f"  {path}"
        )

    print()
    print("=" * 70)
    print("Completed Successfully")
    print("=" * 70)