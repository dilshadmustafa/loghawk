# LogHawk — Codex Project Instructions

## Project

LogHawk is an AI-powered AIOps platform for multi-source log ingestion, anomaly detection, incident correlation, diagnosis/RCA, and deterministic remediation.

Repository: `C:\mywork\loghawk` GitHub: `https://github.com/dilshadmustafa/loghawk`

Environment:

-   Windows 11
-   Python 3.12.10, venv `venv312`
-   OpenJDK 17 at `C:\jdk-17`
-   PySpark 3.5.9
-   Temporal server: `localhost:7233`
-   Temporal task queue: `loghawk-pipeline`
-   Ollama local LLM; llama3.2:3b has been used for identity/schema mapping
-   LiteLLM is intended as the unified LLM gateway
-   AWS Bedrock is intended for cloud/enterprise inference
-   LanceDB for RAG/vector retrieval
-   DuckDB for conversation/history/structured analytics
-   n8n for lightweight integrations, scheduling and notifications
-   RustFS / S3-compatible object storage

Core principle:

> AI decides and explains; deterministic workflows execute, verify and, when necessary, roll back.

Do not replace deterministic execution with an LLM.

## Pipeline

1.  Identity Mapping
2.  Stage A — PySpark feature engineering
3.  Stage B — Isolation Forest anomaly detection
4.  Stage C — deterministic event/incident correlation
5.  Stage D — AI-assisted diagnosis/root-cause analysis
6.  Stage E — deterministic Temporal remediation, verification and rollback

Stage A/B and Temporal are implemented and being tested. Identity Mapping is being separated into its own Temporal activity. Stage C is the next major implementation stage.

## Object storage: critical protocol rule

RustFS is the S3-compatible object store.

Use:

-   `s3://` for Python `s3fs` / `fsspec`
-   `s3a://` for Apache Spark / Hadoop

Never pass `s3://` directly to Spark. Convert it first:

```python
def to_spark_s3_path(path: str) -> str:
    if path.startswith("s3://"):
        return "s3a://" + path[len("s3://"):]
    return path
```

Configuration includes `LH_S3_ENDPOINT`, `LH_S3_ACCESS_KEY_ID`, and `LH_S3_SECRET_ACCESS_KEY`.

## Current data layout

```text
s3://loghawk-data/raw/2026-09-28/
s3://loghawk-data/identitymapping/2026-09-28/
s3://loghawk-data/features/2026-09-28/
```

For every raw file:

```text
raw/<inputfilename>
  -> identitymapping/identitymapping_<inputfilename>.json
  -> features/<inputfilename-with-extension-removed>/
```

## Identity Mapping

Identity mapping is a separate Temporal activity. It must:

1.  Enumerate supported raw files.
2.  Generate one mapping JSON per raw file.
3.  Skip existing mappings.
4.  Be idempotent across Temporal retries.
5.  Complete for ALL raw files before Stage A starts.
6.  Fail if a required mapping cannot be generated.

Supported extensions currently include `.json`, `.json.gz`, `.jsonl`, `.jsonl.gz`, `.log`, `.log.gz`; skip files beginning with `_` or `.`.

Mapping JSON currently has this shape:

```json
{
  "source": {
    "raw_input_path": "...",
    "column_names": ["..."],
    "sample_row": {}
  },
  "identity_mapping": {
    "identity_columns": ["..."],
    "priority_order": ["..."],
    "recommended_entity_column": "...",
    "fallback_entity_id": "unknown-entity",
    "reason": "..."
  }
}
```

The mapping is nested under `identity_mapping`; do not read `priority_order` or `identity_columns` from the root.

LLM mapping rules: use only actual columns; never invent columns; preserve original names; prefer stable logical identity; do not use timestamp, message, severity/level, status, or exception as identity.

Future enhancement, only when requested: add field-role mapping such as `timestamp_column`, `level_column`, `message_column`, `status_code_column`, and `exception_column`. This makes Stage A truly source-independent.

## Stage A

Current source: `src/loghawk/feature_engineering/pyspark_s3_feature_engineering5.py`

Stage A accepts a folder, enumerates raw files, finds each file's corresponding identity mapping, processes each file, and writes a separate Parquet dataset under its own feature directory. Use one SparkSession for the folder run.

Conceptually:

```python
run(
    input_path="s3://loghawk-data/raw/2026-09-28/",
    output_path="s3://loghawk-data/features/2026-09-28/"
)
```

Python/s3fs may enumerate using `s3://`; every path actually passed to Spark must be converted to `s3a://`.

Stage A builds generic `entity_id` using mapping `priority_order`, with fallback `unknown-entity`. Do not hard-code `service` as universal identity.

Current feature columns:

```text
total_log_count
info_count
warning_count
error_count
error_rate
warning_rate
http_4xx_count
http_5xx_count
http_5xx_rate
timeout_count
timeout_rate
connection_error_count
authentication_failure_count
unique_exception_count
unique_error_message_count
```

Aggregation is by `entity_id` and one-minute time window. Avoid partitioning Parquet by high-cardinality `entity_id` unless measured workload justifies it.

Important sample-data caveat: current `container_logs.json` uses `status`, while the current Stage A recognizes `status_code`. Consequently HTTP 4xx/5xx features may not populate correctly until field-role mapping/alias handling is implemented.

## Stage B

Stage B uses scikit-learn Isolation Forest.

Current ML features:

```text
total_log_count
info_count
warning_count
error_count
http_4xx_count
http_5xx_count
timeout_count
connection_error_count
authentication_failure_count
unique_exception_count
unique_error_message_count
error_rate
warning_rate
http_5xx_rate
timeout_rate
```

Preserve metadata where available:

```text
timestamp
entity_id
service
application_id
app_name
container_name
pod_name
namespace
hostname
host
database
device
```

Do not derive service identity by parsing S3 paths. Stage B may need to recursively/enumeratively load the per-file feature datasets under the features date folder. Use `s3://` for fsspec and `s3a://` for Spark.

## Stage C

Initial deterministic correlation rules:

1.  Temporal correlation — anomalies within about 5 minutes.
2.  Entity correlation — multiple anomalous entities in the same time window.
3.  Severity/anomaly correlation — critical anomalies have greater correlation significance.

Example:

```text
10:10 payment-service CRITICAL
10:11 payment-service CRITICAL
10:11 order-service WARNING
10:12 inventory-service WARNING
=> Incident-001, 10:10–10:12

11:30 auth-service CRITICAL
=> Incident-002, 11:30
```

Future signals: dependency topology, error signatures, trace IDs, Kubernetes pod/node relationships, deployment/change events, database/network dependencies.

Initial output target discussed: `s3://loghawk-data/incidents/year=2026/month=09/day=23/correlated_incidents.parquet`

Keep Stage C deterministic unless explicitly asked to introduce AI.

## Temporal

Server: `localhost:7233` Task queue: `loghawk-pipeline`

Target sequence:

```text
Identity Mapping -> Stage A -> Stage B -> Stage C -> Stage D -> Stage E
```

Use:

```python
from temporalio.common import RetryPolicy
```

not `workflow.RetryPolicy`.

Current activities include `run_stage_a(input_path, output_path)` and `run_stage_b(input_path, output_path)`. Next activity should be `run_identity_mapping(raw_folder)`, followed by Stage A. Identity mapping must finish before Stage A starts.

Activities should be retryable and idempotent where practical. Use positional args when the activity signature expects positional parameters.

## AI vs deterministic execution

LLMs are appropriate for identity/schema interpretation, diagnosis, RCA, explanations, and remediation recommendations.

Execution must remain deterministic:

```text
AI recommendation
  -> validated deterministic action
  -> Temporal execution
  -> verification
  -> success OR rollback
```

## Scale / retention context

A representative workload discussed is about 24 GB/day of logs. Retention scenarios discussed include 10 days and a 45-day rolling window. Keep retention configurable; do not hard-code vendor-specific storage behavior.

## Coding guidelines

-   Inspect current code before changing it.
-   Preserve working behavior unless the task explicitly changes it.
-   Prefer small, explicit functions and configuration-driven behavior.
-   Preserve public function signatures where practical.
-   Keep Temporal activities idempotent.
-   Keep `s3://` vs `s3a://` explicit.
-   Avoid hard-coded service names and source-specific assumptions.
-   Do not introduce an LLM where deterministic logic is sufficient.
-   Do not overwrite unrelated user changes.
-   Log enough information to diagnose failures.
-   Never claim tests passed unless they were actually run.

## Known issue from 2026-09-28

Stage A folder enumeration worked and found two files:

-   `container_logs.json`
-   `loghawk_sample_logs.json`

Identity mapping lookup also worked. Both files then failed because `s3://...` was passed directly to Spark, producing: `UnsupportedFileSystemException: No FileSystem for scheme "s3"`.

This is an S3 protocol boundary bug, NOT a reason to redesign the folder architecture. Convert only Spark paths to `s3a://`.

## Useful commands

```powershell
C:\mywork\loghawk\venv312\Scripts\Activate.ps1
python src\loghawk\feature_engineering\pyspark_s3_feature_engineering5.py
git status
git diff
```

## Working style for Codex

When modifying LogHawk:

1.  Read this file and inspect the repository.
2.  Identify the smallest change that satisfies the task.
3.  Inspect all affected pipeline stages before changing cross-stage behavior.
4.  Preserve unrelated user changes.
5.  Run relevant syntax/unit/integration checks where practical.
6.  Report changed files and actual test results.
7.  If tests fail, report and diagnose the failure rather than claiming success.
8.  Before a major redesign, check whether the issue is instead a path/protocol, schema, mapping, retry, or source-specific assumption.

```plaintext
Inspect
↓
Explain
↓
Show proposed changes / diff
↓
Ask for approval
↓
WAIT
↓
Apply approved changes
↓
Verify
```

## Architecture summary

```text
Multi-source logs
      |
      v
RustFS / S3
      |
      v
Identity Mapping
      |
      v
PySpark Feature Engineering
      |
      v
Isolation Forest
      |
      v
Deterministic Incident Correlation
      |
      v
AI Diagnosis / RCA
      |
      v
Temporal Remediation
      |
      v
Verification / Rollback
```

Permanent design decisions:

-   one identity mapping per raw file
-   identity mapping is a separate Temporal activity
-   Stage A processes a raw folder and produces per-file feature datasets
-   generic `entity_id`, not hard-coded `service`
-   `s3://` for Python S3 libraries and `s3a://` for Spark
-   AI recommends/explains; deterministic code executes/verifies/rolls back
-   avoid high-cardinality entity partitioning unless justified by measurements
