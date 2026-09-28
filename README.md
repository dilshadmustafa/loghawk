Welcome to LogHawk
===================

<center><img src="https://raw.githubusercontent.com/dilshadmustafa/loghawk/main/loghawk_logo.jpg" width="50%"></center>

[![](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=H4V87SN5M2GG2)

Introduction
-------------

**LogHawk** is an AI-powered AIOps platform for detecting production anomalies, correlating operational incidents, performing RAG-based root-cause analysis, and progressively automating remediation across cloud, Kubernetes, and other operational environments.

LogHawk started as a local, privacy-focused, detection-first Agentic RAG platform for security threat detection using multi-source logging and local LLMs. Its evolution is toward a broader **AIOps / Intelligent SRE platform** combining logs, metrics and traces with anomaly detection, incident intelligence, operational knowledge, AI-assisted RCA, and controlled automated remediation.

The platform supports private/local deployments on consumer-grade systems with 16GB or 32GB RAM as well as enterprise/cloud deployments using technologies such as Amazon Bedrock, Kubernetes, AWS and Temporal.

> **Core design principle:** **AI decides and explains; deterministic workflows execute, verify and, when necessary, roll back.**

### Project Goals

The long-term goal is to move LogHawk through:

**Observe → Detect → Correlate → Diagnose → Recommend → Remediate → Verify → Learn**

### Objectives And Features

> -   **Multi-source observability:** ingest logs from Elasticsearch, Splunk, files, JSON and CSV, with future support for metrics, traces and OpenTelemetry.
> -   **AI-powered log intelligence:** combine deterministic detection, machine learning and GenAI for operational and security analysis.
> -   **Detection-first anomaly detection:** detect volume, error-rate, HTTP 4xx/5xx, timeout, connection-error, authentication-failure and novel-event anomalies.
> -   **Feature-based detection:** normalize logs, aggregate them into time windows, engineer numerical features, establish baselines and apply statistical/time-series detectors plus Isolation Forest.
> -   **Incident correlation:** group related anomalies into coherent incidents instead of producing alert storms.
> -   **Security threat detection:** retain NIST, CVE and MITRE ATT&CK knowledge and security-focused RAG.
> -   **Operational RAG:** ingest runbooks, incident history, architecture documentation and troubleshooting material in addition to PDF/HTML/Markdown/CSV/JSON sources.
> -   **AI-assisted RCA:** generate incident summaries, probable root causes, supporting evidence, uncertainty and recommended actions from retrieved context.
> -   **Amazon Bedrock integration:** use managed GenAI for enterprise RAG, RCA, agent reasoning and tool selection.
> -   **Agentic AIOps:** allow an AI agent to investigate incidents and recommend actions through controlled tools.
> -   **Durable remediation:** use Temporal for stateful workflows, retries, timeouts, approval waits, verification, rollback and escalation.
> -   **Big-data processing:** use PySpark for large-scale ingestion, normalization, aggregation and feature engineering.
> -   **Workflow automation:** use n8n for scheduled ingestion, knowledge refreshes, notifications and lightweight integrations.
> -   **Durable data pipeline:** use RustFS S3-compatible object storage as a local/cloud-neutral data boundary between ingestion, feature engineering, anomaly detection and downstream AI processing.

---

# LogHawk Architecture

## Core Architectural Principle

LogHawk separates **data processing**, **workflow orchestration**, **AI reasoning**, and **execution**.

The machine-learning and data-processing stages should remain independently executable Python/PySpark programs. Temporal coordinates those stages but does not contain their business logic.

```text
                         LogHawk
                            |
             +--------------+--------------+
             |                             |
             v                             v
       Data Processing              AI / Intelligence
       PySpark / ML                 LiteLLM / RAG / LLM
             |                             |
             +-------------+---------------+
                           |
                           v
                    Workflow Layer
                        Temporal
                           |
                           v
                    Execution Layer
                  Kubernetes / AWS / etc.
```

This separation allows LogHawk to run:

-   individual processing stages during development;
-   the complete pipeline through Temporal;
-   local inference through Ollama;
-   cloud inference through Amazon Bedrock;
-   lightweight external automation through n8n;
-   deterministic remediation through Temporal workflows.

---

# Stage-Based AIOps Architecture

The core LogHawk pipeline is divided into stages.

```text
                    LogHawk AIOps Pipeline

 Stage 0          Stage A             Stage B
 Ingestion        Feature             Anomaly
                  Engineering         Detection
    |                 |                   |
    v                 v                   v
 Raw Logs ------> PySpark ----------> IsolationForest
    |                 |                   |
    |                 v                   v
    |            Feature Parquet      Anomaly Results
    |                 |                   |
    +-----------------+-------------------+
                                      |
                                      v
                                  Stage C
                             Incident Intelligence
                                      |
                                      v
                                  Stage D
                                  AI / RAG
                                      |
                                      v
                                  Stage E
                               Remediation
```

The initial implementation focuses on **Stage A and Stage B**.

---

# Stage A — Identity Mapping and Feature Engineering

Stage A now uses a **folder-based, per-input-file processing convention**.

Before feature engineering begins, a separate Identity Mapping activity enumerates the raw input folder and creates one identity-mapping JSON document for each raw input file.

The processing contract is:

```text
Raw input folder
    |
    +-- input-file-1.json
    +-- input-file-2.json
    +-- input-file-3.json
    |
    v
Identity Mapping
    |
    +-- identitymapping_input-file-1.json
    +-- identitymapping_input-file-2.json
    +-- identitymapping_input-file-3.json
    |
    v
Stage A
    |
    +-- process input-file-1 using its mapping
    +-- process input-file-2 using its mapping
    +-- process input-file-3 using its mapping
    |
    v
Per-input Feature Datasets
```

### Identity Mapping

Identity Mapping is a separate Temporal activity.

For each raw file:

```text
raw/<inputfilename>
        |
        v
identitymapping/identitymapping_<inputfilename>.json
```

Example:

```text
s3://loghawk-data/raw/2026-09-28/container_logs.json

        |

s3://loghawk-data/identitymapping/2026-09-28/
    identitymapping_container_logs.json
```

The mapping contains the source information and the logical identity mapping:

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

The Identity Mapping activity is idempotent:

- existing mappings are skipped;
- successful mappings are not regenerated during a retry;
- Stage A starts only after all required mappings have been generated successfully.

The LLM is used to interpret source identity fields, but it must use only actual source columns and must not invent identity fields.

### Stage A Folder Convention

Stage A enumerates:

```text
s3://loghawk-data/raw/2026-09-28/
```

For every raw file it finds the corresponding mapping and produces a separate feature dataset.

Example:

```text
s3://loghawk-data/raw/2026-09-28/
    |
    +-- container_logs.json
    |
    +-- loghawk_sample_logs.json
```

becomes:

```text
s3://loghawk-data/features/2026-09-28/
    |
    +-- container_logs/
    |     +-- *.parquet
    |
    +-- loghawk_sample_logs/
          +-- *.parquet
```

The complete contract is:

```text
raw/
  container_logs.json
        |
        +--> identitymapping_container_logs.json
        |
        +--> features/container_logs/

  loghawk_sample_logs.json
        |
        +--> identitymapping_loghawk_sample_logs.json
        |
        +--> features/loghawk_sample_logs/
```

### S3 Protocol Convention

LogHawk deliberately distinguishes Python S3 access from Spark S3 access.

```text
Python / s3fs / fsspec
        |
        +--> s3://

Apache Spark / Hadoop
        |
        +--> s3a://
```

For example:

```python
# Python / fsspec
s3://loghawk-data/raw/2026-09-28/

# Spark
s3a://loghawk-data/raw/2026-09-28/container_logs.json
```

An `s3://` URI must not be passed directly to Spark when the configured Hadoop filesystem is `s3a`.

This separation is an important implementation detail of the current local RustFS setup.

### Stage A Processing

For each raw file, Stage A:

1. loads the corresponding identity mapping;
2. infers the source schema;
3. validates the identity mapping;
4. constructs the processing schema;
5. normalizes fields;
6. creates `entity_id`;
7. aggregates events into one-minute windows;
8. generates numerical features;
9. writes the feature dataset to the input-specific output folder.

### Example Features

Logs are aggregated into time windows, initially using a one-minute window.

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

### Entity Identity

Stage A does not assume that `service` is the universal identity.

The identity mapping can identify logical entities such as:

```text
application
service
container
pod
host
database
device
```

The resulting generic field is:

```text
entity_id
```

The identity mapping's `priority_order` determines which identity field is preferred.

Fallback:

```text
unknown-entity
```

### Future Field-Role Mapping

The current mapping primarily determines entity identity.

A future enhancement will allow the same mapping document to identify source-specific field roles:

```json
{
  "identity_mapping": {
    "identity_columns": ["application_id"],
    "priority_order": ["application_id"],
    "recommended_entity_column": "application_id"
  },
  "field_mapping": {
    "timestamp_column": "event_time",
    "level_column": "severity",
    "message_column": "msg",
    "status_code_column": "http_status",
    "exception_column": "exception"
  }
}
```

This will make Stage A more completely source-independent.

### Stage A Responsibility

Stage A is responsible for:

- source-file processing;
- schema inference and normalization;
- timestamp normalization;
- identity/entity construction;
- severity normalization;
- time-window aggregation;
- numerical feature generation;
- writing per-input Parquet feature datasets.

Stage A should **not** perform LLM-based RCA or remediation.

---

# Stage B — Anomaly Detection

Stage B follows **exactly the same folder-based convention as Stage A**.

It enumerates the per-input feature folders generated by Stage A and processes each folder independently.

Example input:

```text
s3://loghawk-data/features/2026-09-28/
    |
    +-- container_logs/
    |     +-- *.parquet
    |
    +-- loghawk_sample_logs/
          +-- *.parquet
```

Stage B produces:

```text
s3://loghawk-data/anomalies/2026-09-28/
    |
    +-- container_logs/
    |     +-- isolation_forest_results.parquet
    |
    +-- loghawk_sample_logs/
          +-- isolation_forest_results.parquet
```

### Per-input Stage B Contract

For every Stage A input folder:

```text
features/<inputfilename>/
        |
        v
Stage B
        |
        +--> optional identity mapping validation
        |
        +--> clean features
        |
        +--> select normal baseline
        |
        +--> train Isolation Forest
        |
        +--> score every feature window
        |
        +--> calculate severity
        |
        +--> generate explanation
        |
        v
anomalies/<inputfilename>/isolation_forest_results.parquet
```

### Identity Mapping in Stage B

Stage B can locate the corresponding mapping:

```text
s3://loghawk-data/identitymapping/2026-09-28/
    identitymapping_<inputfilename>.json
```

The mapping is optional for Stage B.

Stage A has already created `entity_id`, so Stage B does not reconstruct identity.

When a mapping exists, Stage B can load it and validate that the recommended entity column is consistent with the Stage A feature dataset.

This keeps identity construction in Stage A and prevents duplicate identity logic.

### Isolation Forest

The initial ML detector is **scikit-learn Isolation Forest**.

For each input feature dataset Stage B:

1. loads the complete Parquet feature dataset;
2. cleans numerical features;
3. sorts by timestamp;
4. selects the earliest portion as the normal baseline;
5. scales the ML features with `StandardScaler`;
6. trains Isolation Forest;
7. calculates anomaly scores;
8. marks anomalous feature windows;
9. assigns operational severity;
10. generates a human-readable reason;
11. writes the anomaly result for that input.

### Stage B Feature Columns

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

### Stage B Metadata

Stage B preserves generic entity and identity metadata where available:

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

### Stage B Output

Each input file gets its own anomaly dataset:

```text
s3://loghawk-data/anomalies/2026-09-28/<inputfilename>/
    isolation_forest_results.parquet
```

The anomaly result contains fields such as:

```text
timestamp
entity_id
anomaly_score
is_anomaly
severity
reason
```

plus the Stage A feature and identity metadata.

### Per-input Model Artifacts

The current folder-based implementation also keeps Isolation Forest artifacts separate per input dataset:

```text
s3://loghawk-data/models/2026-09-28/
    isolation_forest_container_logs.joblib
    isolation_scaler_container_logs.joblib

    isolation_forest_loghawk_sample_logs.joblib
    isolation_scaler_loghawk_sample_logs.joblib
```

This prevents one input dataset from overwriting another dataset's model.

### Stage B Responsibility

Stage B is responsible for:

- reading per-input feature datasets;
- baseline selection;
- statistical/baseline detection;
- Isolation Forest;
- anomaly scoring;
- anomaly flags;
- severity;
- evidence/reason generation;
- persisting per-input anomaly results.

Stage B remains independent from the LLM layer.

---

# Stage A → Stage B Stitching

Stage A and Stage B remain separate processing components.

Temporal orchestrates them using the durable S3-compatible data contract.

```text
                         Temporal
                            |
                            v
                    Identity Mapping
                         Activity
                            |
                            v
                    Raw File Mappings
                            |
                            v
                    Stage A Activity
                            |
                            v
                  +------------------+
                  | Per-input        |
                  | Feature Folders  |
                  +------------------+
                            |
                            v
                    Stage B Activity
                            |
                            v
                  +------------------+
                  | Per-input        |
                  | Anomaly Folders  |
                  +------------------+
                            |
                            v
                         Stage C
```

The data contract is:

```text
raw/<inputfilename>
        |
        v
identitymapping/identitymapping_<inputfilename>.json
        |
        v
features/<inputfilename>/
        |
        v
anomalies/<inputfilename>/
        |
        v
incidents/
```

Temporal does not replace PySpark or scikit-learn.

Instead:

```text
Temporal
   |
   +--> Identity Mapping
   |
   +--> waits for all mappings
   |
   +--> Stage A
   |
   +--> waits for Stage A completion
   |
   +--> Stage B
   |
   +--> waits for Stage B completion
   |
   +--> Stage C
```

This provides a clean separation between **workflow orchestration** and **data-processing logic**.

---

# Temporal Orchestration

Temporal is the primary workflow orchestration layer for the LogHawk core AIOps pipeline.

The current workflow evolves around the folder-based processing model:

```text
                    LogHawk Pipeline
                           |
                           v
              +-------------------------+
              | Identity Mapping        |
              | Activity                |
              +------------+------------+
                           |
                           v
              All mappings successful?
                           |
                           v
              +-------------------------+
              | Stage A                 |
              | Folder Feature          |
              | Engineering Activity    |
              +------------+------------+
                           |
                           v
              features/<inputfilename>/
                           |
                           v
              +-------------------------+
              | Stage B                 |
              | Folder Anomaly          |
              | Detection Activity      |
              +------------+------------+
                           |
                           v
              anomalies/<inputfilename>/
                           |
                           v
              +-------------------------+
              | Stage C                 |
              | Incident Correlation    |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Stage D                 |
              | AI / RAG / RCA           |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Policy / Approval       |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Stage E                 |
              | Remediation / Verify    |
              +-------------------------+
```

## Current Temporal Activity Model

The intended activity structure is:

```python
@activity.defn
async def run_identity_mapping(raw_folder):
    ...

@activity.defn
async def run_stage_a(raw_folder, feature_folder):
    ...

@activity.defn
async def run_stage_b(feature_folder, anomaly_folder):
    ...
```

The workflow coordinates them:

```python
@workflow.defn
class LogHawkPipeline:

    @workflow.run
    async def run(
        self,
        raw_folder,
        feature_folder,
        anomaly_folder,
    ):

        await workflow.execute_activity(
            run_identity_mapping,
            args=[raw_folder],
            ...
        )

        feature_path = await workflow.execute_activity(
            run_stage_a,
            args=[raw_folder, feature_folder],
            ...
        )

        anomaly_path = await workflow.execute_activity(
            run_stage_b,
            args=[feature_path, anomaly_folder],
            ...
        )

        return anomaly_path
```

The exact workflow implementation will evolve as LogHawk moves from local development to distributed execution.

### Temporal Reliability

Activities should use:

- retries;
- timeouts;
- idempotent processing where practical;
- clear failure propagation;
- durable workflow state.

Identity Mapping must complete successfully before Stage A starts.

Stage A must complete before Stage B starts.

Stage B must complete before Stage C starts.

---

# Why Temporal Instead of n8n for Stage A → Stage B?

LogHawk deliberately separates the responsibilities of **Temporal** and **n8n**.

## Temporal

Temporal is used for the core, durable AIOps execution pipeline.

It is appropriate for:

-   long-running workflows;
-   durable workflow state;
-   retries;
-   timeouts;
-   failure recovery;
-   human approval/wait states;
-   verification;
-   compensation/rollback;
-   escalation;
-   auditability;
-   remediation workflows.

The Stage A → Stage B pipeline therefore belongs primarily to Temporal.

```text
Temporal
   |
   +--> Stage A
   |
   +--> Stage B
   |
   +--> Stage C
   |
   +--> Stage D
   |
   +--> Remediation
   |
   +--> Verification
```

## n8n

n8n is used for lightweight automation and external integrations.

Examples:

```text
Scheduled ingestion
       |
       v
      n8n
       |
       +--> Webhook
       +--> API
       +--> Notification
       +--> Knowledge refresh
       +--> Slack
       +--> Email
       +--> Jira
```

n8n can also trigger a LogHawk Temporal workflow.

For example:

```text
             n8n
              |
       Scheduled trigger
              |
              v
       LogHawk API
              |
              v
          Temporal
              |
              v
       AIOps Workflow
```

Therefore:

> **n8n triggers and integrates; Temporal orchestrates and guarantees the core workflow.**

---

# RustFS S3 / MINIO S3 / AWS S3 as the Data Boundary

RustFS provides the local S3-compatible object-storage layer for the current development environment.

The object-storage boundary allows the processing stages to remain loosely coupled.

```text
                         S3-Compatible Storage
                         RustFS / MINIO / AWS
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
             raw            features          anomalies
              |                 |                 |
              v                 v                 v
        Identity Map         Stage A           Stage B
                                                   |
                                                   v
                                               Stage C/D
```

### Current folder convention

For a processing date such as `2026-09-28`:

```text
s3://loghawk-data/
    |
    +-- raw/
    |     +-- 2026-09-28/
    |           +-- container_logs.json
    |           +-- loghawk_sample_logs.json
    |
    +-- identitymapping/
    |     +-- 2026-09-28/
    |           +-- identitymapping_container_logs.json
    |           +-- identitymapping_loghawk_sample_logs.json
    |
    +-- features/
    |     +-- 2026-09-28/
    |           +-- container_logs/
    |           |     +-- *.parquet
    |           |
    |           +-- loghawk_sample_logs/
    |                 +-- *.parquet
    |
    +-- anomalies/
    |     +-- 2026-09-28/
    |           +-- container_logs/
    |           |     +-- isolation_forest_results.parquet
    |           |
    |           +-- loghawk_sample_logs/
    |                 +-- isolation_forest_results.parquet
    |
    +-- incidents/
    |     +-- 2026-09-28/
    |
    +-- models/
          +-- 2026-09-28/
                +-- isolation_forest_<inputfilename>.joblib
                +-- isolation_scaler_<inputfilename>.joblib
```

### Stage-to-stage contracts

```text
Stage 0 / Ingestion
    raw/<date>/<inputfilename>
        |
        v
Identity Mapping
    identitymapping/<date>/identitymapping_<inputfilename>.json
        |
        v
Stage A
    features/<date>/<inputfilename>/
        |
        v
Stage B
    anomalies/<date>/<inputfilename>/
        |
        v
Stage C
    incidents/<date>/
        |
        v
Stage D
    RCA / evidence
        |
        v
Stage E
    remediation / verification
```

### S3 URI convention

Python filesystem operations use:

```text
s3://
```

Spark/Hadoop operations use:

```text
s3a://
```

This allows RustFS to remain the common object-storage boundary while respecting the filesystem implementation used by each processing engine.

---

# End-to-End AIOps Flow

```text
Data Sources
 Logs | Metrics | Traces | Kubernetes | AWS
                |
                v
       Ingestion / Collection
          n8n / OTel / APIs
                |
                v
            RustFS S3 / MINIO S3 / AWS S3
                |
                v
       +--------------------+
       | Stage A            |
       | Feature Engineering|
       | PySpark             |
       +---------+----------+
                 |
                 v
          Feature Parquet
                 |
                 v
       +--------------------+
       | Stage B            |
       | Anomaly Detection  |
       | Isolation Forest   |
       | Statistical Models |
       +---------+----------+
                 |
                 v
          Anomaly Results
                 |
                 v
       +--------------------+
       | Stage C            |
       | Incident           |
       | Correlation        |
       +---------+----------+
                 |
                 v
       +--------------------+
       | Stage D            |
       | RAG / AI RCA       |
       | LiteLLM             |
       +---------+----------+
                 |
                 v
       +--------------------+
       | Policy / Approval  |
       +---------+----------+
                 |
                 v
       +--------------------+
       | Stage E            |
       | Temporal           |
       | Remediation        |
       +---------+----------+
                 |
                 v
             Verification
              /         \
         Success        Failure
            |              |
         Resolve        Rollback
                           |
                        Escalate
```

---

# System Responsibilities

-   **RustFS S3 / MINIO S3 / AWS S3** — local S3-compatible or AWS cloud S3 object storage and durable data boundary between processing stages.
-   **LanceDB** — RAG retrieval store for embeddings, searchable knowledge chunks, incident evidence and associated metadata.
-   **DuckDB** — structured analytical storage and conversation history, including chat sessions, messages, agent/tool history and incident/history records.
-   **PySpark** — large-scale ingestion, parsing, normalization, aggregation and feature engineering.
-   **scikit-learn Isolation Forest** — initial machine-learning anomaly detection.
-   **Statistical/EWMA detectors** — explainable baseline and time-series anomaly detection.
-   **Temporal** — durable orchestration of the core AIOps workflow and remediation.
-   **n8n** — scheduling, ingestion triggers, knowledge refreshes, notifications and lightweight integrations.
-   **LiteLLM** — unified LLM gateway/router for local and cloud inference.
-   **Ollama + Gemma 2 (`gemma2:latest`)** — local/private LLM inference for development, offline use and privacy-sensitive workloads.
-   **Amazon Bedrock** — managed enterprise GenAI for production/cloud RCA, RAG and agent reasoning.
-   **Kubernetes / AWS / Terraform** — remediation targets.

---

# LLM Gateway and Model Architecture

LogHawk uses **LiteLLM as the model gateway** so the application does not need to be tightly coupled to a single LLM provider.

LiteLLM provides a unified interface for multiple model providers and can provide routing, retries/fallbacks, authentication hooks, logging and cost tracking.

The primary local development path is **Ollama running Gemma 2 (`gemma2:latest`)**.

The cloud/enterprise path is **Amazon Bedrock**.

```text
                         LogHawk AIOps
                              |
                              v
                    +-------------------+
                    |    LLM Gateway    |
                    |      LiteLLM      |
                    +---------+---------+
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
        +------------------+     +----------------------+
        | Local / Private  |     | Cloud / Enterprise   |
        |                  |     |                      |
        | Ollama           |     | Amazon Bedrock       |
        | Gemma 2          |     | Claude / Nova /      |
        | gemma2:latest    |     | other supported      |
        +------------------+     | foundation models   |
                  |               +----------------------+
                  v                       |
             Local inference              |
             Windows / GPU                |
                  |                       |
                  +-----------+-----------+
                              |
                              v
                    Structured AI response
                              |
                              v
                    LogHawk RCA / Agent
```

## Local Mode

```text
LogHawk
   |
   v
LiteLLM
   |
   v
Ollama
   |
   v
Gemma 2
(gemma2:latest)
```

Local mode is intended for development, privacy-sensitive workloads, experimentation and environments where cloud inference is undesirable.

## AWS Mode

```text
LogHawk
   |
   v
LiteLLM
   |
   v
Amazon Bedrock
   |
   +--> Enterprise foundation model
   |
   +--> Managed inference
   |
   +--> Production RAG / RCA / agents
```

## Model Switching

The application should call **LiteLLM rather than Ollama or Bedrock directly** wherever practical.

```text
                  LogHawk AI request
                         |
                         v
                     LiteLLM
                    /       \
                   /         \
          local profile     cloud profile
               |                  |
            Ollama             Bedrock
               |                  |
           Gemma 2          enterprise model
```

This allows the same RCA/agent application code to use local Gemma 2 during development and a Bedrock-hosted model for enterprise/cloud deployment.

The model-selection policy can later support:

-   local-first inference;
-   cloud fallback;
-   task-specific model routing;
-   cost-aware routing;
-   privacy-aware routing;
-   retry/fallback between model deployments.

---

# Provider-Neutral Application Contract

LogHawk's RCA and agent layers should depend on the **LiteLLM endpoint/model alias**, rather than provider-specific SDK calls.

```text
LogHawk application
        |
        | OpenAI-compatible request
        v
     LiteLLM
        |
   +----+----+
   |         |
   v         v
Ollama     Bedrock
Gemma 2    managed model
```

The model/provider configuration is therefore infrastructure configuration rather than application business logic.

---

# Reference Architecture

```text
                              LOGHAWK
                                 |
        +------------------------+------------------------+
        |                        |                        |
        v                        v                        v
 Observability               Security                Knowledge
Logs/Metrics/Traces       NIST/CVE/ATT&CK          Runbooks/History
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                         Ingestion / Collection
                           n8n / OpenTelemetry
                                 |
                                 v
                          RustFS S3 / MINIO S3 / AWS S3
                                 |
                                 v
                      +---------------------+
                      | Stage A              |
                      | PySpark              |
                      | Feature Engineering  |
                      +----------+-----------+
                                 |
                                 v
                          Feature Parquet
                                 |
                                 v
                      +---------------------+
                      | Stage B              |
                      | Isolation Forest     |
                      | Statistical Detection|
                      +----------+-----------+
                                 |
                                 v
                          Anomaly Results
                                 |
                                 v
                      +---------------------+
                      | Stage C              |
                      | Incident Correlation |
                      +----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
                 DuckDB                    LanceDB
          history/analytics          RAG/vector retrieval
                    |                         |
                    +------------+------------+
                                 |
                                 v
                        AIOps Intelligence
                    Detection / Correlation / RCA
                                 |
                         +-------+-------+
                         |               |
                         v               v
                      LiteLLM            n8n
                   LLM Gateway     external automation
                         |
                +--------+--------+
                |                 |
                v                 v
             Ollama          Amazon Bedrock
           Gemma 2           Enterprise GenAI
         gemma2:latest          cloud
                |                 |
                +--------+--------+
                         |
                         v
                    Policy Gate
                         |
                         v
                      Temporal
                 Durable Remediation
                         |
                  +------+------+
                  |      |      |
                  v      v      v
                 AWS     K8s   Terraform
                  \      |      /
                   +-----+-----+
                         |
                         v
                    Verification
                         |
                         v
                   Incident History
                         |
                         +------> DuckDB / Parquet
                         |
                         +------> LanceDB evidence / RAG
```

---

# Anomaly Detection Pipeline

Raw log text is not sent directly into a machine-learning anomaly detector.

LogHawk first normalizes and aggregates events into service/time-window feature vectors.

```text
Raw Logs
   |
   v
Parse + Normalize
   |
   v
1-minute aggregation per service
   |
   v
+----------------------------------+
| total_log_count                  |
| error_count / error_rate         |
| warning_count                    |
| HTTP 4xx / 5xx / 5xx_rate       |
| timeout_count                    |
| connection_error_count           |
| authentication_failure_count     |
| unique_exception_count           |
| unique_error_message_count       |
+----------------+-----------------+
                 |
                 v
     +-----------+-----------+
     |                       |
     v                       v
Rolling/EWMA/Statistical   Isolation Forest
     |                       |
     +-----------+-----------+
                 v
          Anomaly Score
                 |
                 v
       Evidence + Reason
```

## Initial Detector Types

1.  Volume anomaly
2.  Error-rate anomaly
3.  HTTP 4xx/5xx anomaly
4.  Timeout anomaly
5.  Connection-error anomaly
6.  Authentication-failure anomaly
7.  Novel-event anomaly
8.  Isolation Forest multivariate anomaly

---

# Incident Correlation Flow

Stage C consumes the anomaly results produced by Stage B.

```text
Anomaly A ---+
Anomaly B ---+--> same service/time/dependency? --> Correlation Engine
Anomaly C ---+                                      |
Anomaly D ---+                         +------------+------------+
                                       |                         |
                                    Related                  Unrelated
                                       |                         |
                                       v                         v
                                  One Incident             Separate Incident
                                       |
                                       v
                                Severity + Summary
```

Suggested lifecycle:

```text
OPEN
  |
  v
INVESTIGATING
  |
  v
REMEDIATING
  |
  v
VERIFYING
  |
  +------> RESOLVED
  |
  +------> ROLLED_BACK
  |
  +------> ESCALATED
```

---

# AI Root-Cause Analysis Flow

```text
Incident
  |
  +--> anomaly features
  +--> relevant logs
  +--> metrics/traces
  +--> similar incidents
  +--> runbooks
  +--> service/change metadata
  +--> security knowledge
             |
             v
        Retrieval / RAG
             |
             v
        LiteLLM
             |
       +-----+-----+
       |           |
       v           v
    Ollama      Bedrock
       |           |
       +-----+-----+
             |
             v
       Structured RCA
       /           \
      v             v
Root-cause       Recommended
hypothesis       remediation
```

RCA should distinguish:

-   **observed evidence**;
-   **likely explanation**;
-   **uncertainty**;
-   **alternative hypotheses**;
-   **recommended action**.

The AI should not directly execute high-impact infrastructure actions without the appropriate policy and workflow controls.

---

# Autonomous Remediation Flow

```text
Incident
   |
   v
AI recommends action
   |
   v
Policy / Risk Evaluation
   |
   +--> Auto-approved --------+
   |                          |
   +--> Human approval ------>+
                              v
                       Temporal Workflow
                              |
                    +---------+---------+
                    |         |         |
                 Pre-check  Action    Audit
                              |
                    Restart / Scale /
                    Rollback / Config
                              |
                              v
                         Verification
                         /           \
                    Success          Failure
                       |                |
                    Resolve          Rollback
                                         |
                                      Escalate
```

Example controlled tools can include:

```text
get_pod_status
get_pod_logs
restart_pod
scale_deployment
rollback_deployment
get_service_health
get_cloud_metrics
inspect_cloud_resource
```

AI recommends and explains.

Temporal executes deterministic workflows.

Verification determines whether the operation succeeded.

---

# Data Contracts Between Stages

A major design principle is that every stage should have a well-defined input/output contract.

```text
Stage A

raw/
   |
   v
features/

Stage B

features/
   |
   v
anomalies/

Stage C

anomalies/
   |
   v
incidents/

Stage D

incidents/
   +
logs/metrics/traces
   +
RAG knowledge
   |
   v
RCA

Stage E

RCA
   +
policy
   |
   v
remediation
```

This allows every stage to be:

-   independently developed;
-   independently tested;
-   independently executed;
-   retried independently;
-   replaced without redesigning the entire system.

---

# n8n vs Temporal

| Capability | n8n | Temporal |
|---|---|---|
| Scheduled ingestion | Yes | Possible |
| Webhooks | Yes | Possible |
| SaaS integrations | Strong | Not primary purpose |
| Notifications | Strong | Possible |
| Knowledge refresh | Yes | Possible |
| Lightweight automation | Strong | Possible |
| Long-running workflows | Limited compared with Temporal | Strong |
| Durable workflow state | Not its primary role | Strong |
| Workflow retries | Yes | Strong |
| Timeout management | Yes | Strong |
| Human approval/wait states | Possible | Strong |
| Compensation/rollback workflows | Possible | Strong |
| Mission-critical remediation | Not primary role | Strong |
| Kubernetes remediation orchestration | Possible | Strong |
| Auditability of workflow state | Good | Strong |
| Core LogHawk AIOps pipeline | Supporting role | Primary orchestration layer |

### Architectural Rule

```text
n8n
 |
 +--> Trigger
 +--> Integrate
 +--> Notify
 +--> Schedule
 +--> Refresh

Temporal
 |
 +--> Orchestrate
 +--> Retry
 +--> Wait
 +--> Approve
 +--> Execute
 +--> Verify
 +--> Rollback
 +--> Escalate
```

---

# Technology Responsibilities

| Component | Responsibility |
|---|---|
| Elasticsearch / Splunk / Files / JSON / CSV | Log sources |
| OpenTelemetry | Future logs/metrics/traces integration |
| RustFS S3 / MINIO S3 / AWS S3 | Local S3-compatible or AWS cloud S3 object storage and stage-to-stage data boundary |
| PySpark | Big-data ingestion, aggregation and feature engineering |
| DuckDB / Parquet | Structured analytics, conversation history and operational history storage |
| LanceDB | RAG retrieval: embeddings, searchable text, metadata and hybrid vector/keyword search |
| Statistical/EWMA detectors | Explainable baseline detection |
| scikit-learn Isolation Forest | Initial ML anomaly detection on aggregated features |
| Temporal | Durable core workflow orchestration and remediation |
| LiteLLM | Unified LLM gateway/router for local and cloud models |
| Ollama + Gemma 2 (`gemma2:latest`) | Local/private LLM inference |
| Amazon Bedrock | Managed enterprise GenAI, RAG, RCA and agent reasoning |
| n8n | Scheduling, triggers, notifications and lightweight automation |
| Kubernetes / AWS / Terraform | Remediation targets |

---

# Recommended Project Structure

The project maintains separation between processing logic, identity mapping, workflow orchestration, AI/RAG, and integrations.

```text
loghawk/
|
+-- src/
|   +-- loghawk/
|       |
|       +-- config.py
|       |
|       +-- ingestion/
|       |
|       +-- identity_mapping/
|       |   +-- identity_mapping.py
|       |
|       +-- feature_engineering/
|       |   +-- pyspark_s3_feature_engineering.py
|       |   +-- pyspark_s3_feature_engineering5.py
|       |
|       +-- anomaly_detection/
|       |   +-- scikit_s3_isolation_forest.py
|       |   +-- scikit_s3_isolation_forest_folder.py
|       |   +-- statistical.py
|       |
|       +-- incident/
|       |   +-- correlation.py
|       |
|       +-- rag/
|       |   +-- retrieval.py
|       |
|       +-- ai/
|       |   +-- rca.py
|       |   +-- agent.py
|       |
|       +-- workflows/
|       |   +-- temporal/
|       |       +-- workflows.py
|       |       +-- activities.py
|       |
|       +-- integrations/
|           +-- n8n/
|           +-- kubernetes/
|           +-- aws/
|
+-- data/
|   +-- db/
|
+-- admin/
|
+-- tests/
|
+-- AGENTS.md
+-- README.md
```

The important architectural rule is:

```text
identity_mapping/
        |
        +--> maps source fields to logical identity

feature_engineering/
        |
        +--> contains Stage A processing logic

anomaly_detection/
        |
        +--> contains Stage B processing logic

incident/
        |
        +--> contains Stage C correlation logic

ai/ + rag/
        |
        +--> contains Stage D intelligence/RCA logic

workflows/temporal/
        |
        +--> orchestrates Stage A, B, C, D and E
```

Temporal should **call** the processing components rather than duplicating their implementation.

### Project instructions

`AGENTS.md` contains the project-specific instructions used by Codex, including:

- architecture decisions;
- S3/S3A conventions;
- Stage A and Stage B folder contracts;
- coding guidelines;
- testing expectations;
- mandatory approval before file modifications.

---

# Current Implementation Status

As of the current development iteration, the detection foundation is being validated locally on Windows with RustFS-compatible S3 storage.

Current processing model:

```text
Raw files
   |
   v
Identity Mapping
   |
   v
Per-file mappings
   |
   v
Stage A / PySpark
   |
   v
Per-input feature folders
   |
   v
Stage B / Isolation Forest
   |
   v
Per-input anomaly folders
```

Example:

```text
raw/2026-09-28/container_logs.json
        |
        +--> identitymapping/2026-09-28/
        |       identitymapping_container_logs.json
        |
        +--> features/2026-09-28/container_logs/
        |       *.parquet
        |
        +--> anomalies/2026-09-28/container_logs/
                isolation_forest_results.parquet
```

The same pattern applies independently to every input file discovered in the processing-date folder.

# Roadmap

## Phase 1 — Detection Foundation

-   [x] Define normalized feature contract
-   [x] Implement raw-log storage boundary
-   [x] Implement one-minute aggregation
-   [x] Implement anomaly feature extraction
-   [x] Implement Isolation Forest foundation
-   [x] Persist feature datasets as Parquet
-   [x] Persist anomaly results
-   [ ] Add broader synthetic anomalous-log test data
-   [ ] Add anomaly API/dashboard

## Phase 2 — Folder-Based Stage A → Stage B Pipeline

-   [x] Implement per-file Identity Mapping
-   [x] Implement idempotent identity mapping generation
-   [x] Implement folder-based Stage A
-   [x] Implement per-input Stage A feature folders
-   [x] Implement folder-based Stage B
-   [x] Implement per-input Stage B anomaly folders
-   [x] Preserve generic `entity_id`
-   [x] Keep Python `s3://` and Spark `s3a://` conventions explicit
-   [x] Validate raw → mapping → features → anomalies contract
-   [ ] Add broader field-role mapping for source-independent timestamp/message/status fields
-   [ ] Add schema fingerprint/cache to reduce repeated LLM mapping calls
-   [ ] Add stage-level automated integration tests

## Phase 3 — Temporal Orchestration

-   [x] Introduce Temporal
-   [x] Create LogHawk Temporal Worker
-   [x] Implement Stage A Activity
-   [x] Implement Stage B Activity
-   [ ] Implement Identity Mapping Activity
-   [ ] Update workflow ordering to Identity Mapping → Stage A → Stage B
-   [x] Add retries
-   [x] Add activity timeouts
-   [ ] Add workflow-level failure handling for the complete folder pipeline
-   [ ] Add workflow observability
-   [ ] Add workflow audit metadata

Current target:

```text
Temporal Workflow
       |
       v
Identity Mapping Activity
       |
       v
identitymapping/<date>/
       |
       v
Stage A Activity
       |
       v
features/<date>/<inputfilename>/
       |
       v
Stage B Activity
       |
       v
anomalies/<date>/<inputfilename>/
```

## Phase 4 — Incident Intelligence

-   [ ]  Correlate related anomalies
-   [ ]  Generate incident IDs and severity
-   [ ]  Generate incident summaries
-   [ ]  Implement incident lifecycle
-   [ ]  Store incident history
-   [ ]  Add investigation API/UI
-   [ ]  Add incident correlation to Temporal workflow

## Phase 5 — RAG-Based RCA

-   [ ]  Persist structured incident history in DuckDB / Parquet
-   [ ]  Index incident evidence and summaries in LanceDB
-   [ ]  Index runbooks and architecture documentation in LanceDB
-   [ ]  Retrieve similar incidents
-   [ ]  Build structured RCA context
-   [ ]  Generate evidence-based RCA
-   [ ]  Generate remediation recommendations

## Phase 6 — Local and Cloud LLM Gateway

-   [ ]  Add LiteLLM as the unified LLM gateway
-   [ ]  Configure Ollama as the local inference provider
-   [ ]  Configure Gemma 2 (`gemma2:latest`) for local RCA/development
-   [ ]  Add Amazon Bedrock as the managed cloud provider
-   [ ]  Define local-vs-cloud model routing policy
-   [ ]  Add provider fallback and retry policies
-   [ ]  Add model/request observability
-   [ ]  Keep application-level AI code provider-neutral

## Phase 7 — Bedrock and Agentic AIOps

-   [ ]  Integrate Amazon Bedrock through LiteLLM
-   [ ]  Add Bedrock-powered RCA
-   [ ]  Add operational RAG
-   [ ]  Add controlled AI tools
-   [ ]  Add agent planning/tool selection
-   [ ]  Add policy/approval gates
-   [ ]  Add audit trail

## Phase 8 — Temporal Autonomous Operations

-   [ ]  Define remediation workflows
-   [ ]  Implement pre-checks and actions
-   [ ]  Implement retries/timeouts
-   [ ]  Implement human approval states
-   [ ]  Implement verification
-   [ ]  Implement rollback/compensation
-   [ ]  Implement escalation
-   [ ]  Persist remediation history
-   [ ]  Add remediation audit trail

## Phase 9 — n8n Integrations

-   [ ]  Add scheduled ingestion workflows
-   [ ]  Add webhook integration
-   [ ]  Add notification workflows
-   [ ]  Add Slack/Teams integration
-   [ ]  Add email integration
-   [ ]  Add Jira/service-management integration
-   [ ]  Add knowledge-base refresh workflows
-   [ ]  Allow n8n to trigger LogHawk Temporal workflows

## Phase 10 — Full Observability

-   [ ]  Add metrics ingestion
-   [ ]  Add trace ingestion
-   [ ]  Integrate OpenTelemetry
-   [ ]  Correlate logs + metrics + traces
-   [ ]  Add deployment/change correlation
-   [ ]  Improve service dependency mapping
-   [ ]  Improve incident correlation and RCA

---

# Target End State

```text
                 Production Systems
              AWS / Kubernetes / Apps
                         |
                Logs / Metrics / Traces
                         |
                         v
                    +---------+
                    | LogHawk |
                    +---------+
                         |
                         v
                    RustFS S3 / MINIO S3 / AWS S3
                         |
                         v
                +----------------+
                |     Stage A    |
                |    PySpark     |
                |    Features    |
                +-------+--------+
                        |
                        v
                +----------------+
                |     Stage B    |
                | IsolationForest|
                | + Statistics   |
                +-------+--------+
                        |
                        v
                +----------------+
                |     Stage C    |
                |   Incidents    |
                +-------+--------+
                        |
                        v
                +----------------+
                |     Stage D    |
                |   RAG / RCA    |
                | LiteLLM/LLM    |
                +-------+--------+
                        |
                        v
                  Policy Gate
                        |
                        v
                  +-----------+
                  | Temporal  |
                  +-----+-----+
                        |
                 Remediation
                        |
              +---------+---------+
              |         |         |
             AWS       K8s    Terraform
              |         |         |
              +---------+---------+
                        |
                        v
                    Verify
                        |
                  +-----+-----+
                  |           |
                Success      Failure
                  |           |
               Resolve     Rollback
                              |
                           Escalate
```

The intended end state is an enterprise-oriented **GenAI AIOps platform** that moves from passive log analysis to evidence-based incident diagnosis and controlled autonomous remediation.

The architecture deliberately separates:

```text
                 AI
                  |
          Decide + Explain
                  |
                  v
          Policy / Approval
                  |
                  v
              Temporal
                  |
       Execute + Verify + Rollback
```

This allows AI capabilities to evolve independently from deterministic operational workflows.

---

# Architectural Design Principles

## 1\. Processing and orchestration are separate

PySpark and scikit-learn perform data processing and machine learning.

Temporal orchestrates the execution.

```text
PySpark / ML
     |
     | processing logic
     v
Temporal
     |
     | workflow coordination
     v
Next Stage
```

## 2\. Data stages communicate through durable datasets

```text
raw
 ↓
features
 ↓
anomalies
 ↓
incidents
```

RustFS S3 / MINIO S3 / AWS S3 provides the data boundary.

## 3\. AI does not directly control infrastructure

AI produces:

-   analysis;
-   evidence;
-   hypotheses;
-   recommendations;
-   tool selections.

Temporal and deterministic tools perform controlled execution.

## 4\. Provider independence

The application uses LiteLLM rather than embedding provider-specific LLM logic throughout the codebase.

## 5\. Local-first development

The development architecture supports:

```text
Windows
   |
   +--> Ollama
   |      |
   |   Gemma 2
   |
   +--> RustFS S3 / MINIO S3 / AWS S3
   |
   +--> PySpark
   |
   +--> DuckDB
   |
   +--> LanceDB
   |
   +--> Temporal
```

The same application architecture can later be deployed using:

```text
AWS
 |
 +--> S3
 +--> Bedrock
 +--> EKS/Kubernetes
 +--> Temporal
 +--> managed infrastructure
```

---

# Documentation

### QUICKSTART LOGHAWK AIOPS

```plaintext
First time setup:
firsttime_setup_terminal_1.bat
firsttime_setup_terminal_2.bat
firsttime_setup_terminal_3.bat
```

### `Start Temporal workflow:`

### `firsttime_start_workflow_terminal_1.bat`

### `firsttime_start_workflow_terminal_2.bat`

### `Interactive chat assistant:`

### `firsttime_start_chat_assistant.bat`

### `Web UI to upload Runbooks, Documents, Troubleshooting Guides, etc:`

### `firsttime_start_webUI_doc_upload.bat`

### LLM Provider References

-   LiteLLM: https://docs.litellm.ai/
-   Ollama: https://docs.ollama.com/
-   Amazon Bedrock Runtime: https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html

### Workflow References

-   Temporal: https://temporal.io/
-   n8n: https://n8n.io/

### Data Processing References

-   Apache Spark: https://spark.apache.org/
-   scikit-learn: https://scikit-learn.org/
-   RustFS: https://seaweedfs.com/
-   LanceDB: https://lancedb.com/
-   DuckDB: https://duckdb.org/

---

Copyright
-------------------

Copyright (c) Dilshad Mustafa 2026. All Rights Reserved.

License
-------------

Please refer LICENSE.txt file for complete details on the license and terms and conditions.

About The Author
--------------------

Dilshad Mustafa is the creator and programmer of LogHawk AIOps suite of tools and Scabi framework and Cluster. He is a Senior Software Architect with 22+ years of experience in Software and Information Technology industry. He is experienced in DevOps, Cybersecurity, SRE, SecOps and Software Architecture, Development and Maintenance & Support. He has broad experience across various industry domains, Digital Rights DRM, Banking & Finance, Energy & Utilities, Retail, Pharma, Healthcare.

He completed his B.E. in Computer Science & Engineering from Annamalai University, India and completed his M.Sc. in Communication & Network Systems from Nanyang Technological University, Singapore and PG Program in Cybersecurity from Indian Institute of Technology, IIT Kanpur.
