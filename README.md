Welcome to LogHawk
===================

<div align="center">
  <img src="https://raw.githubusercontent.com/dilshadmustafa/loghawk/main/loghawk_logo.jpg" width="50%">
</div>

<div align="center">
  <a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=H4V87SN5M2GG2">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" alt="Donate with PayPal">
  </a>
</div>

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

The current implementation covers Identity Mapping and Stages A–C, with Stage D diagnosis and Stage E remediation still evolving.

---

# Stage A — Identity Mapping and Feature Engineering

The current pipeline separates **training data** from **detection data** beneath one configurable batch folder. The folder name can be a date such as `2026-09-28` or any other single folder name such as `somefolder`.

```text
s3://<bucket>/<batch>/train/    Known-normal logs used to build group models
s3://<bucket>/<batch>/raw/      Logs to be scored for anomalies
```

Identity Mapping runs separately for each phase. Stage A reads all files in a phase, groups related filenames, combines each group, then writes one feature dataset per group.

### Identity Mapping

Identity Mapping is a Temporal activity. It creates one mapping JSON per input file under that phase's mapping tree:

```text
s3://<bucket>/<batch>/identitymapping/<train|raw>/<group>/
    identitymapping_<input-file-stem>.json
```

The mapping stores source details, identity roles, and field roles:

```json
{
  "source": {
    "raw_input_path": "...",
    "column_names": ["..."],
    "sample_row": {},
    "sample_rows": []
  },
  "identity_mapping": {
    "identity_columns": ["..."],
    "priority_order": ["..."],
    "recommended_entity_column": "...",
    "fallback_entity_id": "unknown-entity",
    "reason": "..."
  },
  "field_mapping": {
    "timestamp_column": "timestamp",
    "level_column": "level",
    "message_column": "message",
    "status_code_column": "status_code",
    "exception_column": "exception"
  }
}
```

Mappings use only actual source columns. Existing mappings are controlled by `LH_IDENTITY_MAPPING_SKIP_EXISTING`; `true` reuses an existing mapping and `false` regenerates it. Mapping generation must finish for every file before Stage A starts.

Supported input extensions are `.json`, `.json.gz`, `.jsonl`, `.jsonl.gz`, `.log`, and `.log.gz`. Files whose names begin with `.` or `_` are skipped.

### Source-file grouping

LogHawk removes the supported extension, then strips a trailing number only when it follows `_` or `-`. This groups file batches while preserving names that do not use a separator:

| Input names | Group |
|---|---|
| `elasticsearch_db_1.log`, `elasticsearch_db_2.log` | `elasticsearch_db` |
| `elasticsearch-1.log`, `elasticsearch-2.log` | `elasticsearch` |
| `splunk1.log`, `splunk2.log` | `splunk1`, `splunk2` separately |

Each file keeps its own mapping. Stage A combines the normalized rows of every file in a group before producing that group's feature dataset.

### Stage A output and processing

```text
s3://<bucket>/<batch>/features/train/<group>/*.parquet
s3://<bucket>/<batch>/features/raw/<group>/*.parquet
```

Stage A loads and validates each file's mapping, normalizes the mapped timestamp, level, message, status-code, and exception fields, builds generic `entity_id`, and aggregates by entity and one-minute window. It writes one Parquet dataset per source group and phase.

Feature columns include:

```text
total_log_count, info_count, warning_count, error_count,
error_rate, warning_rate, http_4xx_count, http_5xx_count,
http_5xx_rate, timeout_count, timeout_rate,
connection_error_count, authentication_failure_count,
unique_exception_count, unique_error_message_count
```

Identity priority comes from the mapping; `service` is not assumed to be universal. Missing identity values use `unknown-entity`.

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

For example, Python can enumerate an input folder with `s3://<bucket>/<batch>/raw/`, while Spark reads the file through the equivalent `s3a://<bucket>/<batch>/raw/<file>` URI. Convert S3 paths before passing them to Spark; do not pass `s3://` directly to Spark.

### Stage A processing

Stage A reads every supported file in the selected phase, looks up that file's phase-specific mapping, normalizes mapped fields, builds generic `entity_id`, groups files by normalized source group, then aggregates by entity and one-minute window. The output is one Parquet dataset per group under `features/<phase>/<group>/`.

S3 Select is optional and can reduce input rows before identity mapping and feature engineering. See [S3 Select and environment configuration](#s3-select-and-environment-configuration) for its flags, filter syntax, supported input formats, and row-count logging.

Stage A feature columns include:

```text
total_log_count, info_count, warning_count, error_count,
error_rate, warning_rate, http_4xx_count, http_5xx_count,
http_5xx_rate, timeout_count, timeout_rate,
connection_error_count, authentication_failure_count,
unique_exception_count, unique_error_message_count
```

---

# Stage B — Train and Detect

Stage B has two distinct phases, selected through configuration. Train reads features made from `train/` and saves a separate detector bundle for every configured algorithm and source group. Detect reads features made from `raw/`, loads the matching bundles, scores the windows, and writes anomaly results. Detection data is not used to fit detector models or score calibration.

```text
features/train/<group>/ -> train model -> models/<group>/
features/raw/<group>/   -> load model -> anomalies/raw/<group>/
```

Each source group has a model set, and each selected algorithm has its own detector, scaler, and metadata. With one algorithm, Stage B uses its native anomaly decision and does not save ensemble calibration scores. With multiple algorithms, Stage B calibrates each detector's training scores to percentiles, takes their equal-weight mean, and marks a window anomalous when the combined score meets `LH_ENSEMBLE_THRESHOLD`.

Model artifacts are stored separately by algorithm:

```text
<batch>/models/<group>/model_set.json
<batch>/models/<group>/algorithms/<algorithm-id>/detector/model.joblib  # sklearn/PyOD
<batch>/models/<group>/algorithms/<algorithm-id>/detector/model.tl      # cuML
<batch>/models/<group>/algorithms/<algorithm-id>/scaler/scaler.npz
<batch>/models/<group>/algorithms/<algorithm-id>/metadata.json
<batch>/models/<group>/algorithms/<algorithm-id>/calibration/scores.npz  # ensembles only
```

Set `LH_ANOMALY_ALGORITHMS` to a comma-separated list to choose one or more detectors. Supported IDs and execution devices are:

| Algorithm ID | Backend | Device |
|---|---|---|
| `sklearn-isolationforest` | scikit-learn | CPU |
| `cuml-isolationforest` | cuML | GPU |
| `pyod-isolationforest`, `pyod-copod`, `pyod-ecod`, `pyod-hbos`, `pyod-knn`, `pyod-lof`, `pyod-ocsvm`, `pyod-pca` | PyOD | CPU |
| `pyod-autoencoder` | PyOD/PyTorch | CPU |
| `pyod-autoencoder-gpu` | PyOD/PyTorch | CUDA GPU |

For example:

```ini
LH_ANOMALY_ALGORITHMS=sklearn-isolationforest,pyod-copod
LH_ENSEMBLE_THRESHOLD=0.95
```

The algorithm ID determines its backend and device. When `LH_ANOMALY_ALGORITHMS` is empty, single-detector mode uses `LH_ANOMALY_BACKEND`, `LH_ANOMALY_ALGORITHM`, and `LH_ANOMALY_DEVICE`.

Anomaly output is:

```text
<batch>/anomalies/raw/<group>/anomaly_results.parquet
```

Stage B results preserve timestamp, generic entity metadata, available source metadata, `anomaly_score`, `is_anomaly`, severity, and reason. Ensemble results also include `detector_algorithms` and one normalized score column per selected detector. A deterministic burst rule also marks a window anomalous when either `error_count >= 10` and `error_rate >= 0.5`, or `http_5xx_count >= 10` and `http_5xx_rate >= 0.5`.

Stage C consumes the Detect anomaly outputs and writes correlated incidents to:

```text
<batch>/incidents/correlated_incidents.parquet
```

`LH_CORRELATION_WINDOW_MINUTES` configures the incident correlation window.

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
                  | Grouped Features |
                  | Train and Raw    |
                  +------------------+
                            |
                            v
                    Stage B Activity
                            |
                            v
                  +------------------+
                  | Train models or |
                  | Raw anomalies   |
                  +------------------+
                            |
                            v
                         Stage C
```

The durable contracts are phase-specific:

```text
Train: train/ -> identitymapping/train/ -> features/train/ -> models/<group>/
Detect: raw/  -> identitymapping/raw/   -> features/raw/   -> anomalies/raw/<group>/ -> incidents/
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

The current Temporal workflow uses the folder-based Train/Detect processing model:

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
              features/<phase>/<group>/
                           |
                           v
              +-------------------------+
              | Stage B                 |
              | Folder Anomaly          |
              | Detection Activity      |
              +------------+------------+
                           |
                           v
              anomalies/raw/<group>/
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

The `LogHawkTrainDetectPipeline` workflow uses `LH_TRAIN_PHASE` and `LH_DETECT_PHASE` to select work. When both are enabled, it completes Train before Detect. Train runs identity mapping and Stage A over `train/`, then fits and saves a detector bundle per selected algorithm and source group. Detect runs identity mapping and Stage A over `raw/`, loads each group's model set, scores the feature windows, then runs Stage C. Detect-only requires trained artifacts for every group and selected algorithm.

Run the worker and workflow starter from the repository root:

```powershell
python src\loghawk\workflows\temporal\worker.py
python src\loghawk\workflows\temporal\start_pipeline.py
```

Restart the worker after changing workflow or activity code so the running process loads the new modules. Configuration changes in `.env` take effect when the process that reads them is restarted.

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

`<batch>` is one configurable folder component; it may be a date such as `2026-09-28` or a name such as `somefolder`. Training and detection logs are kept separate. Source groups are derived by removing the supported extension and stripping a trailing number only when preceded by `_` or `-`.

```text
s3://<bucket>/<batch>/
    +-- train/<training log files>
    +-- raw/<logs to detect>
    +-- identitymapping/
    |   +-- train/<group>/identitymapping_<input-stem>.json
    |   +-- raw/<group>/identitymapping_<input-stem>.json
    +-- features/
    |   +-- train/<group>/*.parquet
    |   +-- raw/<group>/*.parquet
    +-- models/<group>/
    |   +-- model_set.json
    |   +-- algorithms/<algorithm-id>/detector/model.joblib  # sklearn/PyOD
    |   +-- algorithms/<algorithm-id>/detector/model.tl      # cuML
    |   +-- algorithms/<algorithm-id>/scaler/scaler.npz
    |   +-- algorithms/<algorithm-id>/metadata.json
    |   +-- algorithms/<algorithm-id>/calibration/scores.npz  # ensembles only
    +-- anomalies/raw/<group>/anomaly_results.parquet
    +-- incidents/correlated_incidents.parquet
```

For example, `elasticsearch_db_1.log` and `elasticsearch_db_2.log` belong to group `elasticsearch_db`; `elasticsearch-1.log` belongs to `elasticsearch`; `splunk1.log` remains a distinct group. Each input file has its own identity mapping, while Stage A combines files in the same group before writing features.

### Stage-to-stage contracts

```text
Train: <batch>/train/ -> identitymapping/train/ -> features/train/ -> models/
Detect: <batch>/raw/  -> identitymapping/raw/   -> features/raw/   -> anomalies/raw/ -> incidents/
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

### S3 Select and environment configuration

Configuration is loaded from the repository-root `.env` file by `src/loghawk/config.py`. These variables control the current Train/Detect and S3 Select behavior:

| Variable | Purpose | Values/default |
|---|---|---|
| `LH_S3_BUCKET` | S3 bucket | Defaults to `loghawk-data` |
| `LH_S3_BATCH_FOLDER` | One batch folder beneath the bucket | Defaults to `2026-09-28`; any single folder name is allowed |
| `LH_S3_ENDPOINT` | RustFS S3 endpoint | Defaults to `http://localhost:9000` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | S3 credentials and region | Set these for the local or deployed environment |
| `LH_TRAIN_PHASE` | Run mapping, Stage A, and model training on `<batch>/train/` | `true` / `false`, default `false` |
| `LH_DETECT_PHASE` | Run mapping, Stage A, detection, and Stage C on `<batch>/raw/` | `true` / `false`, default `false` |
| `LH_IDENTITY_MAPPING_SKIP_EXISTING` | Reuse an existing per-file mapping | `true` / `false`, default `true` |
| `LH_IDENTITY_MAPPING_SAMPLE_STRATEGY` | Mapping sample selection | `reservoir` or `first`, default `reservoir` |
| `LH_IDENTITY_MAPPING_SAMPLE_SIZE` | Rows retained for mapping | Positive integer, default `10` |
| `LH_IDENTITY_MAPPING_SAMPLE_SEED` | Reproducible reservoir sample seed | Integer, default `42` |
| `LH_S3_SELECT_SUPPORTED` | Declare S3 Select supported by the store | `true` / `false`, default `false` |
| `LH_S3_SELECT_USE_TRAIN_PHASE` | Enable S3 Select for Train input | `true` / `false`, default `false` |
| `LH_S3_SELECT_USE_DETECT_PHASE` | Enable S3 Select for Detect input | `true` / `false`, default `false` |
| `LH_S3_SELECT_RECORD_FILTER` | Log levels to include | Default `WARN,ERROR`; `ALL` includes all levels |
| `LH_CORRELATION_WINDOW_MINUTES` | Stage C correlation window | Positive integer, default `5` |

Example `.env` settings (keep real credentials local and do not commit secrets):

```ini
LH_S3_BUCKET=loghawk-data
LH_S3_BATCH_FOLDER=somefolder
LH_TRAIN_PHASE=true
LH_DETECT_PHASE=true
LH_S3_SELECT_SUPPORTED=true
LH_S3_SELECT_USE_TRAIN_PHASE=false
LH_S3_SELECT_USE_DETECT_PHASE=true
LH_S3_SELECT_RECORD_FILTER=WARN,ERROR
LH_IDENTITY_MAPPING_SAMPLE_STRATEGY=reservoir
LH_IDENTITY_MAPPING_SAMPLE_SIZE=10
LH_IDENTITY_MAPPING_SAMPLE_SEED=42
LH_IDENTITY_MAPPING_SKIP_EXISTING=true
LH_CORRELATION_WINDOW_MINUTES=5
```

S3 Select is used only when `LH_S3_SELECT_SUPPORTED=true` and the matching phase use flag is also true, for supported uncompressed JSON/JSONL inputs. The record filter is applied during Identity Mapping and Stage A. `WARN` also matches `WARNING`; `ALL` disables the severity filter. If both phase flags are true, Train runs before Detect. At least one phase flag must be true. Train-only fits models; Detect-only requires saved models. Row counts report records returned by the active S3 Select stream; they do not trigger an extra count read.

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
|       |   +-- identity_mapping5.py
|       |
|       +-- feature_engineering/
|       |   +-- pyspark_s3_feature_engineering7.py
|       |
|       +-- anomaly_detection/
|       |   +-- scikit_s3_isolation_forest5.py
|       |   +-- statistical.py
|       |
|       +-- correlation/
|       |   +-- event_correlation.py
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
|       |       +-- worker.py
|       |       +-- start_pipeline.py
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

Current processing uses a configurable batch folder with separate `train/` and `raw/` inputs. Identity mapping and Stage A run for each enabled phase, grouping related filenames before writing features. Stage B trains per-group artifacts from Train features and reuses them to detect anomalies in Raw features. Stage C correlates detected anomalies. S3 Select can filter eligible JSON/JSONL records before mapping and feature engineering.

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
-   [x] Write grouped Stage A feature datasets per phase
-   [x] Implement folder-based Stage B
-   [x] Write grouped Stage B anomaly datasets
-   [x] Preserve generic `entity_id`
-   [x] Keep Python `s3://` and Spark `s3a://` conventions explicit
-   [x] Validate raw → mapping → features → anomalies contract
-   [x] Add field-role mapping for timestamp/message/status/exception fields
-   [ ] Add schema fingerprint/cache to reduce repeated LLM mapping calls
-   [ ] Add stage-level automated integration tests

## Phase 3 — Temporal Orchestration

-   [x] Introduce Temporal
-   [x] Create LogHawk Temporal Worker
-   [x] Implement Stage A Activity
-   [x] Implement Stage B Activity
-   [x] Implement Identity Mapping Activity
-   [x] Order Identity Mapping → Stage A → Stage B
-   [x] Add Train and Detect phase selection
-   [x] Add retries
-   [x] Add activity timeouts
-   [ ] Add workflow-level failure handling for the complete folder pipeline
-   [ ] Add workflow observability
-   [ ] Add workflow audit metadata

Current phase flow:

```text
Train: train/ -> identitymapping/train/ -> features/train/ -> models/
Detect: raw/  -> identitymapping/raw/   -> features/raw/   -> anomalies/raw/ -> incidents/
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

### Quickstart LogHawk AIOps

Configure the repository-root `.env` first. Set the S3 endpoint, bucket, batch folder, and phase flags as described in [S3 Select and environment configuration](#s3-select-and-environment-configuration). Keep Ollama available with the configured model for Identity Mapping.

The setup scripts install application dependencies and can start RustFS and Temporal, create the bucket, and generate/upload sample data. Their `SKIP_*` flags skip those service/setup actions; they do not skip Python environment or package installation.

#### Windows

Install Python 3.12, Java 17, Docker Desktop, and AWS CLI. Close LogHawk Python processes before rerunning setup so they do not hold files in `venv312`.

Run:

```powershell
.\firsttime_setup.bat
```

The script creates `venv312`, installs PySpark and the CUDA 13.0 PyTorch build, installs `requirements.txt`, and by default starts RustFS and Temporal, creates the `loghawk-data` bucket, and uploads sample Train and Raw logs. CUDA-enabled PyTorch can still run CPU operations when no supported GPU is available; CUDA operations require a compatible NVIDIA GPU and driver.

To rerun setup while RustFS, Temporal, DuckDB, and LanceDB are already set up, run these commands in PowerShell:

```powershell
$env:SKIP_RUSTFS_SETUP = "true"
$env:SKIP_TEMPORAL_SETUP = "true"
$env:SKIP_DUCKDB_SETUP = "true"
$env:SKIP_LANCEDB_SETUP = "true"
.\firsttime_setup.bat
```

`SKIP_RUSTFS_SETUP=true` also skips bucket creation and sample-data generation/upload.

#### Linux and macOS

Install Java 17, Docker, and AWS CLI. `firsttime_setup.sh` installs Miniforge if needed and creates the Python 3.12 Conda environment named `venv312`. Keep Ollama reachable from the environment running LogHawk.

Run:

```sh
bash firsttime_setup.sh
```

On Linux, the script explicitly installs the CUDA 13.0 PyTorch build; on macOS, it installs standard PyTorch. By default it also starts RustFS and Temporal, creates the `loghawk-data` bucket, and uploads sample Train and Raw logs.

To rerun setup while RustFS, Temporal, DuckDB, and LanceDB are already set up, run:

```sh
SKIP_RUSTFS_SETUP=true \
SKIP_TEMPORAL_SETUP=true \
SKIP_DUCKDB_SETUP=true \
SKIP_LANCEDB_SETUP=true \
bash firsttime_setup.sh
```

These flags skip service/setup actions, but the script still creates or updates the Conda environment and installs packages. `SKIP_RUSTFS_SETUP=true` also skips bucket creation and sample-data generation/upload.

#### NVIDIA GPU setup with RAPIDS cuML

Use `firsttime_setup_rapids_cuml.sh` inside Ubuntu on WSL2 or a supported Linux host with NVIDIA GPU access. It checks `nvidia-smi` and Java 17, installs Miniforge if needed, creates or updates the `loghawk-rapids` Conda environment with Python 3.12, cuML, and nvForest, installs the project requirements, verifies GPU access, and configures `LH_ANOMALY_BACKEND=cuml`.

Run from the repository root:

```sh
bash firsttime_setup_rapids_cuml.sh
```

The script uses CUDA 13.2 by default. It also supports `SKIP_RUSTFS_SETUP`, `SKIP_TEMPORAL_SETUP`, `SKIP_DUCKDB_SETUP`, and `SKIP_LANCEDB_SETUP`; setting all four to `true` skips those setup actions and RustFS test-data upload while still installing the RAPIDS environment and Python packages:

```sh
SKIP_RUSTFS_SETUP=true \
SKIP_TEMPORAL_SETUP=true \
SKIP_DUCKDB_SETUP=true \
SKIP_LANCEDB_SETUP=true \
bash firsttime_setup_rapids_cuml.sh
```

In each new terminal, activate the environment with:

```sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate loghawk-rapids
```

The script sets `LH_ANOMALY_BACKEND=cuml`, but a non-empty `LH_ANOMALY_ALGORITHMS` list takes precedence. To select cuML in that mode, include `cuml-isolationforest` in the list. Otherwise, clear the list and configure the single-detector settings for cuML.

With `LH_S3_BATCH_FOLDER=quickstart`, the generated inputs are:

```text
s3://loghawk-data/quickstart/train/elasticsearch-1.log
s3://loghawk-data/quickstart/raw/elasticsearch-1.log
```

#### Start the Temporal workflow

Keep the Temporal server running. In two separate terminals, run the worker first, then start the workflow:

```text
Windows worker:  firsttime_start_workflow_terminal_1.bat
Linux/macOS:     bash firsttime_start_workflow_terminal_1.sh

Windows starter: firsttime_start_workflow_terminal_2.bat
Linux/macOS:     bash firsttime_start_workflow_terminal_2.sh
```

The Temporal UI is available at `http://localhost:8233`.

#### Optional applications

```text
Interactive chat — Windows: firsttime_start_chat_assistant.bat
                   Linux/macOS: bash firsttime_start_chat_assistant.sh

Document upload — Windows: firsttime_start_webUI_doc_upload.bat
                  Linux/macOS: bash firsttime_start_webUI_doc_upload.sh
```

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
