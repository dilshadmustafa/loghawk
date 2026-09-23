Welcome to LogHawk
===================

<center><img src="https://raw.githubusercontent.com/dilshadmustafa/loghawk/main/loghawk_logo.jpg" width="10%"></center>

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

> - **Multi-source observability:** ingest logs from Elasticsearch, Splunk, files, JSON and CSV, with future support for metrics, traces and OpenTelemetry.
> - **AI-powered log intelligence:** combine deterministic detection, machine learning and GenAI for operational and security analysis.
> - **Detection-first anomaly detection:** detect volume, error-rate, HTTP 4xx/5xx, timeout, connection-error, authentication-failure and novel-event anomalies.
> - **Feature-based detection:** normalize logs, aggregate them into time windows, engineer numerical features, establish baselines and apply statistical/time-series detectors plus Isolation Forest.
> - **Incident correlation:** group related anomalies into coherent incidents instead of producing alert storms.
> - **Security threat detection:** retain NIST, CVE and MITRE ATT&CK knowledge and security-focused RAG.
> - **Operational RAG:** ingest runbooks, incident history, architecture documentation and troubleshooting material in addition to PDF/HTML/Markdown/CSV/JSON sources.
> - **AI-assisted RCA:** generate incident summaries, probable root causes, supporting evidence, uncertainty and recommended actions from retrieved context.
> - **Amazon Bedrock integration:** use managed GenAI for enterprise RAG, RCA, agent reasoning and tool selection.
> - **Agentic AIOps:** allow an AI agent to investigate incidents and recommend actions through controlled tools.
> - **Durable remediation:** use Temporal for stateful workflows, retries, timeouts, approval waits, verification, rollback and escalation.
> - **Big-data processing:** use PySpark for large-scale ingestion, normalization, aggregation and feature engineering.
> - **Workflow automation:** use n8n for scheduled ingestion, knowledge refreshes, notifications and lightweight integrations.
> - **Durable data pipeline:** use SeaweedFS S3-compatible object storage as a local/cloud-neutral data boundary between ingestion, feature engineering, anomaly detection and downstream AI processing.

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

- individual processing stages during development;
- the complete pipeline through Temporal;
- local inference through Ollama;
- cloud inference through Amazon Bedrock;
- lightweight external automation through n8n;
- deterministic remediation through Temporal workflows.

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

# Stage A — Feature Engineering

Stage A converts raw application logs into structured numerical feature vectors suitable for machine-learning and statistical anomaly detection.

The processing flow is:

```text
Raw JSON / Application Logs
            |
            v
      SeaweedFS S3
            |
            v
         PySpark
            |
            +--> Parse
            |
            +--> Normalize
            |
            +--> Aggregate
            |
            +--> Feature Engineering
            |
            v
       Feature Dataset
          Parquet
            |
            v
      SeaweedFS S3
```

Example:

```text
Input

s3a://loghawk-data/raw/year=2026/month=09/day=23/


Output

s3a://loghawk-data/features/year=2026/month=09/day=23/
```

### Example Features

Logs are aggregated into time windows, initially using a one-minute window.

Example features include:

```text
total_log_count
error_count
error_rate
warning_count

http_4xx_count
http_4xx_rate

http_5xx_count
http_5xx_rate

timeout_count
connection_error_count
authentication_failure_count

unique_exception_count
unique_error_message_count
```

The output is a structured feature dataset rather than raw log text.

### Stage A Responsibility

Stage A is responsible for:

- log parsing;
- schema normalization;
- timestamp normalization;
- service identification;
- severity normalization;
- time-window aggregation;
- numerical feature generation;
- baseline preparation;
- writing feature datasets to SeaweedFS/S3.

Stage A should **not** perform LLM-based RCA or remediation.

---

# Stage B — Anomaly Detection

Stage B consumes the feature dataset generated by Stage A.

```text
             Stage A
                |
                v
        Feature Parquet
                |
                v
       +----------------+
       | Stage B        |
       | Anomaly        |
       | Detection      |
       +-------+--------+
               |
       +-------+---------+
       |                 |
       v                 v
 Statistical       Isolation Forest
 Detectors              |
       |                 |
       +--------+--------+
                |
                v
          Anomaly Score
                |
                v
        is_anomaly = true/false
                |
                v
       Evidence / Metadata
                |
                v
          SeaweedFS S3
```

The initial ML detector is **scikit-learn Isolation Forest**.

### Stage B Responsibilities

Stage B is responsible for:

- reading feature datasets;
- applying statistical/baseline detectors;
- applying Isolation Forest;
- generating anomaly scores;
- determining anomaly flags;
- attaching detector metadata;
- persisting anomaly results.

Example output:

```text
s3a://loghawk-data/anomalies/year=2026/month=09/day=23/
```

Example logical result:

```text
timestamp
service
error_rate
timeout_count
http_5xx_rate
anomaly_score
is_anomaly
detector
```

Stage B should remain independent from the LLM layer.

---

# Stage A → Stage B Stitching

The recommended architecture is to **keep Stage A and Stage B as separate processing components** and use Temporal to orchestrate them.

```text
                    Temporal
                       |
                       v
              Stage A Activity
                       |
                       v
                 PySpark Job
                       |
                       v
              Feature Parquet
                       |
                       v
                 SeaweedFS
                       |
                       v
              Stage B Activity
                       |
                       v
             Isolation Forest
                       |
                       v
             Anomaly Parquet
                       |
                       v
                 SeaweedFS
```

Temporal does not replace PySpark or scikit-learn.

Instead:

```text
Temporal
   |
   +--> calls Stage A
   |
   +--> waits for Stage A completion
   |
   +--> calls Stage B
   |
   +--> waits for Stage B completion
   |
   +--> starts Stage C
```

This provides a clean separation between **workflow orchestration** and **data-processing logic**.

---

# Temporal Orchestration

Temporal is the primary workflow orchestration layer for the LogHawk core AIOps pipeline.

A simplified workflow is:

```text
                    LogHawk Workflow
                           |
                           v
                 +-------------------+
                 | Validate Input    |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Stage A Activity  |
                 | PySpark Feature   |
                 | Engineering      |
                 +---------+---------+
                           |
                           v
                 Feature Dataset
                           |
                           v
                 +-------------------+
                 | Stage B Activity  |
                 | Anomaly Detection |
                 +---------+---------+
                           |
                           v
                  Anomaly Dataset
                           |
                           v
                 +-------------------+
                 | Stage C Activity  |
                 | Incident          |
                 | Correlation       |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Stage D Activity  |
                 | AI / RAG / RCA    |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Policy / Approval |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Remediation       |
                 | Workflow          |
                 +---------+---------+
                           |
                           v
                       Verify
```

## Temporal Activity Model

Stage A and Stage B should be implemented as independently testable activities.

Conceptually:

```python
@activity.defn
def run_feature_engineering(input_path, output_path):
    # Execute Stage A PySpark processing
    ...


@activity.defn
def run_anomaly_detection(feature_path, output_path):
    # Execute Stage B Isolation Forest processing
    ...
```

The workflow then coordinates them:

```python
@workflow.defn
class LogHawkPipeline:

    @workflow.run
    async def run(self, input_path):

        feature_path = await workflow.execute_activity(
            run_feature_engineering,
            args=[input_path],
            ...
        )

        anomaly_path = await workflow.execute_activity(
            run_anomaly_detection,
            args=[feature_path],
            ...
        )

        return anomaly_path
```

The exact Temporal implementation will evolve as LogHawk moves from local development to distributed execution.

---

# Why Temporal Instead of n8n for Stage A → Stage B?

LogHawk deliberately separates the responsibilities of **Temporal** and **n8n**.

## Temporal

Temporal is used for the core, durable AIOps execution pipeline.

It is appropriate for:

- long-running workflows;
- durable workflow state;
- retries;
- timeouts;
- failure recovery;
- human approval/wait states;
- verification;
- compensation/rollback;
- escalation;
- auditability;
- remediation workflows.

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

# SeaweedFS as the Data Boundary

SeaweedFS provides the local S3-compatible object-storage layer for the development environment.

The object-storage boundary allows the processing stages to remain loosely coupled.

```text
                    SeaweedFS S3
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
       raw           features         anomalies
        |                |                |
        v                v                v
     Stage A          Stage B          Stage C/D
```

Example layout:

```text
s3a://loghawk-data/
    |
    +-- raw/
    |     +-- year=2026/
    |           +-- month=09/
    |                 +-- day=23/
    |
    +-- features/
    |     +-- year=2026/
    |           +-- month=09/
    |                 +-- day=23/
    |
    +-- anomalies/
          +-- year=2026/
                +-- month=09/
                      +-- day=23/
```

This provides a clear contract:

```text
Stage A:
raw → features

Stage B:
features → anomalies

Stage C:
anomalies → incidents

Stage D:
incidents + evidence → RCA

Stage E:
RCA + policy → remediation
```

This contract also makes each stage independently testable.

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
            SeaweedFS
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

- **SeaweedFS S3** — local S3-compatible object storage and durable data boundary between processing stages.
- **LanceDB** — RAG retrieval store for embeddings, searchable knowledge chunks, incident evidence and associated metadata.
- **DuckDB** — structured analytical storage and conversation history, including chat sessions, messages, agent/tool history and incident/history records.
- **PySpark** — large-scale ingestion, parsing, normalization, aggregation and feature engineering.
- **scikit-learn Isolation Forest** — initial machine-learning anomaly detection.
- **Statistical/EWMA detectors** — explainable baseline and time-series anomaly detection.
- **Temporal** — durable orchestration of the core AIOps workflow and remediation.
- **n8n** — scheduling, ingestion triggers, knowledge refreshes, notifications and lightweight integrations.
- **LiteLLM** — unified LLM gateway/router for local and cloud inference.
- **Ollama + Gemma 2 (`gemma2:latest`)** — local/private LLM inference for development, offline use and privacy-sensitive workloads.
- **Amazon Bedrock** — managed enterprise GenAI for production/cloud RCA, RAG and agent reasoning.
- **Kubernetes / AWS / Terraform** — remediation targets.

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

- local-first inference;
- cloud fallback;
- task-specific model routing;
- cost-aware routing;
- privacy-aware routing;
- retry/fallback between model deployments.

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
                          SeaweedFS S3
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

1. Volume anomaly
2. Error-rate anomaly
3. HTTP 4xx/5xx anomaly
4. Timeout anomaly
5. Connection-error anomaly
6. Authentication-failure anomaly
7. Novel-event anomaly
8. Isolation Forest multivariate anomaly

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

- **observed evidence**;
- **likely explanation**;
- **uncertainty**;
- **alternative hypotheses**;
- **recommended action**.

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

- independently developed;
- independently tested;
- independently executed;
- retried independently;
- replaced without redesigning the entire system.

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
| SeaweedFS S3 | Local S3-compatible object storage and stage-to-stage data boundary |
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

The project should maintain separation between processing logic and workflow orchestration.

```text
loghawk/
|
+-- src/
|   +-- loghawk/
|       |
|       +-- ingestion/
|       |
|       +-- feature_engineering/
|       |   +-- pyspark_s3_feature_engineering.py
|       |
|       +-- anomaly_detection/
|       |   +-- isolation_forest.py
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
+-- README.md
```

The important architectural rule is:

```text
feature_engineering/
        |
        +--> contains Stage A processing logic

anomaly_detection/
        |
        +--> contains Stage B processing logic

workflows/temporal/
        |
        +--> orchestrates Stage A, B, C, D and E
```

Temporal should therefore **call** the processing components rather than duplicating their implementation.

---

# Roadmap

## Phase 1 — Detection Foundation

- [ ] Define normalized event schema
- [ ] Parse and normalize logs
- [ ] Implement SeaweedFS S3 raw-log storage
- [ ] Implement 1-minute aggregation
- [ ] Implement anomaly feature extraction
- [ ] Build baseline datasets
- [ ] Implement statistical/EWMA detectors
- [ ] Implement Isolation Forest
- [ ] Persist feature datasets as Parquet
- [ ] Persist anomaly results
- [ ] Add anomaly API/dashboard
- [ ] Add synthetic anomalous-log test data

## Phase 2 — Stage A → Stage B Pipeline

- [ ] Finalize Stage A PySpark interface
- [ ] Finalize Stage A input/output data contract
- [ ] Finalize Stage B Isolation Forest interface
- [ ] Finalize Stage B input/output data contract
- [ ] Validate `raw → features → anomalies`
- [ ] Add stage-level logging
- [ ] Add stage-level error handling
- [ ] Add stage-level test datasets
- [ ] Validate SeaweedFS S3 data boundaries

## Phase 3 — Temporal Orchestration

- [ ] Introduce Temporal
- [ ] Create LogHawk Temporal Worker
- [ ] Implement Stage A Activity
- [ ] Implement Stage B Activity
- [ ] Create Stage A → Stage B workflow
- [ ] Add retries
- [ ] Add timeouts
- [ ] Add workflow failure handling
- [ ] Add workflow observability
- [ ] Add workflow audit metadata

Initial target:

```text
Temporal Workflow
       |
       v
Stage A Activity
       |
       v
features/
       |
       v
Stage B Activity
       |
       v
anomalies/
```

## Phase 4 — Incident Intelligence

- [ ] Correlate related anomalies
- [ ] Generate incident IDs and severity
- [ ] Generate incident summaries
- [ ] Implement incident lifecycle
- [ ] Store incident history
- [ ] Add investigation API/UI
- [ ] Add incident correlation to Temporal workflow

## Phase 5 — RAG-Based RCA

- [ ] Persist structured incident history in DuckDB / Parquet
- [ ] Index incident evidence and summaries in LanceDB
- [ ] Index runbooks and architecture documentation in LanceDB
- [ ] Retrieve similar incidents
- [ ] Build structured RCA context
- [ ] Generate evidence-based RCA
- [ ] Generate remediation recommendations

## Phase 6 — Local and Cloud LLM Gateway

- [ ] Add LiteLLM as the unified LLM gateway
- [ ] Configure Ollama as the local inference provider
- [ ] Configure Gemma 2 (`gemma2:latest`) for local RCA/development
- [ ] Add Amazon Bedrock as the managed cloud provider
- [ ] Define local-vs-cloud model routing policy
- [ ] Add provider fallback and retry policies
- [ ] Add model/request observability
- [ ] Keep application-level AI code provider-neutral

## Phase 7 — Bedrock and Agentic AIOps

- [ ] Integrate Amazon Bedrock through LiteLLM
- [ ] Add Bedrock-powered RCA
- [ ] Add operational RAG
- [ ] Add controlled AI tools
- [ ] Add agent planning/tool selection
- [ ] Add policy/approval gates
- [ ] Add audit trail

## Phase 8 — Temporal Autonomous Operations

- [ ] Define remediation workflows
- [ ] Implement pre-checks and actions
- [ ] Implement retries/timeouts
- [ ] Implement human approval states
- [ ] Implement verification
- [ ] Implement rollback/compensation
- [ ] Implement escalation
- [ ] Persist remediation history
- [ ] Add remediation audit trail

## Phase 9 — n8n Integrations

- [ ] Add scheduled ingestion workflows
- [ ] Add webhook integration
- [ ] Add notification workflows
- [ ] Add Slack/Teams integration
- [ ] Add email integration
- [ ] Add Jira/service-management integration
- [ ] Add knowledge-base refresh workflows
- [ ] Allow n8n to trigger LogHawk Temporal workflows

## Phase 10 — Full Observability

- [ ] Add metrics ingestion
- [ ] Add trace ingestion
- [ ] Integrate OpenTelemetry
- [ ] Correlate logs + metrics + traces
- [ ] Add deployment/change correlation
- [ ] Improve service dependency mapping
- [ ] Improve incident correlation and RCA

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
                    SeaweedFS
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

## 1. Processing and orchestration are separate

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

## 2. Data stages communicate through durable datasets

```text
raw
 ↓
features
 ↓
anomalies
 ↓
incidents
```

SeaweedFS/S3 provides the data boundary.

## 3. AI does not directly control infrastructure

AI produces:

- analysis;
- evidence;
- hypotheses;
- recommendations;
- tool selections.

Temporal and deterministic tools perform controlled execution.

## 4. Provider independence

The application uses LiteLLM rather than embedding provider-specific LLM logic throughout the codebase.

## 5. Local-first development

The development architecture supports:

```text
Windows
   |
   +--> Ollama
   |      |
   |   Gemma 2
   |
   +--> SeaweedFS
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

### LLM Provider References

- LiteLLM: https://docs.litellm.ai/
- Ollama: https://docs.ollama.com/
- Amazon Bedrock Runtime: https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html

### Workflow References

- Temporal: https://temporal.io/
- n8n: https://n8n.io/

### Data Processing References

- Apache Spark: https://spark.apache.org/
- scikit-learn: https://scikit-learn.org/
- SeaweedFS: https://seaweedfs.com/
- LanceDB: https://lancedb.com/
- DuckDB: https://duckdb.org/

---

Copyright
-------------------

Copyright (c) Dilshad Mustafa 2026. All Rights Reserved.

License
-------------

Please refer LICENSE.txt file for complete details on the license and terms and conditions.

About The Author
--------------------

Dilshad Mustafa is the creator and programmer of LogHawk AIOps suite of tools and Scabi framework and Cluster. He is a Senior Software Architect with 23 years of experience in Software and Information Technology industry. He is experienced in DevOps, Cybersecurity, SRE, SecOps and Software Architecture, Development and Maintenance & Support. He has broad experience across various industry domains, Digital Rights DRM, Banking & Finance, Energy & Utilities, Retail, Pharma, Healthcare.

He completed his B.E. in Computer Science & Engineering from Annamalai University, India and completed his M.Sc. in Communication & Network Systems from Nanyang Technological University, Singapore and PG Program in Cybersecurity from Indian Institute of Technology, IIT Kanpur.