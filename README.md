Welcome to LogHawk
===================

<center><img src="https://raw.githubusercontent.com/dilshadmustafa/loghawk/main/loghawk_logo.jpg" width="10%"></center>

\

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
> -   **Workflow automation:** use n8n for scheduled ingestion, knowledge refreshes and lightweight integrations.

## LogHawk Architecture

### System Responsibilities

-   **LanceDB** — RAG vector embeddings.
-   **DuckDB** — analytical data and conversation/incident history.
-   **PySpark** — Big Data ingestion, parsing, normalization, aggregation and feature engineering.
-   **n8n** — scheduling, ingestion pipelines and lightweight automation.
-   **Amazon Bedrock** — managed enterprise GenAI for RCA, RAG and agent reasoning.
-   **Temporal** — durable, auditable remediation workflows.

## End-to-End AIOps Flow

```text
Data Sources
 Logs | Metrics | Traces | Kubernetes | AWS
                |
                v
      Ingestion & Normalization
       PySpark / OpenTelemetry
                |
                v
       Feature Engineering
  time windows / rates / errors /
  timeouts / exceptions / status
                |
                v
       Anomaly Detection
 Statistical + EWMA + Isolation Forest
                |
                v
       Incident Correlation
                |
        +-------+-------+
        |               |
        v               v
 Operational RAG   Infra/Observability
 Runbooks          K8s / AWS / OTel
 History
        \               /
         \             /
          v           v
          AI RCA / Agent
        Amazon Bedrock
                |
                v
       Policy / Approval Gate
                |
                v
        Temporal Workflow
                |
                v
   K8s / AWS / Terraform Action
                |
                v
          Verification
          /         \
     Success       Failure
       |              |
    Resolve       Rollback
                      |
                   Escalate
```

## Reference Architecture

```text
                         LOGHAWK
                            |
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
  Observability         Security             Knowledge
 Logs/Metrics/Traces   NIST/CVE/ATT&CK      Runbooks/History
       |                    |                    |
       +--------------------+--------------------+
                            |
                            v
                 Ingestion / Normalization
                       PySpark / OTel
                            |
              +-------------+-------------+
              |                           |
              v                           v
       DuckDB / Parquet                LanceDB
       analytics/history             RAG vectors
              |                           |
              +-------------+-------------+
                            |
                            v
                  AIOps Intelligence
             Detection / Correlation / RCA
                            |
                   +--------+--------+
                   |                 |
                   v                 v
              Bedrock               n8n
          AI/RAG/Agents       schedules/pipelines
                   |
                   v
              Policy Gate
                   |
                   v
               Temporal
          Durable Remediation
                   |
          +--------+--------+
          |        |        |
          v        v        v
         AWS      K8s    Terraform
          \        |        /
           +-------+-------+
                   |
                   v
              Verification
                   |
                   v
            Incident History
                   |
                   +-----> RAG / Future RCA
```

## Anomaly Detection Pipeline

Raw log text is not sent directly into a machine-learning anomaly detector. LogHawk first normalizes and aggregates events into service/time-window feature vectors.

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
| HTTP 4xx / 5xx / 5xx_rate        |
| timeout_count                    |
| connection_error_count           |
| authentication_failure_count    |
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

Initial detector types:

1.  Volume anomaly
2.  Error-rate anomaly
3.  Novel-event anomaly

## Incident Correlation Flow

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

Suggested lifecycle: **OPEN → INVESTIGATING → REMEDIATING → VERIFYING → RESOLVED**, with **ESCALATED** and **ROLLED\_BACK** paths.

## AI Root-Cause Analysis Flow

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
       Amazon Bedrock
             |
             v
      Structured RCA
       /           \
      v             v
Root-cause       Recommended
hypothesis       remediation
```

RCA should distinguish **observed evidence**, **likely explanation**, **uncertainty**, and **recommended action**.

## Autonomous Remediation Flow

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

Example controlled tools can include `get_pod_status`, `get_pod_logs`, `restart_pod`, `scale_deployment`, `rollback_deployment`, `get_service_health`, and cloud metric/resource inspection.

## n8n vs Temporal

**n8n** is intended for scheduled ingestion, knowledge-base refreshes, notifications and lightweight integrations.

**Temporal** is intended for mission-critical remediation: durable state, retries, timeouts, human approval/wait states, verification, rollback/compensation and auditability.

## Technology Responsibilities

| Component | Responsibility |
|---|---|
| Elasticsearch / Splunk / Files / JSON / CSV | Log sources |
| OpenTelemetry | Future logs/metrics/traces integration |
| PySpark | Big-data ingestion, aggregation and feature engineering |
| DuckDB / Parquet | Analytical and history storage |
| LanceDB | RAG vector storage |
| Statistical/EWMA detectors | Explainable baseline detection |
| scikit-learn Isolation Forest | Initial ML anomaly detection on aggregated features |
| Amazon Bedrock | Enterprise GenAI, RAG, RCA and agent reasoning |
| n8n | Scheduling and lightweight automation |
| Temporal | Durable remediation workflows |
| Kubernetes / AWS / Terraform | Remediation targets |

## Roadmap

### Phase 1 — Detection Foundation

-   [ ]  Define normalized event schema
-   [ ]  Parse and normalize logs
-   [ ]  Implement 1-minute aggregation
-   [ ]  Implement anomaly feature extraction
-   [ ]  Build baseline datasets
-   [ ]  Implement statistical/EWMA detectors
-   [ ]  Implement Isolation Forest
-   [ ]  Persist anomaly results
-   [ ]  Add anomaly API/dashboard
-   [ ]  Add synthetic anomalous-log test data

### Phase 2 — Incident Intelligence

-   [ ]  Correlate related anomalies
-   [ ]  Generate incident IDs and severity
-   [ ]  Generate incident summaries
-   [ ]  Implement incident lifecycle
-   [ ]  Store incident history
-   [ ]  Add investigation API/UI

### Phase 3 — RAG-Based RCA

-   [ ]  Index incident history
-   [ ]  Index runbooks and architecture documentation
-   [ ]  Retrieve similar incidents
-   [ ]  Build structured RCA context
-   [ ]  Generate evidence-based RCA
-   [ ]  Generate remediation recommendations

### Phase 4 — Bedrock and Agentic AIOps

-   [ ]  Integrate Amazon Bedrock
-   [ ]  Add Bedrock-powered RCA
-   [ ]  Add operational RAG
-   [ ]  Add controlled AI tools
-   [ ]  Add agent planning/tool selection
-   [ ]  Add policy/approval gates
-   [ ]  Add audit trail

### Phase 5 — Temporal Autonomous Operations

-   [ ]  Introduce Temporal
-   [ ]  Define remediation workflows
-   [ ]  Implement pre-checks and actions
-   [ ]  Implement retries/timeouts
-   [ ]  Implement verification
-   [ ]  Implement rollback/compensation
-   [ ]  Implement escalation
-   [ ]  Persist remediation history

### Phase 6 — Full Observability

-   [ ]  Add metrics ingestion
-   [ ]  Add trace ingestion
-   [ ]  Integrate OpenTelemetry
-   [ ]  Correlate logs + metrics + traces
-   [ ]  Add deployment/change correlation
-   [ ]  Improve service dependency mapping
-   [ ]  Improve incident correlation and RCA

## Target End State

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
     Observe -> Detect -> Correlate
                    |
          Diagnose -> Recommend
                    |
             Remediate -> Verify
                    |
                  Learn
                    |
              +-----+-----+
              |           |
            Human      Automation
              |           |
              +-----+-----+
                    |
              Safer SRE Ops
```

The intended end state is an enterprise-oriented **GenAI AIOps platform** that moves from passive log analysis to evidence-based incident diagnosis and controlled autonomous remediation.

## Architectural Design

## Documentation

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
