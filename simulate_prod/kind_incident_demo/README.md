# Lightweight Kind incident demo for LogHawk

A small order-processing system for learning incident investigation. It runs in the existing Kind cluster; LogHawk, RustFS, Temporal and Ollama remain outside Kubernetes.

## What runs

```text
traffic-generator -> order-service -> inventory-service
                                 \-> payment-service
```

All four pods use one Python 3.12 image with no pip dependencies. Requests are sequential at approximately one order/second, with small minute-to-minute variation. Each server permits at most 16 simultaneous connections. The traffic generator allows one request in flight and caps its rate at five/second.

| Pod | CPU request / limit | RAM request / limit |
|---|---|---|
| order-service | 25m / 200m | 32 / 128 MiB |
| payment-service | 25m / 200m | 32 / 128 MiB |
| inventory-service | 25m / 200m | 32 / 128 MiB |
| traffic-generator | 10m / 100m | 16 / 64 MiB |
| Total | 85m / 700m | 112 / 448 MiB |

Limits are configuration, not measured consumption. Cluster/runtime overhead is additional. The inspected laptop has an i5-1240P, 12 cores/16 logical processors and approximately 16 GB RAM. Available RAM was tight, so avoid running Spark training and a large Ollama inference simultaneously while learning the demo.

The workload creates application errors and delays; it does not intentionally exhaust host RAM/CPU or kill Kubernetes nodes. An application failure does not make `/health` fail. Pods may remain Ready while orders fail: `up == 1` and pod readiness are not proof of business health.

## Prerequisites

Run commands in **WSL Ubuntu** from the repository:

```bash
cd /mnt/c/mywork/loghawk
DEMO="$PWD/simulate_prod/kind_incident_demo"
kubectl --context kind-loghawk get nodes
docker info
kind get clusters
```

The existing Kind cluster is named `loghawk`. Set `KIND_CLUSTER` if yours differs. The script always uses its explicit `kind-<name>` context, rather than whichever context is currently selected.

The deployment requires the existing Prometheus Operator and its ServiceMonitor CRD. `kubernetes.yaml` uses `release: loghawk-monitoring`, matching the inspected Prometheus selector. A differently named installation may require changing that label. No monitoring stack is installed by this demo.

The first build downloads `python:3.12-slim` if absent. Later builds normally reuse it. It does not install PyTorch or LogHawk in the demo pods.

## 1. Deploy and start normal traffic

```bash
bash "$DEMO/run_demo.sh" deploy
bash "$DEMO/run_demo.sh" status
bash "$DEMO/run_demo.sh" logs payment-service
```

The namespace is `loghawk-demo`. The services use internal ClusterIP endpoints only. Service accounts are not mounted, containers run as a non-root user, and their root filesystems are read-only.

Deployment starts the traffic generator automatically. Normal dependency traffic has approximately a 1% transient-error background, so a normal baseline is not completely constant.

Fault controls are intentionally simple lab endpoints, reachable inside the cluster. They are not production authentication/authorization examples. Do not publish the service or `/fault` endpoint to an external network.

## 2. Try an incident

```bash
bash "$DEMO/run_demo.sh" scenario payment-errors 180
```

| Scenario | Behavior | Expected evidence |
|---|---|---|
| `normal` | Clear injected faults | Successful requests, occasional background errors |
| `payment-errors` | Payment returns HTTP 503 | Payment errors; order dependency errors |
| `payment-timeout` | Payment sleeps 1.2 seconds, exceeding the order caller's 0.6-second deadline | Payment latency rises; orders record HTTP 504 and timeout messages |
| `inventory-errors` | Inventory returns HTTP 503 | Inventory and order errors; payment may receive fewer requests |

Fault duration defaults to 180 seconds and is capped at 900 seconds. Fault expiry uses a monotonic clock, and the service automatically returns to normal. Starting another scenario first resets all three services. In-flight delayed requests can take another 1.2 seconds to finish.

```bash
bash "$DEMO/run_demo.sh" scenario payment-timeout 180
bash "$DEMO/run_demo.sh" reset
```

`reset` clears faults; it does **not** stop traffic. To stop or resume traffic:

```bash
bash "$DEMO/run_demo.sh" pause
bash "$DEMO/run_demo.sh" resume
```

Keep `scenarios.json` as the operator's ground truth. For a realistic RCA exercise, first give the assistant only the request logs and metrics, then compare its explanation with the fault you actually injected.

## 3. Observe Prometheus

Use the existing Prometheus port-forward if it is already running. Otherwise, in a separate WSL terminal:

```bash
kubectl --context kind-loghawk -n monitoring port-forward \
  service/loghawk-monitoring-kube-pr-prometheus 9090:9090
```

Open http://localhost:9090 and allow at least two scrape intervals after deployment. Examples:

```promql
sum by (service, status) (rate(demo_requests_total{namespace="loghawk-demo"}[2m]))
```

```promql
sum by (service) (rate(demo_requests_total{namespace="loghawk-demo",status=~"503|504"}[2m]))
```

```promql
histogram_quantile(0.95, sum by (service, le) (rate(demo_request_duration_seconds_bucket{namespace="loghawk-demo"}[2m])))
```

For an operator check, `demo_fault_active` indicates whether the deliberate fault is active. Do not use this metric as hidden ground truth when assessing the assistant's diagnostic ability.

In LogHawk AI Investigation/RCA chat, an example request is:

> Use the Prometheus collector to investigate application errors in namespace loghawk-demo during the last five minutes. Compare order-service, payment-service and inventory-service. Distinguish application errors from pod readiness and do not infer root cause from service names alone.

The built-in instant-query collector accepts valid PromQL such as `rate(...[5m])`, which produces an instant vector from the recent window. Historical range queries require a collector supporting Prometheus `/api/v1/query_range` or an appropriate explicit PromQL timestamp. Do not assume a current metric describes an old incident.

If metrics are absent:

```bash
kubectl --context kind-loghawk -n loghawk-demo get servicemonitor
kubectl --context kind-loghawk -n monitoring get prometheus -o yaml
kubectl --context kind-loghawk -n loghawk-demo get services,endpointslices
```

Check the ServiceMonitor label against the Prometheus `serviceMonitorSelector`, and inspect Prometheus **Status / Targets**. `kubectl top` requires Metrics Server; its absence does not prevent Prometheus scraping.

## 4. Capture a Train baseline

For a first experiment, collect approximately **30–60 minutes** of normal traffic. Shorter samples produce fewer one-minute windows and less reliable ML baselines. Detection is not guaranteed just because a fault was injected.

Keep the following variables in the same WSL shell. Use a NEW batch name for each complete exercise:

```bash
BATCH_S3_URL="s3://loghawk-data/kind-demo-$(date -u +%Y%m%dT%H%M%SZ)/"
bash "$DEMO/run_demo.sh" reset
TRAIN_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'Batch: %s\nTrain start: %s\n' "$BATCH_S3_URL" "$TRAIN_START"
```

After the baseline period:

```bash
TRAIN_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
```

Activate the existing LogHawk **venv312** environment for S3 uploads. For the Conda-based WSL setup:

```bash
conda activate venv312
python "$DEMO/export_evidence.py" \
  --phase train --start "$TRAIN_START" --end "$TRAIN_END" \
  --upload --batch-s3-url "$BATCH_S3_URL"
```

If your environment uses a different activation mechanism, activate the environment that has `boto3` and `python-dotenv`. Do not run a Windows virtual environment from WSL.

Without `--upload`, the exporter only saves local files under `data/kind-demo/<export-id>/`. Timestamps must include a timezone; windows are start-inclusive/end-exclusive and cannot extend into the future. Maximum window: two hours. Per-service export is capped at 8 MiB and 10,000 records.

The exporter reads current pod logs. Export before deleting/restarting pods; Kubernetes log rotation can discard older records. It refuses restarted/replaced pods and an exhausted byte limit, and records first/last observed timestamps for you to review. It cannot reconstruct already rotated logs.

## 5. Capture a Detect window

Start with about two minutes of normal traffic, inject a three-minute fault, then allow several minutes of recovery. A ten-minute window is a useful starting point.

```bash
RAW_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Allow about two minutes of normal traffic, then:
bash "$DEMO/run_demo.sh" scenario payment-errors 180
# After fault expiry and several minutes of recovery:
RAW_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python "$DEMO/export_evidence.py" \
  --phase raw --start "$RAW_START" --end "$RAW_END" \
  --upload --batch-s3-url "$BATCH_S3_URL"
```

Comments above describe real waiting periods; running the commands immediately back-to-back does not create a ten-minute sample. The exporter prints the actual observed log range and row count.

All three services should have requests somewhere in the selected window. Inventory errors can stop orders before they call payment, so include normal/recovery traffic rather than exporting only the inventory outage.

S3 layout:

```text
<batch>/train/order-service.jsonl
<batch>/train/payment-service.jsonl
<batch>/train/inventory-service.jsonl
<batch>/raw/order-service.jsonl
<batch>/raw/payment-service.jsonl
<batch>/raw/inventory-service.jsonl
<batch>/evidence/train-export-manifest.json
<batch>/evidence/raw-export-manifest.json
```

The exporter does not upload `.env` or credentials. It reads the repository `.env` by default; existing process environment values override the file. For WSL, ensure `LH_S3_ENDPOINT` is reachable from WSL. For example, explicitly override the endpoint in the shell if RustFS is reachable only through a Windows host address.

Each service's filename is stable between Train and Detect so pipeline grouping is consistent. Files are directly under `train/` and `raw/`, matching the current non-recursive internal-S3 enumeration.

**Existing objects are not overwritten.** Matching content hashes are skipped; different content at the same key is rejected. Conditional S3 writes protect against a concurrent writer. If the object store rejects conditional PUT, the export fails rather than silently using an unconditional overwrite. A partial upload can leave some objects present; inspect the printed results before retrying. Keep local evidence until upload is confirmed.

## 6. Run the LogHawk pipeline

In the Web UI, create a separate Config Set for this demo:

1. Train input: `<batch>/train/`.
2. Detect input: `<batch>/raw/`.
3. Processing output: the same dedicated demo bucket/batch for both phases.
4. Ensure the input filename allowlist accepts these `.jsonl` files.
5. Preserve INFO records as well as errors. Use a matching empty filter rule, an empty default or no filter for these filenames. Training only on errors would distort the baseline.
6. Run **Train-only** and wait for success before **Detect-only**.
7. Keep the saved ML model output location consistent between the two pipelines.

Use the single-level batch name shown above (for example, `kind-demo-20261103T100000Z`) so the Web UI batch listing can discover its immediate `train/` and `raw/` folders. If the UI expects a bucket and batch name separately, split `BATCH_S3_URL` accordingly. If selecting the demo folders as External S3 inputs, use your reachable RustFS endpoint and existing credentials.

Logs use the fields understood by Stage A: `timestamp`, `service`, `level`, `message`, `status_code` plus provenance fields. Confirm identity mapping uses the stable `service` field. Do not choose `request_id`, `scenario_id`, timestamp or message as the entity identity. Per-request IDs are for tracing an order, not ML entity grouping.

The simulator and exporter do not fabricate Stage B anomalies or Stage C incidents. Stage C creates correlated incidents from the pipeline's actual anomaly output. No incidents may mean the model/window/settings need inspection; it does not mean the demo should insert a fake ML result.

## 7. Use existing and new incidents in Investigations

Your inspected historical source is:

```text
s3://loghawk-data/somefolder/incidents/correlated_incidents.parquet
```

At inspection it contained:

- `INC-20260928110500-0000`: payment-service, CRITICAL, September 28, 2026.
- `INC-20261008100400-0001`: inventory-service, CRITICAL, October 8, 2026.

To investigate one in the Web UI:

1. Open **AI Investigation → Investigations** and create an investigation.
2. List the S3 incident file, choose **Stage C**, and import it.
3. Preview the records and select the relevant incident row.
4. Create an incident from that row.
5. Import related historical Stage B, Stage A or raw-log files, select the relevant records and attach them to that incident.
6. Open RCA Chat with the investigation.

Those old incidents do not prove a payment timeout or inventory outage. The demo scenarios are hypotheses/reproductions inspired by their service names. Clearly identify new demo evidence as a **current simulation**, and do not treat it as evidence of the historical root cause. The old source timestamps also do not identify their timezone explicitly; establish that before comparing with UTC demo evidence.

For a new live investigation, import the incident file generated in the demo batch instead. Attach its exported JSONL logs as **Raw logs**, Stage A/B files with their corresponding evidence types, and exporter metadata as **Other**. Other evidence can attach directly to the investigation or to an incident.

Signed-in Hawk Console examples:

```text
/investigate Investigate simulated payment errors in the Kind demo
/investigation s3 s3://loghawk-data/kind-demo-YOUR_SESSION/incidents/
```

Use `/help` for selecting S3 objects, importing, previewing, creating incidents and attaching records. `/investigation upload "<local JSONL path>" raw-logs` uploads local evidence. Imported sources must be explicitly attached to the investigation/incident before RCA uses them.

## 8. Stop and clean up

First export the evidence you want to keep:

```bash
bash "$DEMO/run_demo.sh" pause
bash "$DEMO/run_demo.sh" cleanup
```

Cleanup deletes only the labelled `loghawk-demo` namespace, including its pods and their logs. It does not delete the Kind cluster, Prometheus, S3 data, local exports or the Docker image. Retained Prometheus history follows your existing Prometheus retention settings.

The scripts do not start/stop LogHawk, RustFS, Temporal or Ollama. No pipeline or incident import is triggered automatically.

For a code rebuild, export logs, clean up and deploy again. Loading a replacement image with the same tag does not automatically restart existing pods.
