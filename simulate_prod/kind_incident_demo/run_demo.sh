#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KIND_CLUSTER="${KIND_CLUSTER:-loghawk}"
CONTEXT="kind-${KIND_CLUSTER}"
IMAGE="loghawk-incident-demo:local"
NAMESPACE="loghawk-demo"
K=(kubectl --context "$CONTEXT" --request-timeout=30s)
usage() {
  cat <<'HELP'
Usage: bash run_demo.sh <command>
  deploy                          Build, load into Kind, deploy and start normal traffic
  scenario <name> [seconds]       normal, payment-errors, payment-timeout, inventory-errors
  reset                           Reset faults; traffic continues
  pause                           Stop traffic (preserves services/metrics)
  resume                          Resume traffic
  status                          Pods, services and ServiceMonitor
  logs <service>                  Last 50 lines from a service
  cleanup                         Remove ONLY the demo namespace (deletes demo pod logs)
Set KIND_CLUSTER to choose another Kind cluster (default: loghawk).
HELP
}
command="${1:-help}"
if [[ "$command" == help || "$command" == --help ]]; then usage; exit 0; fi
command -v kubectl >/dev/null || { echo "kubectl is required" >&2; exit 1; }
"${K[@]}" get namespace kube-system >/dev/null
case "$command" in
  deploy)
    command -v docker >/dev/null
    command -v kind >/dev/null
    "${K[@]}" get crd servicemonitors.monitoring.coreos.com >/dev/null
    # Refuse to deploy into an unrelated namespace of the same name.
    if "${K[@]}" get namespace "$NAMESPACE" >/dev/null 2>&1; then
      owner="$("${K[@]}" get namespace "$NAMESPACE" -o 'jsonpath={.metadata.labels.app\.kubernetes\.io/part-of}')"
      [[ "$owner" == loghawk-incident-demo ]] || { echo "Namespace exists but is not owned by this demo" >&2; exit 1; }
    fi
    docker build --tag "$IMAGE" --file "$ROOT_DIR/Dockerfile" "$ROOT_DIR"
    kind load docker-image "$IMAGE" --name "$KIND_CLUSTER"
    "${K[@]}" apply -f "$ROOT_DIR/kubernetes.yaml"
    for name in order-service payment-service inventory-service traffic-generator; do
      "${K[@]}" -n "$NAMESPACE" rollout status "deployment/$name" --timeout=120s
    done
    echo "Normal traffic is running. UTC baseline start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Rebuilding an existing deployment does not restart its pods. Use cleanup then deploy for code changes, after exporting logs."
    ;;
  scenario|reset)
    name="${2:-normal}"
    [[ "$command" == reset ]] && name=normal
    args=(control --scenario "$name")
    [[ -z "${3:-}" ]] || args+=(--duration "$3")
    "${K[@]}" -n "$NAMESPACE" exec -i deployment/order-service -- python /app/app.py "${args[@]}" < "$ROOT_DIR/scenarios.json"
    echo "UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ;;
  pause) "${K[@]}" -n "$NAMESPACE" scale deployment/traffic-generator --replicas=0 ;;
  resume) "${K[@]}" -n "$NAMESPACE" scale deployment/traffic-generator --replicas=1 ;;
  status) "${K[@]}" -n "$NAMESPACE" get pods,services,servicemonitors ;;
  logs)
    case "${2:-}" in order-service|inventory-service|payment-service) ;; *) usage; exit 1;; esac
    "${K[@]}" -n "$NAMESPACE" logs "deployment/$2" --tail=50
    ;;
  cleanup)
    owner="$("${K[@]}" get namespace "$NAMESPACE" -o 'jsonpath={.metadata.labels.app\.kubernetes\.io/part-of}')"
    [[ "$owner" == loghawk-incident-demo ]] || { echo "Refusing to delete an unrelated namespace" >&2; exit 1; }
    "${K[@]}" delete namespace "$NAMESPACE"
    ;;
  *) usage; exit 1 ;;
esac
