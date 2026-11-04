"""Small, bounded Kubernetes incident lab. Python standard library only."""
import argparse
import collections
import datetime as dt
import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICES = ("order-service", "inventory-service", "payment-service")
SERVICE = os.environ.get("SERVICE_NAME", "order-service")
MAX_FAULT_SECONDS = 900
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
LOCK = threading.Lock()
COUNTS = collections.Counter()
LATENCY = [0] * len(BUCKETS)
LATENCY_SUM = 0.0
FAULT = {"mode": "normal", "until": 0.0, "scenario": "normal", "delay": 1.0}
PRINT_LOCK = threading.Lock()


def emit(**fields):
    record = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
              "service": SERVICE, "pod_name": os.environ.get("POD_NAME", "unknown"),
              "namespace": os.environ.get("POD_NAMESPACE", "loghawk-demo"), **fields}
    with PRINT_LOCK:
        print(json.dumps(record, separators=(",", ":")), flush=True)


def get_fault():
    with LOCK:
        if FAULT["until"] <= time.monotonic():
            return {"mode": "normal", "scenario": "normal", "delay": 0.0}
        return dict(FAULT)


def http_json(url, body=None, timeout=0.6):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read(16384))


def metrics():
    with LOCK:
        counts, buckets, total = dict(COUNTS), list(LATENCY), LATENCY_SUM
    fault = get_fault()
    label = 'service="' + SERVICE + '"'
    lines = ['# TYPE demo_requests_total counter']
    for code in (200, 503, 504):
        lines.append(f'demo_requests_total{{{label},status="{code}"}} {counts.get(code, 0)}')
    lines.append('# TYPE demo_request_duration_seconds histogram')
    for bound, count in zip(BUCKETS, buckets):
        lines.append(f'demo_request_duration_seconds_bucket{{{label},le="{bound}"}} {count}')
    count = sum(counts.values())
    lines += [f'demo_request_duration_seconds_bucket{{{label},le="+Inf"}} {count}',
              f'demo_request_duration_seconds_count{{{label}}} {count}',
              f'demo_request_duration_seconds_sum{{{label}}} {total}',
              '# TYPE demo_fault_active gauge',
              f'demo_fault_active{{{label}}} {int(fault["mode"] != "normal")}']
    return ('\n'.join(lines) + '\n').encode()


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, handler):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, handler)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_):
        pass

    def reply(self, status, value, content_type="application/json"):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # Expected when an injected delay outlasts the caller's deadline.

    def do_POST(self):
        if self.path != "/fault":
            self.reply(404, {"error": "unknown endpoint"}); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid body size")
            value = json.loads(self.rfile.read(length))
            mode = value["mode"]
            duration = int(value.get("duration_seconds", 180))
            delay = float(value.get("delay_seconds", 1.0))
            scenario = str(value.get("scenario", mode))
            if mode not in {"normal", "error", "delay"} or not 1 <= duration <= MAX_FAULT_SECONDS:
                raise ValueError("invalid fault or duration")
            if not 0 <= delay <= 2 or len(scenario) > 100:
                raise ValueError("invalid delay or scenario")
            with LOCK:
                FAULT.update(mode=mode, until=time.monotonic()+duration,
                             delay=delay, scenario=scenario)
            emit(event="fault_change", level="INFO", message="Demo fault configuration changed",
                 scenario_id=scenario, fault_mode=mode, duration_seconds=duration)
            self.reply(200, {"service": SERVICE, "mode": mode, "duration_seconds": duration})
        except (ValueError, TypeError, KeyError, socket.timeout) as exc:
            self.reply(400, {"error": str(exc)})

    def do_GET(self):
        global LATENCY_SUM
        if self.path == "/health":
            self.reply(200, {"status": "ok"}); return
        if self.path == "/metrics":
            self.reply(200, metrics(), "text/plain; version=0.0.4"); return
        if self.path != "/work":
            self.reply(404, {"error": "unknown endpoint"}); return
        started = time.monotonic()
        request_id = self.headers.get("X-Request-ID", "direct")[:100]
        fault = get_fault()
        status, message, cause = 200, "Request completed", None
        if fault["mode"] == "delay":
            time.sleep(fault["delay"])
        if fault["mode"] == "error":
            status, message = 503, "Simulated dependency unavailable"
        elif SERVICE != "order-service":
            # A small normal error background prevents a completely constant baseline.
            if random.random() < 0.01:
                status, message = 503, "Transient dependency connection error"
        if status == 200 and SERVICE == "order-service":
            for dependency in ("inventory-service", "payment-service"):
                try:
                    req = urllib.request.Request(f"http://{dependency}:8080/work",
                                                 headers={"X-Request-ID": request_id})
                    with urllib.request.urlopen(req, timeout=0.6) as response:
                        response.read(16384)
                except urllib.error.HTTPError as exc:
                    exc.close()
                    status, message, cause = 503, "Dependency returned an error", dependency
                    break
                except (urllib.error.URLError, TimeoutError, socket.timeout):
                    status, message, cause = 504, "Dependency timeout or connection error", dependency
                    break
        elapsed = time.monotonic() - started
        with LOCK:
            COUNTS[status] += 1
            LATENCY_SUM += elapsed
            for index, bound in enumerate(BUCKETS):
                if elapsed <= bound:
                    LATENCY[index] += 1
        emit(event="request", level="INFO" if status == 200 else "ERROR", status_code=status,
             message=message, request_id=request_id, dependency=cause,
             duration_ms=round(elapsed*1000, 2), scenario_id=fault["scenario"])
        self.reply(status, {"service": SERVICE, "status_code": status, "request_id": request_id})


def traffic():
    """Single in-flight request, capped rate; no uncontrolled worker creation."""
    base = min(5.0, max(0.1, float(os.environ.get("REQUESTS_PER_SECOND", "1"))))
    counter = 0
    while True:
        started = time.monotonic()
        counter += 1
        req = urllib.request.Request("http://order-service:8080/work",
                                     headers={"X-Request-ID": f"{time.time_ns()}-{counter}"})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                response.read(16384)
        except urllib.error.HTTPError as exc:
            exc.close()
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            pass
        # Small rate variation provides differing counts between minute windows.
        rate = base * (0.75, 1.0, 1.25)[int(time.time() // 60) % 3]
        time.sleep(max(0.0, 1/min(rate, 5.0) - (time.monotonic()-started)))


def control(name, duration):
    import sys
    scenarios = json.load(sys.stdin)
    if name not in scenarios:
        raise SystemExit("Unknown scenario. Choose: " + ", ".join(scenarios))
    scenario = scenarios[name]
    duration = scenario.get("duration_seconds", 180) if duration is None else duration
    if not 1 <= duration <= MAX_FAULT_SECONDS:
        raise SystemExit("Duration must be 1..900 seconds")
    # Reset first; abort if any service is unreachable, avoiding an unknown mixed state.
    for service in SERVICES:
        http_json(f"http://{service}:8080/fault", {"mode": "normal", "scenario": "normal"}, timeout=5)
    if scenario["mode"] != "normal":
        result = http_json(f'http://{scenario["service"]}:8080/fault',
                           {**scenario, "scenario": name, "duration_seconds": duration}, timeout=5)
        print(json.dumps(result))
    else:
        print("All demo faults reset. Normal traffic continues.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve", "traffic", "control"])
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--duration", type=int)
    args = parser.parse_args()
    if args.command == "traffic":
        traffic()
    elif args.command == "control":
        control(args.scenario, args.duration)
    else:
        if SERVICE not in SERVICES:
            raise SystemExit("Invalid SERVICE_NAME")
        emit(event="lifecycle", level="INFO", message="Demo service started")
        BoundedServer(("0.0.0.0", 8080), Handler).serve_forever()
