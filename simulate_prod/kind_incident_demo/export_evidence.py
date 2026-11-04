"""Export bounded demo request logs. Upload is opt-in; existing S3 data is protected."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from uuid import uuid4

SERVICES = ("order-service", "inventory-service", "payment-service")
BYTE_LIMIT = 8 * 1024 * 1024
REPO = Path(__file__).resolve().parents[2]


def timestamp(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone; use UTC with Z")
    return result.astimezone(dt.timezone.utc)


def kubectl(context, *arguments):
    result = subprocess.run(["kubectl", "--context", context, "--request-timeout=30s",
                             "-n", "loghawk-demo", *arguments],
                            capture_output=True, timeout=45, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[:2000])
    return result.stdout


def upload(files, batch_url, phase, env_file):
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
        from dotenv import dotenv_values
    except ImportError as exc:
        raise RuntimeError("Use LogHawk's venv312 environment for uploads (boto3 and python-dotenv required).") from exc
    values = dotenv_values(env_file) if env_file.is_file() else {}
    def setting(name, default=""):
        return os.environ.get(name, values.get(name) or default)
    endpoint = setting("LH_S3_ENDPOINT")
    access = setting("LH_S3_ACCESS_KEY_ID")
    secret = setting("LH_S3_SECRET_ACCESS_KEY")
    if not endpoint or not access or not secret:
        raise ValueError("Set LH_S3_ENDPOINT, LH_S3_ACCESS_KEY_ID and LH_S3_SECRET_ACCESS_KEY in the repository .env or environment")
    uri = urlparse(batch_url)
    prefix = uri.path.strip("/")
    if uri.scheme != "s3" or not uri.netloc or uri.query or uri.fragment or uri.username or not prefix:
        raise ValueError("Use a batch URL such as s3://loghawk-data/kind-demo-session-01/")
    if any(part in {".", ".."} for part in prefix.split("/")):
        raise ValueError("Invalid batch prefix")
    client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access,
                          aws_secret_access_key=secret, region_name=setting("LH_S3_REGION", "us-east-1"),
                          config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts":2},
                                        s3={"addressing_style":"path"}))
    planned = []
    for path in files:
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        folder = "evidence" if path.name == "export_manifest.json" else phase
        name = f"{phase}-export-manifest.json" if folder == "evidence" else path.name
        key = f"{prefix}/{folder}/{name}"
        exists = False
        try:
            old = client.head_object(Bucket=uri.netloc, Key=key)
            if old.get("Metadata", {}).get("sha256") != digest or old["ContentLength"] != len(body):
                raise ValueError(f"Refusing to overwrite s3://{uri.netloc}/{key}. Use a new batch prefix.")
            exists = True
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchKey", "NotFound"}:
                raise
        planned.append((key, body, digest, exists))
    # Check all existing keys before starting uploads; conditional writes also
    # protect against a concurrent exporter. Do not fall back to unconditional writes.
    for key, body, digest, exists in planned:
        if not exists:
            client.put_object(Bucket=uri.netloc, Key=key, Body=body, IfNoneMatch="*",
                              ContentType="application/json" if key.endswith('.json') else "application/x-ndjson",
                              Metadata={"sha256":digest})
        print(("Already present: " if exists else "Uploaded: ") + f"s3://{uri.netloc}/{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "raw"], required=True)
    parser.add_argument("--start", required=True, help="Inclusive UTC timestamp, e.g. 2026-11-03T10:00:00Z")
    parser.add_argument("--end", required=True, help="Exclusive UTC timestamp")
    parser.add_argument("--context", default="kind-loghawk")
    parser.add_argument("--output-dir", type=Path, default=REPO/"data"/"kind-demo")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--batch-s3-url", help="Same separate batch prefix for the train and raw exports")
    parser.add_argument("--env-file", type=Path, default=REPO/".env")
    args = parser.parse_args()
    start, end = timestamp(args.start), timestamp(args.end)
    if start >= end or end > dt.datetime.now(dt.timezone.utc):
        raise ValueError("Choose a completed time window with start before end")
    if end-start > dt.timedelta(hours=2):
        raise ValueError("Export windows must be at most two hours; choose a smaller window")
    if args.upload and not args.batch_s3_url:
        raise ValueError("--upload requires --batch-s3-url")
    output = args.output_dir / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"simulation":True, "phase":args.phase, "start_inclusive":start.isoformat(),
                "end_exclusive":end.isoformat(), "context":args.context,
                "namespace":"loghawk-demo", "sources":[],
                "notice":"Current pod logs only. Pod replacement/rotation can remove earlier logs. Verify window coverage."}
    paths = []
    for service in SERVICES:
        pods = json.loads(kubectl(args.context, "get", "pods", "-l", "app="+service, "-o", "json"))["items"]
        pods = [pod for pod in pods if not pod["metadata"].get("deletionTimestamp")]
        if len(pods) != 1:
            raise ValueError(f"Expected exactly one current {service} pod. Wait for rollout to complete.")
        pod = pods[0]
        name = pod["metadata"]["name"]
        restarts = sum(c.get("restartCount",0) for c in pod.get("status",{}).get("containerStatuses",[]))
        if restarts or timestamp(pod["metadata"]["creationTimestamp"]) > start:
            raise ValueError(f"{name} restarted or was created after the window began; choose a window fully covered by current logs")
        raw = kubectl(args.context, "logs", name, "-c", "app", "--since-time="+start.isoformat(),
                      "--limit-bytes="+str(BYTE_LIMIT), "--timestamps=false")
        if len(raw) >= BYTE_LIMIT:
            raise ValueError(f"{service} reached the export byte limit; choose a smaller time window")
        rows = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "request" and start <= timestamp(record["timestamp"]) < end:
                if record.get("service") != service:
                    raise ValueError("Unexpected service identity in log")
                rows.append(record)
        if not rows:
            raise ValueError(f"No request logs for {service}; check time window, traffic and log rotation")
        if len(rows) > 10000:
            raise ValueError("More than 10000 records in one file; choose a smaller time window for investigation imports")
        path = output / f"{service}.jsonl"
        with path.open("x",encoding="utf-8",newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
        paths.append(path)
        manifest["sources"].append({"service":service,"pod":name,"records":len(rows),
                                    "first_record":rows[0]["timestamp"],"last_record":rows[-1]["timestamp"],
                                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path = output/"export_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Saved evidence:",output)
    for item in manifest["sources"]:
        print(item["service"],item["records"],"records",item["first_record"],"to",item["last_record"])
    if args.upload:
        upload([*paths,manifest_path],args.batch_s3_url,args.phase,args.env_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Export failed: {exc}") from exc
