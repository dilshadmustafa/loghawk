#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required. Install and start Docker, then rerun this script."
    exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
    echo "AWS CLI is required for bucket setup. Install it, then rerun this script."
    exit 1
fi

mkdir -p "$HOME/rustfs/data" "$HOME/rustfs/logs"
if docker container inspect loghawk-rustfs >/dev/null 2>&1; then
    docker start loghawk-rustfs
else
    docker run -d --name loghawk-rustfs \
        -p 9000:9000 -p 9001:9001 \
        -v "$HOME/rustfs/data:/data" \
        -v "$HOME/rustfs/logs:/logs" \
        rustfs/rustfs:latest
fi

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-rustfsadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-rustfsadmin}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
for attempt in {1..30}; do
    if aws --endpoint-url http://localhost:9000 s3 ls >/dev/null 2>&1; then
        break
    fi
    if [[ "$attempt" == "30" ]]; then
        echo "RustFS did not become ready at http://localhost:9000."
        exit 1
    fi
    sleep 2
done

if ! aws --endpoint-url http://localhost:9000 s3 ls s3://loghawk-data >/dev/null 2>&1; then
    aws --endpoint-url http://localhost:9000 s3 mb s3://loghawk-data
fi
aws --endpoint-url http://localhost:9000 s3 ls s3://loghawk-data

echo "RustFS is running. Console: http://localhost:9001"
echo "Run generate_upload_testdata.sh to generate and upload Train/Raw sample logs."
