#!/usr/bin/env bash
set -euo pipefail

export LH_PIPELINE_USERID="${LH_PIPELINE_USERID:-admin}"
export LH_PIPELINE_PASSWORD="${LH_PIPELINE_PASSWORD:-admin}"
export LH_PIPELINE_NAME="${LH_PIPELINE_NAME:-pce3}"
export LH_PIPELINE_BATCH_S3_URL="${LH_PIPELINE_BATCH_S3_URL:-s3://loghawk-data/somefolder/}"

cd "$(dirname "$0")"
exec python -m loghawk.admin.pipeline_cron
