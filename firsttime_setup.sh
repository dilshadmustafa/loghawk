#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

command -v python3.12 >/dev/null || {
    echo "Python 3.12 is required."
    exit 1
}
command -v java >/dev/null || {
    echo "Java 17 is required."
    exit 1
}
command -v docker >/dev/null || {
    echo "Docker is required."
    exit 1
}
command -v aws >/dev/null || {
    echo "AWS CLI is required."
    exit 1
}

if [[ -z "${JAVA_HOME:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        JAVA_HOME="$(/usr/libexec/java_home -v 17)"
    else
        JAVA_BIN="$(readlink -f "$(command -v java)")"
        JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi
fi
export JAVA_HOME
export HADOOP_HOME="${HADOOP_HOME:-$JAVA_HOME}"

echo
echo "============================================================"
echo "Creating Python 3.12 virtual environment"
echo "============================================================"
python3.12 -m venv venv312
# shellcheck disable=SC1091
source venv312/bin/activate
echo
echo "============================================================"
echo "Installing PySpark"
echo "============================================================"
python -m pip install --upgrade pip setuptools wheel
python -m pip install pyspark==3.5.9
python -c 'from pyspark.sql import SparkSession; s=SparkSession.builder.master("local[*]").getOrCreate(); print("Spark:",s.version); print("Python:",__import__("sys").version); print("Java:",s.sparkContext._jvm.java.lang.System.getProperty("java.version")); print("Hadoop:",s.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()); s.stop()'
echo
echo "============================================================"
echo "Installing packages mentioned in requirements.txt"
echo "============================================================"
python -m pip install -r requirements.txt
python -m pip install -e .
echo
echo "============================================================"
echo "Setting up DuckDB"
echo "============================================================"
python -m loghawk.admin.setup_duckdb
echo
echo "============================================================"
echo "Setting up LanceDB"
echo "============================================================"
python -m loghawk.admin.setup_lancedb

echo
echo "============================================================"
echo "Starting RustFS S3 server"
echo "============================================================"
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
echo
echo "============================================================"
echo "Creating the loghawk-data S3 bucket"
echo "============================================================"
aws --endpoint-url http://localhost:9000 s3 mb s3://loghawk-data

echo
echo "============================================================"
echo "Generating and uploading test data to S3"
echo "============================================================"
python src/loghawk/test/testdata.py
echo
echo "============================================================"
echo "Starting Temporal server"
echo "============================================================"
docker compose -f src/loghawk/admin/temporal-docker-compose.yml up -d

echo "Setup complete."
echo "RustFS console: http://localhost:9001"
echo "Temporal UI: http://localhost:8233"
echo "Start the worker and workflow using the firsttime_start_workflow scripts."
