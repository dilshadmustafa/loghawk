#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 is required. Install it, then rerun this script."
    exit 1
fi
if ! command -v java >/dev/null 2>&1; then
    echo "Java 17 is required. Install it, then rerun this script."
    exit 1
fi

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

python3.12 -m venv venv312
# shellcheck disable=SC1091
source venv312/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel

python -m pip install pyspark==3.5.9
pyspark --version
python -c 'from pyspark.sql import SparkSession; s=SparkSession.builder.master("local[*]").getOrCreate(); print("Spark:", s.version); print("Python:", __import__("sys").version); print("Java:", s.sparkContext._jvm.java.lang.System.getProperty("java.version")); print("Hadoop:", s.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()); s.stop()'

python -m pip install -r requirements.txt
python -m pip install -e .
python -m loghawk.admin.setup_duckdb
python -m loghawk.admin.setup_lancedb2

echo "Setup complete."
