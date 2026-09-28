# generic entity, not service
from pathlib import Path
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

import loghawk.config as config

os.environ["JAVA_HOME"] = config.JAVA_HOME
os.environ["HADOOP_HOME"] = config.HADOOP_HOME
os.environ["PATH"] = (str(Path(config.JAVA_HOME)) + str(Path("\\bin")) + ";" + str(Path(config.HADOOP_HOME)) + str(Path("\\bin")) + ";" + os.environ.get("PATH", ""))
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

INPUT_PATH = "s3a://loghawk-data/raw/year=2026/month=09/day=23/"
OUTPUT_PATH = "s3a://loghawk-data/features/year=2026/month=09/day=23/"

# Candidate identity fields. Raw logs do not need to contain all of them.
IDENTITY_COLUMNS = [
    "service", "application_id", "app_name", "container_name",
    "pod_name", "namespace", "hostname", "host", "database", "device"
]

def first_non_empty(columns):
    expressions = [
        F.when(
            F.col(c).isNotNull() & (F.trim(F.col(c)) != ""),
            F.trim(F.col(c)),
        )
        for c in columns
    ]
    return F.coalesce(*expressions, F.lit("unknown-entity"))

def run(input_path: str, output_path: str):
    spark = (
        SparkSession.builder
        .appName("LogHawk-FeatureEngineering-S3")
        .master("local[1]")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", config.LH_S3_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", config.LH_S3_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", config.LH_S3_SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.multipart.purge", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        print("Input :", input_path)
        print("Output:", output_path)

        # Common fields plus optional identity fields. service is NOT mandatory.
        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("service", StringType(), True),
            StructField("application_id", StringType(), True),
            StructField("app_name", StringType(), True),
            StructField("container_name", StringType(), True),
            StructField("pod_name", StringType(), True),
            StructField("namespace", StringType(), True),
            StructField("hostname", StringType(), True),
            StructField("host", StringType(), True),
            StructField("database", StringType(), True),
            StructField("device", StringType(), True),
            StructField("level", StringType(), True),
            StructField("message", StringType(), True),
            StructField("status_code", IntegerType(), True),
            StructField("exception", StringType(), True),
        ])

        print("Reading raw logs from S3...")
        logs = spark.read.schema(schema).json(input_path)

        logs = (logs
            .withColumn("event_time", F.to_timestamp("timestamp"))
            .withColumn("level", F.upper(F.trim(F.col("level"))))
            .withColumn("message", F.coalesce(F.col("message"), F.lit("")))
            .withColumn("exception", F.coalesce(F.col("exception"), F.lit(""))))

        for c in IDENTITY_COLUMNS:
            logs = logs.withColumn(
                c, F.when(F.col(c).isNotNull() & (F.trim(F.col(c)) != ""), F.trim(F.col(c))).otherwise(None)
            )

        # Generic identity used by Stage B/C. Prefer application/service identity
        # and fall back to runtime/infrastructure identity.
        logs = logs.withColumn("entity_id", first_non_empty(IDENTITY_COLUMNS))

        logs = (logs
            .withColumn("is_info", (F.col("level") == "INFO").cast("int"))
            .withColumn("is_warn", F.col("level").isin("WARN", "WARNING").cast("int"))
            .withColumn("is_error", (F.col("level") == "ERROR").cast("int"))
            .withColumn("is_http_4xx", ((F.col("status_code") >= 400) & (F.col("status_code") < 500)).cast("int"))
            .withColumn("is_http_5xx", ((F.col("status_code") >= 500) & (F.col("status_code") < 600)).cast("int"))
            .withColumn("is_timeout", F.when(F.lower(F.col("message")).contains("timeout"), 1).otherwise(0))
            .withColumn("is_connection_error", F.when(F.lower(F.col("message")).rlike("connection refused|connection reset|connection failed"), 1).otherwise(0))
            .withColumn("is_auth_failure", F.when(F.lower(F.col("message")).rlike("authentication failed|unauthorized|invalid credentials|login failed"), 1).otherwise(0)))

        logs = logs.filter(F.col("event_time").isNotNull())
        logs = logs.withColumn("window", F.window(F.col("event_time"), "1 minute"))

        # Aggregate by generic entity, not by service. Keep identity metadata
        # for Stage C correlation without making it part of the ML feature set.
        features = (logs.groupBy("entity_id", "window").agg(
            F.first("service", ignorenulls=True).alias("service"),
            F.first("application_id", ignorenulls=True).alias("application_id"),
            F.first("app_name", ignorenulls=True).alias("app_name"),
            F.first("container_name", ignorenulls=True).alias("container_name"),
            F.first("pod_name", ignorenulls=True).alias("pod_name"),
            F.first("namespace", ignorenulls=True).alias("namespace"),
            F.first("hostname", ignorenulls=True).alias("hostname"),
            F.first("host", ignorenulls=True).alias("host"),
            F.first("database", ignorenulls=True).alias("database"),
            F.first("device", ignorenulls=True).alias("device"),
            F.count("*").alias("total_log_count"),
            F.sum("is_info").alias("info_count"),
            F.sum("is_warn").alias("warning_count"),
            F.sum("is_error").alias("error_count"),
            F.sum("is_http_4xx").alias("http_4xx_count"),
            F.sum("is_http_5xx").alias("http_5xx_count"),
            F.sum("is_timeout").alias("timeout_count"),
            F.sum("is_connection_error").alias("connection_error_count"),
            F.sum("is_auth_failure").alias("authentication_failure_count"),
            F.countDistinct(F.when(F.col("exception") != "", F.col("exception"))).alias("unique_exception_count"),
            F.countDistinct(F.when(F.col("level") == "ERROR", F.col("message"))).alias("unique_error_message_count"),
        ))

        features = features.withColumn("timestamp", F.col("window.start")).drop("window")

        features = (features
            .withColumn("error_rate", F.when(F.col("total_log_count") > 0, F.col("error_count") / F.col("total_log_count")).otherwise(0.0))
            .withColumn("warning_rate", F.when(F.col("total_log_count") > 0, F.col("warning_count") / F.col("total_log_count")).otherwise(0.0))
            .withColumn("http_5xx_rate", F.when(F.col("total_log_count") > 0, F.col("http_5xx_count") / F.col("total_log_count")).otherwise(0.0))
            .withColumn("timeout_rate", F.when(F.col("total_log_count") > 0, F.col("timeout_count") / F.col("total_log_count")).otherwise(0.0)))

        features = features.select(
            "timestamp", "entity_id",
            "service", "application_id", "app_name", "container_name",
            "pod_name", "namespace", "hostname", "host", "database", "device",
            "total_log_count", "info_count", "warning_count", "error_count",
            "error_rate", "warning_rate", "http_4xx_count", "http_5xx_count",
            "http_5xx_rate", "timeout_count", "timeout_rate",
            "connection_error_count", "authentication_failure_count",
            "unique_exception_count", "unique_error_message_count"
        )

        print("\nGenerated feature dataset:\n")
        features.orderBy(F.col("timestamp")).show(20, truncate=False)

        # Do not partition by service/entity_id. They can be high-cardinality
        # and create many small files. The date is already in output_path.
        print("\nWriting feature data to S3...\n")
        features.write.mode("overwrite").parquet(output_path)

        print("\n==============================================")
        print("LogHawk feature generation completed")
        print("==============================================\n")
        print("Input :", input_path)
        print("Output:", output_path)
        return output_path
    finally:
        spark.stop()

if __name__ == "__main__":
    run(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
