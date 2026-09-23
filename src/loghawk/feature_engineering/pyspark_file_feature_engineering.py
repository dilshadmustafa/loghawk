from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
)
import loghawk.config as config

import os

os.environ["JAVA_HOME"] = r"C:\\jdk-17"
os.environ["HADOOP_HOME"] = r"C:\\hadoop"
os.environ["PATH"] += r";C:\\hadoop\\bin"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

# ---------------------------------------------------------
# 1. Spark session
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("LogHawk-FeatureEngineering")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. Input
# ---------------------------------------------------------

INPUT_PATH = str(config.LH_LOG_DIR)
OUTPUT_PATH = str(config.LH_FEATURE_DIR)

# Expected JSON example:
#
# {
#   "timestamp": "2026-09-15T10:32:11Z",
#   "service": "payment-service",
#   "level": "ERROR",
#   "message": "Database connection timeout",
#   "status_code": 500,
#   "exception": "ConnectionTimeoutException"
# }


# ---------------------------------------------------------
# 3. Read raw logs
# ---------------------------------------------------------

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("service", StringType(), True),
    StructField("level", StringType(), True),
    StructField("message", StringType(), True),
    StructField("status_code", IntegerType(), True),
    StructField("exception", StringType(), True),
])

logs = (
    spark.read
    .schema(schema)
    .json(INPUT_PATH)
)


# ---------------------------------------------------------
# 4. Normalize
# ---------------------------------------------------------

logs = (
    logs
    .withColumn(
        "event_time",
        F.to_timestamp("timestamp")
    )
    .withColumn(
        "level",
        F.upper(F.trim(F.col("level")))
    )
    .withColumn(
        "service",
        F.coalesce(
            F.col("service"),
            F.lit("unknown")
        )
    )
    .withColumn(
        "message",
        F.coalesce(
            F.col("message"),
            F.lit("")
        )
    )
)


# ---------------------------------------------------------
# 5. Create useful flags
# ---------------------------------------------------------

logs = (
    logs

    # Log levels
    .withColumn(
        "is_info",
        (F.col("level") == "INFO").cast("int")
    )
    .withColumn(
        "is_warn",
        (F.col("level") == "WARN").cast("int")
    )
    .withColumn(
        "is_error",
        (F.col("level") == "ERROR").cast("int")
    )

    # HTTP status
    .withColumn(
        "is_http_4xx",
        (
            (F.col("status_code") >= 400) &
            (F.col("status_code") < 500)
        ).cast("int")
    )
    .withColumn(
        "is_http_5xx",
        (
            (F.col("status_code") >= 500) &
            (F.col("status_code") < 600)
        ).cast("int")
    )

    # Timeout
    .withColumn(
        "is_timeout",
        F.when(
            F.lower(F.col("message")).contains("timeout"),
            1
        ).otherwise(0)
    )

    # Connection errors
    .withColumn(
        "is_connection_error",
        F.when(
            F.lower(F.col("message")).rlike(
                "connection refused|connection reset|connection failed"
            ),
            1
        ).otherwise(0)
    )

    # Authentication failures
    .withColumn(
        "is_auth_failure",
        F.when(
            F.lower(F.col("message")).rlike(
                "authentication failed|unauthorized|invalid credentials|login failed"
            ),
            1
        ).otherwise(0)
    )
)


# ---------------------------------------------------------
# 6. Create 1-minute window
# ---------------------------------------------------------

logs = logs.withColumn(
    "window",
    F.window(
        F.col("event_time"),
        "1 minute"
    )
)


# ---------------------------------------------------------
# 7. Aggregate
# ---------------------------------------------------------

features = (
    logs
    .groupBy(
        "service",
        "window"
    )
    .agg(

        # Volume
        F.count("*").alias("total_log_count"),

        # Levels
        F.sum("is_info").alias("info_count"),
        F.sum("is_warn").alias("warning_count"),
        F.sum("is_error").alias("error_count"),

        # HTTP
        F.sum("is_http_4xx").alias("http_4xx_count"),
        F.sum("is_http_5xx").alias("http_5xx_count"),

        # Patterns
        F.sum("is_timeout").alias("timeout_count"),
        F.sum("is_connection_error").alias(
            "connection_error_count"
        ),
        F.sum("is_auth_failure").alias(
            "authentication_failure_count"
        ),

        # Diversity
        F.countDistinct("exception").alias(
            "unique_exception_count"
        ),

        F.countDistinct(
            F.when(
                F.col("level") == "ERROR",
                F.col("message")
            )
        ).alias(
            "unique_error_message_count"
        )
    )
)


# ---------------------------------------------------------
# 8. Flatten window
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "timestamp",
        F.col("window.start")
    )
    .drop("window")
)


# ---------------------------------------------------------
# 9. Calculate rates
# ---------------------------------------------------------

features = (
    features

    .withColumn(
        "error_rate",
        F.when(
            F.col("total_log_count") > 0,
            F.col("error_count") /
            F.col("total_log_count")
        ).otherwise(0.0)
    )

    .withColumn(
        "warning_rate",
        F.when(
            F.col("total_log_count") > 0,
            F.col("warning_count") /
            F.col("total_log_count")
        ).otherwise(0.0)
    )

    .withColumn(
        "http_5xx_rate",
        F.when(
            F.col("total_log_count") > 0,
            F.col("http_5xx_count") /
            F.col("total_log_count")
        ).otherwise(0.0)
    )

    .withColumn(
        "timeout_rate",
        F.when(
            F.col("total_log_count") > 0,
            F.col("timeout_count") /
            F.col("total_log_count")
        ).otherwise(0.0)
    )
)


# ---------------------------------------------------------
# 10. Select final feature set
# ---------------------------------------------------------

features = features.select(
    "timestamp",
    "service",

    "total_log_count",

    "info_count",
    "warning_count",
    "error_count",

    "error_rate",
    "warning_rate",

    "http_4xx_count",
    "http_5xx_count",
    "http_5xx_rate",

    "timeout_count",
    "timeout_rate",

    "connection_error_count",
    "authentication_failure_count",

    "unique_exception_count",
    "unique_error_message_count"
)


# ---------------------------------------------------------
# 11. Save
# ---------------------------------------------------------

(
    features
    .write
    .mode("overwrite")
    .partitionBy("service")
    .parquet(OUTPUT_PATH)
)


# ---------------------------------------------------------
# 12. Inspect
# ---------------------------------------------------------

features.orderBy(
    F.col("timestamp")
).show(
    20,
    truncate=False
)

print("Feature generation completed.")

spark.stop()
