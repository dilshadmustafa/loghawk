from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)

import os
import loghawk.config as config


# =========================================================
# 0. Local Spark / Java configuration
# =========================================================

os.environ["JAVA_HOME"] = r"C:\\jdk-17"
os.environ["HADOOP_HOME"] = r"C:\\hadoop"
os.environ["PATH"] = r"C:\\jdk-17\\bin;C:\\hadoop\\bin;" + os.environ.get("PATH", "")
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

input_path = "s3a://loghawk-data/raw/year=2026/month=09/day=23/"
output_path = "s3a://loghawk-data/features/year=2026/month=09/day=23/"

def run(input_path: str, output_path: str):
    """
    Execute Stage A feature engineering.
    """

    # =========================================================
    # 1. Spark session
    # =========================================================

    spark = (
        SparkSession.builder
        .appName("LogHawk-FeatureEngineering-S3")
        .master("local[1]")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:8333")
        .config("spark.hadoop.fs.s3a.access.key", "somekey")
        .config("spark.hadoop.fs.s3a.secret.key", "somesecret")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        #.config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
        )

        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config(
        "spark.hadoop.fs.s3a.threads.keepalivetime",
        "60"
        )
        .config(
            "spark.hadoop.fs.s3a.multipart.purge",
            "false"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")


    # =========================================================
    # 2. Input / Output
    # =========================================================


    print("Input :", input_path)
    print("Output:", output_path)


    # =========================================================
    # 3. Expected JSON schema
    # =========================================================

    schema = StructType([
        StructField("timestamp", StringType(), True),
        StructField("service", StringType(), True),
        StructField("level", StringType(), True),
        StructField("message", StringType(), True),
        StructField("status_code", IntegerType(), True),
        StructField("exception", StringType(), True),
    ])


    # =========================================================
    # 4. Read compressed JSON logs from S3
    # =========================================================

    print("Reading raw logs from S3...")

    logs = (
        spark.read
        .schema(schema)
        .json(input_path)
    )


    # =========================================================
    # 5. Normalize
    # =========================================================

    logs = (
        logs
        .withColumn("event_time", F.to_timestamp("timestamp"))
        .withColumn("level", F.upper(F.trim(F.col("level"))))
        .withColumn("service", F.coalesce(F.col("service"), F.lit("unknown")))
        .withColumn("message", F.coalesce(F.col("message"), F.lit("")))
        .withColumn("exception", F.coalesce(F.col("exception"), F.lit("")))
    )


    # =========================================================
    # 6. Create useful flags
    # =========================================================

    logs = (
        logs
        .withColumn("is_info", (F.col("level") == "INFO").cast("int"))
        .withColumn("is_warn", (F.col("level") == "WARN").cast("int"))
        .withColumn("is_error", (F.col("level") == "ERROR").cast("int"))
        .withColumn(
            "is_http_4xx",
            ((F.col("status_code") >= 400) & (F.col("status_code") < 500)).cast("int")
        )
        .withColumn(
            "is_http_5xx",
            ((F.col("status_code") >= 500) & (F.col("status_code") < 600)).cast("int")
        )
        .withColumn(
            "is_timeout",
            F.when(F.lower(F.col("message")).contains("timeout"), 1).otherwise(0)
        )
        .withColumn(
            "is_connection_error",
            F.when(
                F.lower(F.col("message")).rlike("connection refused|connection reset|connection failed"),
                1
            ).otherwise(0)
        )
        .withColumn(
            "is_auth_failure",
            F.when(
                F.lower(F.col("message")).rlike("authentication failed|unauthorized|invalid credentials|login failed"),
                1
            ).otherwise(0)
        )
    )


    # =========================================================
    # 7. Remove records with invalid timestamps
    # =========================================================

    logs = logs.filter(F.col("event_time").isNotNull())


    # =========================================================
    # 8. Create 1-minute window
    # =========================================================

    logs = logs.withColumn("window", F.window(F.col("event_time"), "1 minute"))


    # =========================================================
    # 9. Aggregate
    # =========================================================

    features = (
        logs.groupBy("service", "window")
        .agg(
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
        )
    )


    # =========================================================
    # 10. Flatten window
    # =========================================================

    features = (
        features
        .withColumn("timestamp", F.col("window.start"))
        .drop("window")
    )


    # =========================================================
    # 11. Calculate rates
    # =========================================================

    features = (
        features
        .withColumn(
            "error_rate",
            F.when(F.col("total_log_count") > 0, F.col("error_count") / F.col("total_log_count")).otherwise(0.0)
        )
        .withColumn(
            "warning_rate",
            F.when(F.col("total_log_count") > 0, F.col("warning_count") / F.col("total_log_count")).otherwise(0.0)
        )
        .withColumn(
            "http_5xx_rate",
            F.when(F.col("total_log_count") > 0, F.col("http_5xx_count") / F.col("total_log_count")).otherwise(0.0)
        )
        .withColumn(
            "timeout_rate",
            F.when(F.col("total_log_count") > 0, F.col("timeout_count") / F.col("total_log_count")).otherwise(0.0)
        )
    )


    # =========================================================
    # 12. Select final feature set
    # =========================================================

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


    # =========================================================
    # 13. Inspect feature data
    # =========================================================

    print("")
    print("Generated feature dataset:")
    print("")

    features.orderBy(F.col("timestamp")).show(20, truncate=False)


    # =========================================================
    # 14. Write features to S3 as Parquet
    # =========================================================

    print("")
    print("Writing feature data to S3...")
    print("")

    (
        features
        .write
        .mode("overwrite")
        .partitionBy("service")
        .parquet(output_path)
    )


    # =========================================================
    # 15. Completion
    # =========================================================

    print("")
    print("==============================================")
    print("LogHawk feature generation completed")
    print("==============================================")
    print("")
    print("Input : ", input_path)
    print("Output: ", output_path)
    print("")

    spark.stop()

    return output_path


if __name__ == "__main__":
    run(
        input_path=input_path,
        output_path=output_path,
    )






