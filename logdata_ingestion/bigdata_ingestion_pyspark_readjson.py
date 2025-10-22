from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("SimplePySparkJob").getOrCreate()

# Read a CSV file into a DataFrame
input_file = "C:\\aiopsmain\\my_work\\input\\2.json"
df = spark.read.option("multiLine", "true").json(input_file)

df.show()

spark.stop()
