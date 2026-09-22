from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("SimplePySparkJob").getOrCreate()

# Read a CSV file into a DataFrame
input_file = "C:\\aiopsmain\\my_work\\input\\1.csv"
df = spark.read.csv(input_file, header=True, inferSchema=True)

df.show()

spark.stop()
