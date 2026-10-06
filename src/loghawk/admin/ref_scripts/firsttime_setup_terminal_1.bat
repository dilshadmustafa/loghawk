
# download and install python 3.12
# https://www.python.org/downloads/release/python-31210/
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools wheel

pip install pyspark==3.5.9
pyspark --version
python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[*]').getOrCreate(); print('Spark:',s.version); print('Python:',__import__('sys').version); print('Java:',s.sparkContext._jvm.java.lang.System.getProperty('java.version')); print('Hadoop:',s.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()); s.stop()"

pip install -r requirements.txt
python -m pip install -e .
python -m loghawk.admin.setup_duckdb
python -m loghawk.admin.setup_lancedb

