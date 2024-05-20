import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, StructField, StructType
from petastorm.codecs import ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

# 스키마 정의
WeatherSchema = Unischema('WeatherSchema', [
    UnischemaField('date', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('temperature', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('humidity', np.float32, (), ScalarCodec(FloatType()), False),
])

# Spark 스키마 정의
spark_schema = StructType([
    StructField('date', StringType(), nullable=False),
    StructField('temperature', FloatType(), nullable=False),
    StructField('humidity', FloatType(), nullable=False)
])

def generate_petastorm_dataset_from_csv(csv_path, output_url='file:///tmp/weather_dataset'):
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()

    # CSV 파일을 PySpark를 이용해 직접 읽음
    df = spark.read.csv(csv_path, schema=spark_schema, header=True)
    
    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    with materialize_dataset(spark, output_url, WeatherSchema, rowgroup_size_mb):
        
        # PySpark DataFrame을 RDD로 변환하고, Petastorm의 dict_to_spark_row 함수를 사용하여 변환
        rows_rdd = df.rdd \
            .map(lambda row: {'date': row['date'], 'temperature': row['temperature'], 'humidity': row['humidity']}) \
            .map(lambda x: dict_to_spark_row(WeatherSchema, x))

        # 변환된 RDD를 다시 DataFrame으로 변환하고 Parquet으로 저장
        spark.createDataFrame(rows_rdd, WeatherSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)

# 함수 실행
generate_petastorm_dataset_from_csv('weatherAUG.csv')
