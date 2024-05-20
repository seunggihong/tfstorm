import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

# 데이터셋 스키마 정의
HelloWorldSchema = Unischema('HelloWorldSchema', [
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
    UnischemaField('array_4d', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
])

def row_generator(x):
    """데이터셋의 각 행을 생성하는 함수."""
    return {'id': x,
            'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
            'array_4d': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}

def generate_petastorm_dataset(output_url='file:///tmp/hello_world_dataset'):
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext

    # 데이터셋 생성 및 저장
    rows_count = 10
    with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):
        rows_rdd = sc.parallelize(range(rows_count))\
            .map(row_generator)\
            .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

        spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)
