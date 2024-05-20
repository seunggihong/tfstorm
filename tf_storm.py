from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter, make_spark_converter
import tensorflow as tf

spark = SparkSession.builder.appName("PetastormToTensorFlow").getOrCreate()

# HDFS에 저장된 Parquet 파일 경로
file_path = "hdfs:///path/to/your/parquet_file"

# Parquet 파일을 불러와 Spark DataFrame으로 변환
df = spark.read.parquet(file_path)

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:///tmp/petastorm_cache')

converter = make_spark_converter(df)

# Petastorm의 make_tf_dataset 함수를 사용하여 TensorFlow의 Dataset 객체 생성
with converter.make_tf_dataset() as dataset:
    # 필요한 경우 여기서 데이터 전처리 진행
    dataset = dataset.batch(32)  # 예시로 배치 사이즈를 32로 설정
    
    # 간단한 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    
    # 모델 학습
    model.fit(dataset, epochs=10)
converter.delete()