from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel

conf = SparkConf().setMaster("local").setAppName("TopicModeling")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

spark = SparkSession.builder.appName("TopicModeling").getOrCreate()
new_df = sqlContext.read.json("archive/document_parses/pdf_json/000a0fc8bbef80410199e690191dc3076a290117.json", multiLine=True)
print(new_df.printSchema())
test = new_df.select("paper_id", "abstract.text", "body_text.text", "metadata.title")
print(test.show())