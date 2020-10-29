from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql.functions import concat_ws, col, explode, flatten

conf = SparkConf().setMaster("local").setAppName("TopicModeling")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

spark = SparkSession.builder.appName("TopicModeling").getOrCreate()
new_df = sqlContext.read.json("document_parses/test/000a0fc8bbef80410199e690191dc3076a290117.json", multiLine=True)
print(new_df.printSchema())
test = new_df.selectExpr("paper_id", "abstract.text as abs", "body_text.text as body", "metadata.title")
print(test.show())
temp = test.withColumn("abs", explode(test.abs))
print(temp.show())
