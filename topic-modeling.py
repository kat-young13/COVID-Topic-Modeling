from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql.functions import concat_ws, col, explode, flatten, collect_list
from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.ml.clustering import LDA

conf = SparkConf().setMaster("local").setAppName("TopicModeling")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


def normalizeWords(text):
    # print(text)
    test = re.compile(r'\W+', re.UNICODE).split(text[0].lower())
    test = [word for word in test if word.isalpha()]
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                  'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                  'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                  'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'html']
    test = [word for word in test if not word in stop_words]
    return (test, text[1])

spark = SparkSession.builder.appName("TopicModeling").getOrCreate()
new_df = sqlContext.read.json("document_parses/test/000a0fc8bbef80410199e690191dc3076a290117.json", multiLine=True)
print(new_df.printSchema())
print(new_df.show())
test = new_df.selectExpr("paper_id", "abstract.text as abs", "body_text.text as body", "metadata.title").cache()
print(test.printSchema())
print(test.show())
temp = test.withColumn("full_text", concat_ws(' ',test.abs,test.body,test.title)).cache()
print(temp.show())
text = temp.rdd.map(lambda x : (x['full_text'], x['paper_id'])).filter(lambda x: x is not None)
token = text.map(normalizeWords)
print(token.collect())
df_txts = sqlContext.createDataFrame(token, ["full_text", 'paper_id'])
print(df_txts.show())
cv = CountVectorizer(inputCol="full_text", outputCol="raw_features", minDF=1.0)
cvmodel = cv.fit(df_txts)
result_cv = cvmodel.transform(df_txts)
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)
print(result_tfidf.show())
num_topics = 10
max_iterations = 100
lda = LDA(k=10, maxIter=100)
result_tfidf = result_tfidf.select("paper_id", "features")
model = lda.fit(result_tfidf)