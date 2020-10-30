from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql.functions import concat_ws, col, explode, flatten, collect_list, lit
from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType, ArrayType

conf = SparkConf().setMaster("local").setAppName("TopicModeling")
sc = SparkContext(conf=conf)
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

# read in json files
new_df = sqlContext.read.json("document_parses/test/000a0fc8bbef80410199e690191dc3076a290117.json", multiLine=True)
print(new_df.show())

# parse the paper id and the text we want
test = new_df.selectExpr("paper_id", "abstract.text as abs", "body_text.text as body", "metadata.title")

new_df1 = sqlContext.read.json("document_parses/pmc_json/PMC1054884.xml.json", multiLine=True)
test2 = new_df1.selectExpr("paper_id", "body_text.text as body", "metadata.title")

# combine all the text columns into one new column
temp = test.withColumn("full_text", concat_ws(' ', test.abs, test.body, test.title))
temp2 = test2.withColumn("full_text", concat_ws(' ',test2.body, test2.title))

# union the two different datasets
temp2 = temp2.withColumn('abs', lit(None).cast(temp.dtypes[1][1]))
temp2 = temp2.select("paper_id", "abs", "body", "title", "full_text")
temp = temp.union(temp2)
print(temp.show())

# convert dataframe to rdd for preprocessing text data
text = temp.rdd.map(lambda x: (x['full_text'], x['paper_id'])).filter(lambda x: x is not None)

# tokenize and clean the data
token = text.map(normalizeWords)
print(token.collect())

# convert rdd back to dataframe
df_txts = sqlContext.createDataFrame(token, ["full_text", 'paper_id'])
print(df_txts.show())

# create the document frequencies in the form of a count vectorizer
cv = CountVectorizer(inputCol="full_text", outputCol="raw_features", minDF=1.0)
cvmodel = cv.fit(df_txts)
result_cv = cvmodel.transform(df_txts)

# create the tfidf vectorization using the df computed above
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)
print(result_tfidf.show())

# create and train our clustering model
num_topics = 10
max_iterations = 100
lda = LDA(k=10, maxIter=100)
result_tfidf = result_tfidf.select("paper_id", "features")
model = lda.fit(result_tfidf)

# get the topics and transform our data
wordNumbers = 5
topics = model.describeTopics(wordNumbers)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)
transformed = model.transform(result_tfidf)
transformed.show()
# this holds all the words
vocabArray = cvmodel.vocabulary

print(vocabArray)