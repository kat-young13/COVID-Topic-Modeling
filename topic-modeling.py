from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql.functions import concat_ws, col, explode, flatten, collect_list, lit
import re as re
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
import time
import numpy as np
import json
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.functions import col


def normalizeWords(text):
    ''' Text preprocessing '''
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
    test = [word for word in test if len(word) > 2]
    return (test, text[1], len(test))

def getTopic(line):
    ''' Used with transformed data from LDA in order to assign topic '''
    maximum = max(line)
    for i in range(0, len(line)):
        if line[i] == maximum:
            return line, i

def getWords(line):
    ''' Used when obtaining top x words for a topic '''
    words = []
    for i in line[0]:
        words.append(vocabArray[i])
    return line[0], line[1], words

def addVectors(x,y):
    ''' Used for calculating word frequency over all documents '''
    added = []
    for i in range(0, len(x)):
        added.append(x[i] + y[i])
    return added

def export_for_pyLDAvis(filename, model, transformed, df_txts, vocab, term_frequency):
    ''' Export for visualization purposes -- pyLDAvis '''
    #topic_term_dists
    topic_term_dists = np.array(model.topicsMatrix().toArray()).T
    #doc_topic_dists
    doc_topic_dists = np.array([x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']])
    #doc_lengths
    doc_lengths = [r[0] for r in df_txts.select("length").collect()]

    pyLDA = {}
    pyLDA.update({"topic_term_dists":topic_term_dists.tolist()})
    pyLDA.update({"doc_topic_dists":doc_topic_dists.tolist()})
    pyLDA.update({"doc_lengths":doc_lengths})
    pyLDA.update({"vocab":vocab})
    pyLDA.update({"term_frequency":term_frequency})

    with open(filename, 'w') as outfile:
        json.dump(pyLDA, outfile, indent=4)



if __name__ == "__main__":

    # create spark conf
    conf = SparkConf().setMaster("local").setAppName("TopicModeling")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    start_time = time.time()
    spark = SparkSession.builder.appName("TopicModeling").config("spark.driver.memory", "15g").getOrCreate()

    ##############################
    ### read in pdf json files ###
    ##############################

    # read in a sample file to infer schema
    new_df = sqlContext.read.json("test/pdf_json/0a9c92624fa4e3cfa24493d242d9dbd2192c5a88.json", multiLine=True)
    print(new_df.show())
    schema = new_df.schema
    # read in whole thing with correct schema (saves a ton of time)
    new_df = sqlContext.read.schema(schema).json("test/pdf_json/*", multiLine=True)
    print("time to read first set of json set")
    print("--- %s seconds ---" % (time.time() - start_time))

    # parse the paper id and the text we want
    test = new_df.selectExpr("paper_id", "abstract.text as abs", "body_text.text as body", "metadata.title")

    ##############################
    ### read in pmc json files ###
    ##############################

    new_df1 = sqlContext.read.json("test/pmc_json/PMC59549.xml.json", multiLine=True)
    schema1 = new_df1.schema
    new_dfs1 = sqlContext.read.schema(schema1).json("test/pmc_json/*", multiLine=True)
    print("time to read second set of json set")
    print("--- %s seconds ---" % (time.time() - start_time))
    test2 = new_dfs1.selectExpr("paper_id", "body_text.text as body", "metadata.title")

    #######################
    ### data processing ###
    #######################

    # combine all the text columns into one new column
    temp = test.withColumn("full_text", concat_ws(' ', test.abs, test.body, test.title))
    temp2 = test2.withColumn("full_text", concat_ws(' ',test2.body, test2.title))

    # union the two different datasets
    temp = temp.select("paper_id", "full_text")
    temp2 = temp2.select("paper_id", "full_text")
    temp = temp.union(temp2)
    #print(temp.show())

    # convert dataframe to rdd for preprocessing text data
    temp = temp.rdd.map(lambda x: (x['full_text'], x['paper_id'])).filter(lambda x: x is not None)

    # tokenize and clean the data
    temp = temp.map(normalizeWords)

    # convert rdd back to dataframe with full_text, paper_id, and length
    df_txts = sqlContext.createDataFrame(temp, ["full_text", 'paper_id', 'length'])
    df_txts.show()

    ############################
    ### TF-IDF VECTORIZATION ###
    ############################


    # create the document frequencies in the form of a count vectorizer
    cv = CountVectorizer(inputCol="full_text", outputCol="raw_features", minDF=2.0)
    cvmodel = cv.fit(df_txts)
    result_cv = cvmodel.transform(df_txts)

    # create the tfidf vectorization using the df computed above
    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(result_cv)
    result_tfidf = idfModel.transform(result_cv)
    result_tfidf.show()

    ######################
    ### LDA CLUSTERING ###
    ######################

    # create and train our clustering model
    num_topics = 3 # tune number of topics
    max_iterations = 100
    lda = LDA(k=num_topics, maxIter=max_iterations)
    result_tfidf = result_tfidf.select("paper_id", "features")
    model = lda.fit(result_tfidf)


    # calculate perplexity for use in tuning number of topics
    likelihood = model.logLikelihood(result_tfidf)
    perplexity = model.logPerplexity(result_tfidf)
    #print("The lower bound on the log likelihood of the entire corpus: " + str(likelihood))
    #print("The upper bound on perplexity: " + str(perplexity))
    topic_tuning_results = spark.createDataFrame([(likelihood, perplexity)], ("likelihood", "perplexity"))


    # transform our data and get corresponding topic
    transformed = model.transform(result_tfidf)
    trans_rdd = transformed.rdd.map(lambda x : ((x['paper_id'], x['features']), x['topicDistribution']))
    trans_rdd = trans_rdd.mapValues(getTopic)

    trans_rdd = trans_rdd.map(lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))
    docs_with_topics = sqlContext.createDataFrame(trans_rdd, ["paper_id", "features", "topicDistribution", "topic"])
    docs_with_topics.show()

    # get the top words from topics
    wordNumbers = 5
    topicIndices = model.describeTopics(wordNumbers)

    vocabArray = cvmodel.vocabulary # get vocabulary
    topicInd = topicIndices.rdd.map(lambda x : (x['topic'], (x['termIndices'], x['termWeights'])))
    topicInd = topicInd.mapValues(getWords)
    topicInd = topicInd.map(lambda x: (x[0], x[1][0], x[1][1], x[1][2]))
    final_df = sqlContext.createDataFrame(topicInd, ["topic", "termIndices", "termWeights", "termWords"])
    final_df.show() # info about each topic

    ################################
    ### Export for Visualization ###
    ################################

    # get each word's total frequency across documents
    rf_rdd = result_cv.rdd.map(lambda x : (x['raw_features']))\
        .map(lambda x: (1, DenseVector(x)))\
        .reduceByKey(addVectors)
    frequencies = rf_rdd.collect()[0][1]

    # export all info for pyLDAvis
    export_for_pyLDAvis("topic-info.json", model, transformed, df_txts, vocabArray, frequencies)

    #################################
    ### Export Useful Information ###
    #################################
    paper_topics = docs_with_topics.select("paper_id", "topic")
    topic_info = final_df.withColumn("termIndices", final_df["termIndices"].cast("string"))\
        .withColumn("termWeights", final_df["termWeights"].cast("string"))\
        .withColumn("termWords", final_df["termWords"].cast("string"))

    # output topics assigned to each paper, and x most common words in topic
    paper_topics.coalesce(1).write.options(header='true').csv('out/paper_topics')
    topic_info.coalesce(1).write.options(header='true').csv('out/topic_info')
    topic_tuning_results.coalesce(1).write.options(header='true').csv('out/perplexity_likelihood')


    print("--- %s seconds ---" % (time.time() - start_time))
