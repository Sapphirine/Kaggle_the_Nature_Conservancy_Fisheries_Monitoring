import joblib
import re
import pandas as pd
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint

a = joblib.load('/Users/pengfeiwang/Desktop/fisheries/output/bow_feature.pkl')

def find_label(path):
    '''define the label from path name'''
    if re.search('ALB', path):
        return(0)
    elif re.search('BET', path):
        return(1)
    elif re.search('DOL', path):
        return(2)
    elif re.search('LAG', path):
        return(3)
    elif re.search('NoF', path):
        return(4)
    elif re.search('OTHER', path):
        return(5)
    elif re.search('SHARK', path):
        return(6)
    elif re.search('YFT', path):
        return(7)
    
a['image_path'] = a['image_path'].apply(lambda x: find_label(x)).to_frame()

df = spark.createDataFrame(a)
df1 = df.rdd.map(lambda x: (x['image_path'], x['features']))
(trainingData, testData) = df1.randomSplit([0.8, 0.2])
trainingData = trainingData.map(lambda x: LabeledPoint(x[0],x[1]))

maxDepth_selection = [5, 10, 15, 20, 30]

model = RandomForest.trainClassifier(trainingData, numClasses=8, categoricalFeaturesInfo={}, \
                                     numTrees=500, featureSubsetStrategy="auto", \
                                     impurity='gini', maxDepth=5, maxBins=32)

predictions = model.predict(testData.map(lambda x: x[1]))
labelsAndPredictions = testData.map(lambda x: x[0]).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print "Precision is " + testErr


