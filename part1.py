import numpy as np
import random
import sys
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":

    # 3 arguments: script name, path to data, output path
    # exit if not right number of arguments
    if len(sys.argv) != 3:
        sys.exit(2)

    spark = SparkSession.builder.appName(
        "Assignment 3: Group Component"
    ).getOrCreate()

    # load data
    data = spark.read.csv(sys.argv[1], inferSchema=True)

    # make list with features
    features = np.char.add(
        np.repeat("_c", 41).astype(str), (np.arange(0, 41).astype(str))
    )
    label = "_c41"

    # transform label from normal/anomaly into 0/1
    indexer = StringIndexer(inputCol=label, outputCol="label")
    indexed = indexer.fit(data).transform(data)

    # gather all features into one column
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    dataTransform = assembler.transform(indexed)

    # Decision Tree
    iterations = 10
    TrainAccuraciesDT = np.array([])
    TestAccuraciesDT = np.array([])
    times_dt = np.array([])
    for i in range(iterations):
        random.seed(i)
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

        (trainingData, testData) = dataTransform.randomSplit([0.7, 0.3])

        dt_start = time.time()
        model = dt.fit(trainingData)
        dt_end = time.time()

        exec_time = dt_end - dt_start
        times_dt = np.append(times_dt, exec_time)

        # predictions on training and test data sets
        train_pred = model.transform(trainingData)
        test_pred = model.transform(testData)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )

        # calculate training and test accuracy - add to results array
        trainAccuracy = evaluator.evaluate(train_pred)
        testAccuracy = evaluator.evaluate(test_pred)
        trainAccuraciesDT = np.append(trainAccuraciesDT, trainAccuracy)
        testAccuraciesDT = np.append(testAccuraciesDT, testAccuracy)

    # summary statistics
    minTrainDT = np.min(trainAccuraciesDT)
    maxTrainDT = np.max(trainAccuraciesDT)
    meanTrainDT = np.mean(trainAccuraciesDT)
    stdTrainDT = np.std(trainAccuraciesDT)
    minTestDT = np.min(testAccuraciesDT)
    maxTestDT = np.max(testAccuraciesDT)
    meanTestDT = np.mean(testAccuraciesDT)
    stdTestDT = np.std(testAccuraciesDT)
    minTimeDT = np.min(times_dt)
    maxTimeDT = np.max(times_dt)
    meanTimeDT = np.mean(times_dt)
    stdTimeDT = np.std(times_dt)
    
    dt_columns = ["Measure", "Training", "Test"]
    dt_values = [
        ("Min Accuracy", minTrainDT, minTestDT),
        ("Max Accuracy", maxTrainDT, maxTestDT),
        ("Mean Accuracy", meanTrainDT, meanTestDT),
        ("Stdev Accuracy", stdTrainDT, stdTestDT),
        ("Min Run-Time", minTimeDT, ""),
        ("Max Run-Time", maxTimeDT, ""),
        ("Mean Run-Time", meanTimeDT, ""),
        ("Stdev Run-Time", stdTimeDT, "")
    ]

    dt_results = spark.createDataFrame(dt_values, dt_columns)
    dt_results.coalesce(1).write.csv(sys.argv[2] + "/results", header=True)