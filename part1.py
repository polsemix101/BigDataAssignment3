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
    dt_results.coalesce(1).write.csv(sys.argv[2] + "/dt_results", header=True)

    #Logistic Regression
    iterations = 10
    TrainAccuraciesLR = np.array([])
    TestAccuraciesLR = np.array([])
    times_lr = np.array([])
    for i in range(iterations):
        random.seed(i)
        lr = LogisticRegression(labelCol="label",featuresCol="features")

        (trainingData, testData) = dataTransform.randomSplit([0.7,0.3])

        lr_start = time.time()
        model = lr.fit(trainingData)
        lr_end = time.time()

        exec_time = lr_end - lr_start
        times_lr = np.append(times_lr, exec_time)

        train_pred = model.transform(trainingData)
        test_pred = model.transform(testData)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )

        trainAccuracy = evaluator.evaluate(train_pred)
        testAccuracy = evaluator.evaluate(test_pred)
        trainAccuraciesLR = np.append(trainAccuraciesLR, trainAccuracy)
        testAccuraciesLR = np.append(testAccuraciesLR, testAccuracy)

    # summary statistics
    minTrainLR = np.min(trainAccuraciesLR)
    maxTrainLR = np.max(trainAccuraciesLR)
    meanTrainLR = np.mean(trainAccuraciesLR)
    stdTrainLR = np.std(trainAccuraciesLR)
    minTestLR = np.min(testAccuraciesLR)
    maxTestLR = np.max(testAccuraciesLR)
    meanTestLR = np.mean(testAccuraciesLR)
    stdTestLR = np.std(testAccuraciesLR)
    minTimeLR = np.min(times_lr)
    maxTimeLR = np.max(times_lr)
    meanTimeLR = np.mean(times_lr)
    stdTimeLR = np.std(times_lr)
    
    lr_columns = ["Measure", "Training", "Test"]
    lr_values = [
        ("Min Accuracy", minTrainLR, minTestLR),
        ("Max Accuracy", maxTrainLR, maxTestLR),
        ("Mean Accuracy", meanTrainLR, meanTestLR),
        ("Stdev Accuracy", stdTrainLR, stdTestLR),
        ("Min Run-Time", minTimeLR, ""),
        ("Max Run-Time", maxTimeLR, ""),
        ("Mean Run-Time", meanTimeLR, ""),
        ("Stdev Run-Time", stdTimeLR, "")
    ]

    lr_results = spark.createDataFrame(lr_values, lr_columns)
    lr_results.coalesce(1).write.csv(sys.argv[2] + "/lr_results", header=True)

    spark.stop()