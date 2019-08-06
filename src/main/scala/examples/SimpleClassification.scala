package examples

import org.apache.spark.sql.SparkSession
import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.classification._
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.result.ClassificationResultsHolder
import org.incal.spark_ml.models.setting.ClassificationLearningSetting

import scala.concurrent.ExecutionContext.Implicits.global

object SimpleClassification extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val sepalLength, sepalWidth, petalLength, petalWidth, clazz = Value
  }

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val outputColumnName = Column.clazz.toString
  val featureColumnNames = columnNames.filter(_ != outputColumnName)

  // read a csv and create a data frame with given column names
  val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  val df = remoteCsvToDataFrame(url, false)(session).toDF(columnNames :_*)

  // index the clazz column since it's of the string type
  val df2 = indexStringCols(Seq((outputColumnName, Nil)))(df)

  // turn the data frame into ML-ready one with features and a label
  val finalDf = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df2)

  // logistic regression spec
  val logisticRegressionSpec = LogisticRegression(
    family = Some(LogisticModelFamily.Multinomial),
    maxIteration = Left(Some(200)),
    regularization = Left(Some(0.1)),
    elasticMixingRatio = Left(Some(0.5))
  )

  // multi-layer perceptron spec
  val multiLayerPerceptronSpec = MultiLayerPerceptron(
    hiddenLayers = Seq(5, 5)
  )

  // random forest spec
  val randomForestSpec = RandomForest(
    core = TreeCore(maxDepth = Left(Some(5)))
  )

  // learning setting
  val learningSetting = ClassificationLearningSetting(
    repetitions = Some(10),
    collectOutputs = false // if outputs should be collected -> accessible in a results-holder at `expectedActualOutputs`
  )

  // aux function to get a mean training and test accuracy
  def calcMeanAccuracy(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.accuracy).get
    (trainingAccuracy.mean, testAccuracy.mean)
  }

  for {
    // run the logistic regression and get results
    logisticRegressionResults <- mlService.classify(finalDf, logisticRegressionSpec, learningSetting)

    // run the multi-layer perceptron and get results
    multiLayerPerceptronResults <- mlService.classify(finalDf, multiLayerPerceptronSpec, learningSetting)

    // run the random forest and get results
    randomForestResults <- mlService.classify(finalDf, randomForestSpec, learningSetting)
  } yield {
    val (lrTrainingAccuracy, lrTestAccuracy) = calcMeanAccuracy(logisticRegressionResults)
    val (mlpTrainingAccuracy, mlpTestAccuracy) = calcMeanAccuracy(multiLayerPerceptronResults)
    val (rfTrainingAccuracy, rfTestAccuracy) = calcMeanAccuracy(randomForestResults)

    println(s"Logistic regression    (accuracy): $lrTrainingAccuracy / $lrTestAccuracy")
    println(s"Multi-layer perceptron (accuracy): $mlpTrainingAccuracy / $mlpTestAccuracy")
    println(s"Random forest          (accuracy): $rfTrainingAccuracy / $rfTestAccuracy")
  }
})