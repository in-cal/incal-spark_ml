package examples

import org.apache.spark.sql.SparkSession
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.classification.{ClassificationEvalMetric, LogisticModelFamily, LogisticRegression, RandomForest}
import org.incal.spark_ml.models.result.ClassificationResultsHolder
import org.incal.spark_ml.models.setting.ClassificationLearningSetting
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}

import scala.concurrent.ExecutionContext.Implicits.global

object IrisClassificationWithCrossValidation extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val sepalLength, sepalWidth, petalLength, petalWidth, clazz = Value
  }

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val outputColumnName = Column.clazz.toString
  val featureColumnNames = columnNames.filter(_ != outputColumnName)

  // read csv and create a data frame with given column names
  val url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  val df = remoteCsvToDataFrame(url, false)(session).toDF(columnNames :_*)

  // turn the data frame into ML-ready one with features and a label
  val df2 = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df)

  // index the label column since it's a string
  val finalDf = indexStringCols(Seq(("label", Nil)))(df2)

  // logistic regression spec (note that regularization and elasticMixingRatio are defined by value sequences, which will be used for a param-grid cross-validation model selection)
  val logisticRegressionSpec = LogisticRegression(
    family = Some(LogisticModelFamily.Multinomial),
    regularization = Right(Seq(0, 0.1, 0.01)),
    elasticMixingRatio = Right(Seq(0, 0.25, 0.5, 0.75, 1))
  )

  // learning setting
  val learningSetting = ClassificationLearningSetting(
    repetitions = Some(10),
    crossValidationFolds = Some(5),
    crossValidationEvalMetric = Some(ClassificationEvalMetric.accuracy)
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
  } yield {
    val (lrTrainingAccuracy, lrTestAccuracy) = calcMeanAccuracy(logisticRegressionResults)

    println(s"Logistic regression (accuracy): $lrTrainingAccuracy / $lrTestAccuracy")
  }
})