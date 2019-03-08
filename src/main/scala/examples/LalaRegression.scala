package examples

import org.apache.spark.sql.SparkSession
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.regression._
import org.incal.spark_ml.models.result.RegressionResultsHolder
import org.incal.spark_ml.models.setting.RegressionLearningSetting
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}

import scala.concurrent.ExecutionContext.Implicits.global

object LalaRegression extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

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

  // linear regression spec
  val linearRegressionSpec = LinearRegression(
    maxIteration = Left(Some(200)),
    regularization = Left(Some(0.3)),
    elasticMixingRatio = Left(Some(0.8))
  )

  // Gaussian linear regression spec
  val generalizedLinearRegressionSpec = GeneralizedLinearRegression(
    family = Some(GeneralizedLinearRegressionFamily.Gaussian),
    link = Some(GeneralizedLinearRegressionLinkType.Identity),
    maxIteration = Left(Some(200)),
    regularization = Left(Some(0.3))
  )

  // random regression forest spec
  val randomRegressionForestSpec = RandomRegressionForest(
    core = TreeCore(maxDepth = Left(Some(10)))
  )

  // gradient-boost regression tree spec
  val gradientBoostRegressionTreeSpec = GradientBoostRegressionTree(
    maxIteration = Left(Some(50))
  )

  // learning setting
  val learningSetting = RegressionLearningSetting(repetitions = Some(10))

  // aux function to get a mean training and test RMSE
  def calcMeanRMSE(results: RegressionResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingRMSE, Some(testRMSE), _) = metricStatsMap.get(RegressionEvalMetric.rmse).get
    (trainingRMSE.mean, testRMSE.mean)
  }

  for {
    // run the linear regression and get results
    linearRegressionResults <- mlService.regress(finalDf, linearRegressionSpec, learningSetting)

    // run the generalized linear regression and get results
    generalizedLinearRegressionResults <- mlService.regress(finalDf, generalizedLinearRegressionSpec, learningSetting)

    // run the random regression forest and get results
    randomRegressionForestResults <- mlService.regress(finalDf, randomRegressionForestSpec, learningSetting)

    // run the gradient-boost regression and get results
    gradientBoostRegressionTreeResults <- mlService.regress(finalDf, gradientBoostRegressionTreeSpec, learningSetting)
  } yield {
    val (lrTrainingRMSE, lrTestRMSE) = calcMeanRMSE(linearRegressionResults)
    val (glrTrainingRMSE, glrTestRMSE) = calcMeanRMSE(generalizedLinearRegressionResults)
    val (rrfTrainingRMSE, rrfTestRMSE) = calcMeanRMSE(randomRegressionForestResults)
    val (gbrtTrainingRMSE, gbrtTestRMSE) = calcMeanRMSE(gradientBoostRegressionTreeResults)

    println(s"Linear regression (RMSE): $lrTrainingRMSE / $lrTestRMSE")
    println(s"Generalized linear regression (RMSE): $glrTrainingRMSE / $glrTestRMSE")
    println(s"Random regression forest (RMSE): $rrfTrainingRMSE / $rrfTestRMSE")
    println(s"Gradient-boost regression tree (RMSE): $gbrtTrainingRMSE / $gbrtTestRMSE")
  }
})