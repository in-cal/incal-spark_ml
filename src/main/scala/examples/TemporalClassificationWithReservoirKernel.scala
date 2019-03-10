package examples

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.BooleanType
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.models.classification.{ClassificationEvalMetric, LogisticModelFamily, LogisticRegression}
import org.incal.spark_ml.models.result.ClassificationResultsHolder
import org.incal.spark_ml.models.setting.{ClassificationLearningSetting, TemporalClassificationLearningSetting}
import org.incal.spark_ml.transformers.BooleanLabelIndexer
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}

import scala.concurrent.ExecutionContext.Implicits.global

object TemporalClassificationWithSlidingWindow extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val index,AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4,eyeDetection,eyeDetectionBool = Value
  }

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val outputColumnName = Column.eyeDetectionBool.toString
  val orderColumnName = Column.index.toString
  val featureColumnNames = columnNames.filter(name => name != outputColumnName && name != orderColumnName)

  // read csv and create a data frame with given column names
  val url = "https://in-cal.org/data/EEG_Eye_State_by_DBWH.csv"
  val df = remoteCsvToDataFrame(url, true)(session)

  val df2 = df.withColumn(outputColumnName, df(Column.eyeDetection.toString).cast(BooleanType))

  // turn the data frame into ML-ready one with features and a label
  val df3 = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df2)
  val finalDf = BooleanLabelIndexer(Some(outputColumnName)).transform(df3)

  finalDf.show(truncate = false)

  // logistic regression spec
  val logisticRegressionSpec = LogisticRegression(
    family = Some(LogisticModelFamily.Binomial),
    regularization = Right(Seq(10, 1, 0.1, 0.01, 0.001)),
    elasticMixingRatio = Right(Seq(0, 0.5, 1))
  )

  // learning setting
  val classificationLearningSetting = ClassificationLearningSetting(
    trainingTestSplitRatio = Some(0.75),
    featuresNormalizationType = Some(VectorScalerType.StandardScaler),
    crossValidationEvalMetric = Some(ClassificationEvalMetric.areaUnderROC),
    crossValidationFolds = Some(5),
    featuresSelectionNum = Some(3)
  )

  val temporalLearningSetting = TemporalClassificationLearningSetting(
    core = classificationLearningSetting,
    predictAhead = 100,
    slidingWindowSize = Right(Seq(4,5,6))
  )

  // aux function to get a mean training and test accuracy and AUROC
  def calcMeanAccuracyAndAUROC(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.accuracy).get
    val (trainingAUROC, Some(testAUROC), _) = metricStatsMap.get(ClassificationEvalMetric.areaUnderROC).get

    ((trainingAccuracy.mean, testAccuracy.mean), (trainingAUROC.mean, Some(testAUROC.mean)))
  }

  for {
    // run the logistic regression and get results
    results <- mlService.classifyTimeSeries(finalDf, logisticRegressionSpec, temporalLearningSetting)
  } yield {
    val ((trainingAccuracy, testAccuracy), (trainingAUROC, testAUROC)) = calcMeanAccuracyAndAUROC(results)

    println(s"AUROC   : $trainingAUROC / $testAUROC")
    println(s"Accuracy: $trainingAccuracy / $testAccuracy")
  }
})