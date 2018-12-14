package org.incal.spark_ml

import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.min
import org.incal.spark_ml.models.results._
import org.incal.spark_ml.IncalSparkMLException
import org.incal.spark_ml.models.classification.ClassificationEvalMetric
import org.incal.spark_ml.models.regression.RegressionEvalMetric
import org.incal.core.util.STuple3

object MachineLearningUtil {

  val randomSplit = (splitRatio: Double) => (dataFrame: DataFrame) => {
    val Array(training, test) = dataFrame.randomSplit(Array(splitRatio, 1 - splitRatio))
    (training, test)
  }

  val seqSplit = (orderColumn: String) => (splitRatio: Double) => (df: DataFrame) => {
    val splitValue = df.stat.approxQuantile(orderColumn, Array(splitRatio), 0.001)(0)
    val headDf = df.where(df(orderColumn) <= splitValue)
    val tailDf = df.where(df(orderColumn) > splitValue)
    (headDf, tailDf)
  }

  val independentTestPredictions =
    (mlModel: Transformer, testDf: Dataset[_], _: Dataset[_]) => mlModel.transform(testDf)

  val orderDependentTestPredictions = (orderColumn: String) =>
    (mlModel: Transformer, testDf: Dataset[_], mainDf: Dataset[_]) => {
      val allPredictions = mlModel.transform(mainDf)
      val minTestIndexVal = testDf.agg(min(testDf(orderColumn))).head.getInt(0)
      allPredictions.where(allPredictions(orderColumn) >= minTestIndexVal)
    }

  val orderDependentTestPredictionsWithParams = (orderColumn: String) =>
    (mlModel: Transformer, testDf: Dataset[_], mainDf: Dataset[_], paramMap: ParamMap) => {
      val allPredictions = mlModel.transform(mainDf, paramMap)
      val minTestIndexVal = testDf.agg(min(testDf(orderColumn))).head.getInt(0)
      allPredictions.where(allPredictions(orderColumn) >= minTestIndexVal)
    }

  def calcMetricStats[T <: Enumeration#Value](results: Traversable[Performance[T]]): Map[T, (MetricStatsValues, MetricStatsValues, Option[MetricStatsValues])] =
    results.map { result =>
      val trainingStats = new SummaryStatistics
      val testStats = new SummaryStatistics
      val replicationStats = new SummaryStatistics

      result.trainingTestReplicationResults.foreach { case (trainValue, testValue, replicationValue) =>
        trainingStats.addValue(trainValue)
        testStats.addValue(testValue)
        if (replicationValue.isDefined)
          replicationStats.addValue(replicationValue.get)
      }

      val sortedTrainValues = result.trainingTestReplicationResults.map(_._1).toSeq.sorted
      val sortedTestValues = result.trainingTestReplicationResults.map(_._2).toSeq.sorted
      val sortedReplicationValues = result.trainingTestReplicationResults.flatMap(_._3).toSeq.sorted

      (result.evalMetric, (
        toStats(trainingStats, median(sortedTrainValues)),
        toStats(testStats, median(sortedTestValues)),
        if (replicationStats.getN > 0) Some(toStats(replicationStats, median(sortedReplicationValues))) else None
      ))
    }.toMap

  def median(seq: Seq[Double]): Double = {
    val middle = seq.size / 2
    if (seq.size % 2 == 1)
      seq(middle)
    else {
      val med1 = seq(middle - 1)
      val med2 = seq(middle)
      (med1 + med2) /2
    }
  }

  def toStats(summaryStatistics: SummaryStatistics, median: Double) =
    MetricStatsValues(summaryStatistics.getMean, summaryStatistics.getMin, summaryStatistics.getMax, summaryStatistics.getVariance, Some(median))

  def createClassificationResult(
    setting: ClassificationSetting,
    results: Traversable[ClassificationPerformance],
    binCurves: Traversable[STuple3[Option[BinaryClassificationCurves]]]
  ): ClassificationResult =
    createClassificationResult(
      setting,
      calcMetricStats(results),
      binCurves
    )

  def createClassificationResult(
    setting: ClassificationSetting,
    evalMetricStatsMap: Map[ClassificationEvalMetric.Value, (MetricStatsValues, MetricStatsValues, Option[MetricStatsValues])],
    binCurves: Traversable[STuple3[Option[BinaryClassificationCurves]]]
  ): ClassificationResult = {
    // helper functions
    def trainingStatsOptional(metric: ClassificationEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._1)

    def testStatsOptional(metric: ClassificationEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._2)

    def replicationStatsOptional(metric: ClassificationEvalMetric.Value) =
      evalMetricStatsMap.get(metric).flatMap(_._3)

    def trainingStats(metric: ClassificationEvalMetric.Value) =
      trainingStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Classification training stats for metrics '${metric.toString}' not found.")
      )

    def testStats(metric: ClassificationEvalMetric.Value) =
      testStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Classification test stats for metrics '${metric.toString}' not found.")
      )

    def replicationStats(metric: ClassificationEvalMetric.Value) =
      replicationStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Classification replication stats for metrics '${metric.toString}' not found.")
      )

    import ClassificationEvalMetric._

    val trainingMetricStats = ClassificationMetricStats(
      f1 = trainingStats(f1),
      weightedPrecision = trainingStats(weightedPrecision),
      weightedRecall = trainingStats(weightedRecall),
      accuracy = trainingStats(accuracy),
      areaUnderROC = trainingStatsOptional(areaUnderROC),
      areaUnderPR = trainingStatsOptional(areaUnderPR)
    )

    val testMetricStats = ClassificationMetricStats(
      f1 = testStats(f1),
      weightedPrecision = testStats(weightedPrecision),
      weightedRecall = testStats(weightedRecall),
      accuracy = testStats(accuracy),
      areaUnderROC = testStatsOptional(areaUnderROC),
      areaUnderPR = testStatsOptional(areaUnderPR)
    )

    val replicationMetricStats =
    // we assume if accuracy is defined the rest is fine, otherwise nothing is defined
      if (replicationStatsOptional(accuracy).isDefined)
        Some(
          ClassificationMetricStats(
            f1 = replicationStats(f1),
            weightedPrecision = replicationStats(weightedPrecision),
            weightedRecall = replicationStats(weightedRecall),
            accuracy = replicationStats(accuracy),
            areaUnderROC = replicationStatsOptional(areaUnderROC),
            areaUnderPR = replicationStatsOptional(areaUnderPR)
          )
        )
      else
        None

    val binCurvesSeq = binCurves.toSeq

    ClassificationResult(
      None,
      setting.copy(inputFieldNames = setting.inputFieldNames.sorted),
      trainingMetricStats,
      testMetricStats,
      replicationMetricStats,
      binCurvesSeq.flatMap(_._1),
      binCurvesSeq.flatMap(_._2),
      binCurvesSeq.flatMap(_._3)
    )
  }

  def createRegressionResult(
    setting: RegressionSetting,
    results: Traversable[RegressionPerformance]
  ): RegressionResult =
    createRegressionResult(
      setting,
      calcMetricStats(results)
    )

  def createRegressionResult(
    setting: RegressionSetting,
    evalMetricStatsMap: Map[RegressionEvalMetric.Value, (MetricStatsValues, MetricStatsValues, Option[MetricStatsValues])]
  ): RegressionResult = {
    // helper functions
    def trainingStatsOptional(metric: RegressionEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._1)

    def testStatsOptional(metric: RegressionEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._2)

    def replicationStatsOptional(metric: RegressionEvalMetric.Value) =
      evalMetricStatsMap.get(metric).flatMap(_._3)

    def trainingStats(metric: RegressionEvalMetric.Value) =
      trainingStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Regression training stats for metrics '${metric.toString}' not found.")
      )

    def testStats(metric: RegressionEvalMetric.Value) =
      testStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Regression test stats for metrics '${metric.toString}' not found.")
      )

    def replicationStats(metric: RegressionEvalMetric.Value) =
      replicationStatsOptional(metric).getOrElse(
        throw new IncalSparkMLException(s"Regression replication stats for metrics '${metric.toString}' not found.")
      )

    import RegressionEvalMetric._

    val trainingMetricStats = RegressionMetricStats(
      mse = trainingStats(mse),
      rmse = trainingStats(rmse),
      r2 = trainingStats(r2),
      mae = trainingStats(mae)
    )

    val testMetricStats = RegressionMetricStats(
      mse = testStats(mse),
      rmse = testStats(rmse),
      r2 = testStats(r2),
      mae = testStats(mae)
    )

    val replicationMetricStats =
    // we assume if mse is defined the rest is fine, otherwise nothing is defined
      if (replicationStatsOptional(mse).isDefined)
        Some(
          RegressionMetricStats(
            mse = replicationStats(mse),
            rmse = replicationStats(rmse),
            r2 = replicationStats(r2),
            mae = replicationStats(mae)
          )
        )
      else
        None

    RegressionResult(
      None,
      setting.copy(inputFieldNames = setting.inputFieldNames.sorted),
      trainingMetricStats,
      testMetricStats,
      replicationMetricStats
    )
  }
}