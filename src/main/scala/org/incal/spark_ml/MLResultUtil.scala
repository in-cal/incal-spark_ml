package org.incal.spark_ml

import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.incal.spark_ml.models.result._
import org.incal.spark_ml.models.result.ClassificationConstructors._
import org.incal.spark_ml.models.result.RegressionConstructors._
import org.incal.spark_ml.models.classification.ClassificationEvalMetric
import org.incal.spark_ml.models.regression.RegressionEvalMetric
import org.incal.core.util.STuple3
import org.incal.spark_ml.models.setting.{ClassificationRunSpec, RegressionRunSpec, TemporalClassificationRunSpec, TemporalRegressionRunSpec}

object MLResultUtil {

  ////////////////////
  // Classification //
  ////////////////////

  type ClassificationEvalMetricStatsMap = Map[ClassificationEvalMetric.Value, (MetricStatsValues, Option[MetricStatsValues], Option[MetricStatsValues])]

  def createTemporalClassificationResult(
    runSpec: TemporalClassificationRunSpec,
    evalMetricStatsMap: ClassificationEvalMetricStatsMap,
    binCurves: Traversable[STuple3[Option[BinaryClassificationCurves]]]
  ): TemporalClassificationResult = {
    val specWithSortedFields = runSpec.copy(ioSpec = runSpec.ioSpec.copy(inputFieldNames = runSpec.ioSpec.inputFieldNames.sorted))
    createClassificationResult[TemporalClassificationResult](specWithSortedFields, evalMetricStatsMap, binCurves)
  }

  def createStandardClassificationResult(
    runSpec: ClassificationRunSpec,
    evalMetricStatsMap: ClassificationEvalMetricStatsMap,
    binCurves: Traversable[STuple3[Option[BinaryClassificationCurves]]]
  ): StandardClassificationResult = {
    val specWithSortedFields = runSpec.copy(ioSpec = runSpec.ioSpec.copy(inputFieldNames = runSpec.ioSpec.inputFieldNames.sorted))
    createClassificationResult[StandardClassificationResult](specWithSortedFields, evalMetricStatsMap, binCurves)
  }

  protected def createClassificationResult[C <: ClassificationResult](
    runSpec: C#R,
    evalMetricStatsMap: ClassificationEvalMetricStatsMap,
    binCurves: Traversable[STuple3[Option[BinaryClassificationCurves]]])(
    implicit constructor: ClassificationResultConstructor[C]
  ): C = {
    // helper functions
    def trainingStatsOptional(metric: ClassificationEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._1)

    def testStatsOptional(metric: ClassificationEvalMetric.Value) =
      evalMetricStatsMap.get(metric).flatMap(_._2)

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

    val testMetricStats =
      // we assume if accuracy is defined the rest is fine, otherwise nothing is defined
      if (testStatsOptional(accuracy).isDefined)
        Some(
          ClassificationMetricStats(
            f1 = testStats(f1),
            weightedPrecision = testStats(weightedPrecision),
            weightedRecall = testStats(weightedRecall),
            accuracy = testStats(accuracy),
            areaUnderROC = testStatsOptional(areaUnderROC),
            areaUnderPR = testStatsOptional(areaUnderPR)
          )
        )
      else
        None

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

    constructor.apply(
      runSpec,
      trainingMetricStats,
      testMetricStats,
      replicationMetricStats,
      binCurvesSeq.flatMap(_._1),
      binCurvesSeq.flatMap(_._2),
      binCurvesSeq.flatMap(_._3)
    )
  }

  ////////////////
  // Regression //
  ////////////////

  type RegressionEvalMetricStatsMap = Map[RegressionEvalMetric.Value, (MetricStatsValues, Option[MetricStatsValues], Option[MetricStatsValues])]

  def createTemporalRegressionResult(
    runSpec: TemporalRegressionRunSpec,
    evalMetricStatsMap: RegressionEvalMetricStatsMap
  ): TemporalRegressionResult = {
    val specWithSortedFields = runSpec.copy(ioSpec = runSpec.ioSpec.copy(inputFieldNames = runSpec.ioSpec.inputFieldNames.sorted))
    createRegressionResult[TemporalRegressionResult](specWithSortedFields, evalMetricStatsMap)
  }

  def createStandardRegressionResult(
    runSpec: RegressionRunSpec,
    evalMetricStatsMap: RegressionEvalMetricStatsMap
  ): StandardRegressionResult = {
    val specWithSortedFields = runSpec.copy(ioSpec = runSpec.ioSpec.copy(inputFieldNames = runSpec.ioSpec.inputFieldNames.sorted))
    createRegressionResult[StandardRegressionResult](specWithSortedFields, evalMetricStatsMap)
  }

  protected def createRegressionResult[C <: RegressionResult](
    runSpec: C#R,
    evalMetricStatsMap: RegressionEvalMetricStatsMap)(
    implicit constructor: RegressionResultConstructor[C]
  ): C = {
    // helper functions
    def trainingStatsOptional(metric: RegressionEvalMetric.Value) =
      evalMetricStatsMap.get(metric).map(_._1)

    def testStatsOptional(metric: RegressionEvalMetric.Value) =
      evalMetricStatsMap.get(metric).flatMap(_._2)

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

    val testMetricStats =
      // we assume if mse is defined the rest is fine, otherwise nothing is defined
      if (testStatsOptional(mse).isDefined)
        Some(
          RegressionMetricStats(
            mse = testStats(mse),
            rmse = testStats(rmse),
            r2 = testStats(r2),
            mae = testStats(mae)
          )
        )
      else
        None

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

    constructor.apply(
      runSpec,
      trainingMetricStats,
      testMetricStats,
      replicationMetricStats
    )
  }

  // Aux

  def calcMetricStats[T <: Enumeration#Value](results: Traversable[Performance[T]]): Map[T, (MetricStatsValues, Option[MetricStatsValues], Option[MetricStatsValues])] =
    results.map { result =>
      val trainingStats = new SummaryStatistics
      val testStats = new SummaryStatistics
      val replicationStats = new SummaryStatistics

      result.trainingTestReplicationResults.foreach { case (trainValue, testValue, replicationValue) =>
        trainingStats.addValue(trainValue)
        if (testValue.isDefined)
          testStats.addValue(testValue.get)
        if (replicationValue.isDefined)
          replicationStats.addValue(replicationValue.get)
      }

      val sortedTrainValues = result.trainingTestReplicationResults.map(_._1).toSeq.sorted
      val sortedTestValues = result.trainingTestReplicationResults.flatMap(_._2).toSeq.sorted
      val sortedReplicationValues = result.trainingTestReplicationResults.flatMap(_._3).toSeq.sorted

      (result.evalMetric, (
        toStats(trainingStats, median(sortedTrainValues)),
        if (testStats.getN > 0) Some(toStats(testStats, median(sortedTestValues))) else None,
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
}