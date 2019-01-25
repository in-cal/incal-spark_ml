package org.incal.spark_ml.models.result

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.setting.{ClassificationRunSpec, RunSpec, TemporalClassificationRunSpec}

case class ClassificationResult(
  _id: Option[BSONObjectID],
  runSpec: ClassificationRunSpec,
  trainingStats: ClassificationMetricStats,
  testStats: Option[ClassificationMetricStats],
  replicationStats: Option[ClassificationMetricStats] = None,
  trainingBinCurves: Seq[BinaryClassificationCurves] = Nil,
  testBinCurves: Seq[BinaryClassificationCurves] = Nil,
  replicationBinCurves: Seq[BinaryClassificationCurves] = Nil,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractClassificationResult[ClassificationRunSpec]

case class TemporalClassificationResult(
  _id: Option[BSONObjectID],
  runSpec: TemporalClassificationRunSpec,
  trainingStats: ClassificationMetricStats,
  testStats: Option[ClassificationMetricStats],
  replicationStats: Option[ClassificationMetricStats] = None,
  trainingBinCurves: Seq[BinaryClassificationCurves] = Nil,
  testBinCurves: Seq[BinaryClassificationCurves] = Nil,
  replicationBinCurves: Seq[BinaryClassificationCurves] = Nil,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractClassificationResult[TemporalClassificationRunSpec]

trait AbstractClassificationResult[T <: RunSpec] extends AbstractResult[T] {
  val trainingStats: ClassificationMetricStats
  val testStats: Option[ClassificationMetricStats]
  val replicationStats: Option[ClassificationMetricStats]
  val trainingBinCurves: Seq[BinaryClassificationCurves]
  val testBinCurves: Seq[BinaryClassificationCurves]
  val replicationBinCurves: Seq[BinaryClassificationCurves]
}

case class ClassificationMetricStats(
  f1: MetricStatsValues,
  weightedPrecision: MetricStatsValues,
  weightedRecall: MetricStatsValues,
  accuracy: MetricStatsValues,
  areaUnderROC: Option[MetricStatsValues],
  areaUnderPR: Option[MetricStatsValues]
)

case class BinaryClassificationCurves(
  // ROC - FPR vs TPR (false positive rate vs true positive rate)
  roc: Seq[(Double, Double)],
  // PR - recall vs precision
  precisionRecall: Seq[(Double, Double)],
  // threshold vs F-Measure: curve with beta = 1.0.
  fMeasureThreshold: Seq[(Double, Double)],
  // threshold vs precision
  precisionThreshold: Seq[(Double, Double)],
  // threshold vs recall
  recallThreshold: Seq[(Double, Double)]
)