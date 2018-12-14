package org.incal.spark_ml.models.results

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.models.LearningSetting
import org.incal.spark_ml.models.classification.ClassificationEvalMetric

case class ClassificationResult(
  _id: Option[BSONObjectID],
  setting: ClassificationSetting,
  trainingStats: ClassificationMetricStats,
  testStats: ClassificationMetricStats,
  replicationStats: Option[ClassificationMetricStats] = None,
  trainingBinCurves: Seq[BinaryClassificationCurves] = Nil,
  testBinCurves: Seq[BinaryClassificationCurves] = Nil,
  replicationBinCurves: Seq[BinaryClassificationCurves] = Nil,
  timeCreated: ju.Date = new ju.Date()
)

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

case class ClassificationSetting(
  mlModelId: BSONObjectID,
  outputFieldName: String,
  inputFieldNames: Seq[String],
  filterId: Option[BSONObjectID],
  featuresNormalizationType: Option[VectorScalerType.Value],
  featuresSelectionNum: Option[Int],
  pcaDims: Option[Int],
  trainingTestingSplit: Option[Double],
  replicationFilterId: Option[BSONObjectID],
  samplingRatios: Seq[(String, Double)],
  repetitions: Option[Int],
  crossValidationFolds: Option[Int],
  crossValidationEvalMetric: Option[ClassificationEvalMetric.Value],
  binCurvesNumBins: Option[Int]
) {
  def fieldNamesToLoads =
    if (inputFieldNames.nonEmpty) (inputFieldNames ++ Seq(outputFieldName)).toSet.toSeq else Nil

  def learningSetting =
    LearningSetting[ClassificationEvalMetric.Value](featuresNormalizationType, pcaDims, trainingTestingSplit, samplingRatios, repetitions, crossValidationFolds, crossValidationEvalMetric)
}