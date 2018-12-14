package org.incal.spark_ml.models.results

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.models.LearningSetting
import org.incal.spark_ml.models.regression.RegressionEvalMetric

case class RegressionResult(
  _id: Option[BSONObjectID],
  setting: RegressionSetting,
  trainingStats: RegressionMetricStats,
  testStats: RegressionMetricStats,
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
)

case class RegressionMetricStats(
  mse: MetricStatsValues,
  rmse: MetricStatsValues,
  r2: MetricStatsValues,
  mae: MetricStatsValues
)

case class RegressionSetting(
  mlModelId: BSONObjectID,
  outputFieldName: String,
  inputFieldNames: Seq[String],
  filterId: Option[BSONObjectID],
  featuresNormalizationType: Option[VectorScalerType.Value],
//  featuresSelectionNum: Option[Int],
  pcaDims: Option[Int],
  trainingTestingSplit: Option[Double],
  replicationFilterId: Option[BSONObjectID],
//  samplingRatios: Seq[(String, Double)],
  repetitions: Option[Int],
  crossValidationFolds: Option[Int],
  crossValidationEvalMetric: Option[RegressionEvalMetric.Value]
) {
  def fieldNamesToLoads =
    if (inputFieldNames.nonEmpty) (inputFieldNames ++ Seq(outputFieldName)).toSet.toSeq else Nil

  def learningSetting =
    LearningSetting[RegressionEvalMetric.Value](featuresNormalizationType, pcaDims, trainingTestingSplit, Nil, repetitions, crossValidationFolds, crossValidationEvalMetric)
}