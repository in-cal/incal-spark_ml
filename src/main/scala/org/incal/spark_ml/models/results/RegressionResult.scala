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
  // IO
  inputFieldNames: Seq[String],
  outputFieldName: String,
  filterId: Option[BSONObjectID] = None,
  replicationFilterId: Option[BSONObjectID] = None,

  // Learning setting
  mlModelId: BSONObjectID,
  featuresNormalizationType: Option[VectorScalerType.Value] = None,
  outputNormalizationType: Option[VectorScalerType.Value] = None,
  pcaDims: Option[Int],
  trainingTestSplitRatio: Option[Double] = None,
  trainingTestSplitOrderValue: Option[Double] = None,
  repetitions: Option[Int] = None,
  crossValidationFolds: Option[Int] = None,
  crossValidationEvalMetric: Option[RegressionEvalMetric.Value] = None
) {
  def fieldNamesToLoads =
    if (inputFieldNames.nonEmpty) (inputFieldNames ++ Seq(outputFieldName)).toSet.toSeq else Nil

  def learningSetting =
    LearningSetting[RegressionEvalMetric.Value](featuresNormalizationType, pcaDims, trainingTestSplitRatio, Nil, repetitions, crossValidationFolds, crossValidationEvalMetric)
}