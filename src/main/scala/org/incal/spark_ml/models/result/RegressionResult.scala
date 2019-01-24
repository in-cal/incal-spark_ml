package org.incal.spark_ml.models.result

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.setting.{RegressionRunSpec, TemporalRegressionRunSpec}

case class RegressionResult(
  _id: Option[BSONObjectID],
  spec: RegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractRegressionResult

case class TemporalRegressionResult(
  _id: Option[BSONObjectID],
  spec: TemporalRegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractRegressionResult

trait AbstractRegressionResult {
  val trainingStats: RegressionMetricStats
  val testStats: Option[RegressionMetricStats]
  val replicationStats: Option[RegressionMetricStats]
}

case class RegressionMetricStats(
  mse: MetricStatsValues,
  rmse: MetricStatsValues,
  r2: MetricStatsValues,
  mae: MetricStatsValues
)