package org.incal.spark_ml.models.result

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.setting.{RegressionRunSpec, RunSpec, TemporalRegressionRunSpec}

case class RegressionResult(
  _id: Option[BSONObjectID],
  runSpec: RegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractRegressionResult[RegressionRunSpec]

case class TemporalRegressionResult(
  _id: Option[BSONObjectID],
  runSpec: TemporalRegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends AbstractRegressionResult[TemporalRegressionRunSpec]

trait AbstractRegressionResult[T <: RunSpec] extends AbstractResult[T] {
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