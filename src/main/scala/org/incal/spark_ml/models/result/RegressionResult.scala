package org.incal.spark_ml.models.result

import java.{util => ju}

import reactivemongo.bson.BSONObjectID
import org.incal.spark_ml.models.setting.{RegressionRunSpec, TemporalRegressionRunSpec}

case class StandardRegressionResult(
  _id: Option[BSONObjectID],
  runSpec: RegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends RegressionResult { type R = RegressionRunSpec }

case class TemporalRegressionResult(
  _id: Option[BSONObjectID],
  runSpec: TemporalRegressionRunSpec,
  trainingStats: RegressionMetricStats,
  testStats: Option[RegressionMetricStats],
  replicationStats: Option[RegressionMetricStats] = None,
  timeCreated: ju.Date = new ju.Date()
) extends RegressionResult { type R = TemporalRegressionRunSpec }

trait RegressionResult extends MLResult {
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

// handy constructors

trait RegressionResultConstructor[C <: RegressionResult] {

  def apply: (
    C#R,
    RegressionMetricStats,
    Option[RegressionMetricStats],
    Option[RegressionMetricStats]
  ) => C
}

object RegressionConstructors {

  implicit object StandardRegression extends RegressionResultConstructor[StandardRegressionResult] {
    override def apply = StandardRegressionResult(None, _, _, _, _)
  }

  implicit object TemporalRegression extends RegressionResultConstructor[TemporalRegressionResult] {
    override def apply = TemporalRegressionResult(None, _, _, _, _)
  }
}