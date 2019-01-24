package org.incal.spark_ml.models.result

case class MetricStatsValues(
  mean: Double,
  min: Double,
  max: Double,
  variance: Double,
  median: Option[Double]
)