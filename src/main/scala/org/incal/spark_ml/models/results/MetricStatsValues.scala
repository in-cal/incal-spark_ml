package org.incal.spark_ml.models.results

case class MetricStatsValues(
  mean: Double,
  min: Double,
  max: Double,
  variance: Double,
  median: Option[Double]
)