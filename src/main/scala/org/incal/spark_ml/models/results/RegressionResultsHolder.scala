package org.incal.spark_ml.models.results

import org.incal.spark_ml.models.regression.RegressionEvalMetric

case class RegressionResultsHolder(
  performanceResults: Traversable[RegressionPerformance],
  counts: Traversable[Long],
  expectedAndActualOutputs: Traversable[Traversable[Seq[(Double, Double)]]]
)

case class RegressionResultsAuxHolder(
  evalResults: Traversable[(RegressionEvalMetric.Value, Double, Seq[Double])],
  count: Long,
  expectedAndActualOutputs: Traversable[Seq[(Double, Double)]]
)
