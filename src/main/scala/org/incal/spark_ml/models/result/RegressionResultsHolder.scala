package org.incal.spark_ml.models.result

import org.incal.core.util.STuple3
import org.incal.spark_ml.models.regression.RegressionEvalMetric

case class RegressionResultsHolder(
  performanceResults: Traversable[RegressionPerformance],
  counts: Traversable[Long],
  expectedActualOutputs: Traversable[STuple3[Seq[(Double, Double)]]]
)

case class RegressionResultsAuxHolder(
  evalResults: Traversable[(RegressionEvalMetric.Value, Double, Seq[Double])],
  count: Long,
  expectedActualOutputs: STuple3[Seq[(Double, Double)]]
)
