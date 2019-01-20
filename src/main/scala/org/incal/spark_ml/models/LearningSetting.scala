package org.incal.spark_ml.models

case class LearningSetting[T](
  featuresNormalizationType: Option[VectorScalerType.Value] = None,
  pcaDims: Option[Int] = None,
  trainingTestSplitRatio: Option[Double] = None,
  samplingRatios: Seq[(String, Double)] = Nil,
  repetitions: Option[Int] = None,
  crossValidationFolds: Option[Int] = None,
  crossValidationEvalMetric: Option[T] = None
)