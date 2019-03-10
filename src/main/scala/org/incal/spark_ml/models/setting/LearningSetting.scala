package org.incal.spark_ml.models.setting

import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import org.incal.spark_ml.models.{ReservoirSpec, VectorScalerType}

trait LearningSetting[T] {
  val featuresNormalizationType: Option[VectorScalerType.Value]
  val pcaDims: Option[Int]
  val trainingTestSplitRatio: Option[Double]
  val repetitions: Option[Int]
  val crossValidationFolds: Option[Int]
  val crossValidationEvalMetric: Option[T]
}

trait TemporalLearningSetting {
  val predictAhead: Int
  val slidingWindowSize: ValueOrSeq[Int]
  val reservoirSetting: Option[ReservoirSpec]
  val minCrossValidationTrainingSizeRatio: Option[Double]
  val trainingTestSplitOrderValue: Option[Double]
}