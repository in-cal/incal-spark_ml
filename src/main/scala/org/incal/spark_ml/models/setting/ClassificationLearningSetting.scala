package org.incal.spark_ml.models.setting

import org.incal.spark_ml.models.{ReservoirSpec, VectorScalerType}
import org.incal.spark_ml.models.classification.{ClassificationEvalMetric, Classifier}

case class ClassificationLearningSetting(
  featuresNormalizationType: Option[VectorScalerType.Value] = None,
  featuresSelectionNum: Option[Int] = None,
  pcaDims: Option[Int] = None,
  trainingTestSplitRatio: Option[Double] = None,
  samplingRatios: Seq[(String, Double)] = Nil,
  repetitions: Option[Int] = None,
  crossValidationFolds: Option[Int] = None,
  crossValidationEvalMetric: Option[ClassificationEvalMetric.Value] = None,
  binCurvesNumBins: Option[Int] = None
) extends LearningSetting[ClassificationEvalMetric.Value]

case class TemporalClassificationLearningSetting(
  core: ClassificationLearningSetting = ClassificationLearningSetting(),
  predictAhead: Int = 1,
  slidingWindowSize: Option[Int] = None,
  reservoirSetting: Option[ReservoirSpec] = None,
  minCrossValidationTrainingSizeRatio: Option[Double] = None,
  trainingTestSplitOrderValue: Option[Double] = None,
  groupIdColumnName: Option[String] = None
) extends TemporalLearningSetting