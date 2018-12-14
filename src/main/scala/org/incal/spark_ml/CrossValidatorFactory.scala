package org.incal.spark_ml

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ForwardChainingCrossValidator}
import org.apache.spark.ml.Estimator

object CrossValidatorFactory {

  type CrossValidatorCreator = (Estimator[_], Array[ParamMap], Evaluator) => Estimator[_]

  def withFolds(
    folds: Int
  ): CrossValidatorCreator =
    (trainer: Estimator[_], paramMaps: Array[ParamMap], crossValidationEvaluator: Evaluator) =>
      new CrossValidator()
        .setEstimator(trainer)
        .setEstimatorParamMaps(paramMaps)
        .setEvaluator(crossValidationEvaluator)
        .setNumFolds(folds)

  def withForwardChaining(
    orderCol: String,
    minTrainingSize: Option[Double])(
    folds: Int
  ): CrossValidatorCreator =
    (trainer: Estimator[_], paramMaps: Array[ParamMap], crossValidationEvaluator: Evaluator) =>
      new ForwardChainingCrossValidator()
        .setEstimator(trainer)
        .setEstimatorParamMaps(paramMaps)
        .setEvaluator(crossValidationEvaluator)
        .setNumFolds(folds)
        .setOrderCol(orderCol)
        .setMinTrainingSize(minTrainingSize.getOrElse(1d / (folds + 1)))
}