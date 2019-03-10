package org.incal.spark_ml

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ForwardChainingCrossValidator}
import org.apache.spark.ml.Estimator
import org.apache.spark.sql.DataFrame

object CrossValidatorFactory {

  type CrossValidatorCreator = (Estimator[_], Array[ParamMap], Evaluator) => Estimator[_]
  type CrossValidatorCreatorWithProcessor = Option[DataFrame => DataFrame] => CrossValidatorCreator

  def withFolds(
    folds: Int
  ): CrossValidatorCreatorWithProcessor =
    _ => (trainer: Estimator[_], paramMaps: Array[ParamMap], crossValidationEvaluator: Evaluator) => {
      new CrossValidator()
        .setEstimator(trainer)
        .setEstimatorParamMaps(paramMaps)
        .setEvaluator(crossValidationEvaluator)
        .setNumFolds(folds)
    }

  def withForwardChaining(
    orderCol: String,
    minTrainingSizeRatio: Option[Double])(
    folds: Int
  ): CrossValidatorCreatorWithProcessor =
    (predictionsProcessor: Option[DataFrame => DataFrame]) => (trainer: Estimator[_], paramMaps: Array[ParamMap], crossValidationEvaluator: Evaluator) => {
      val validator = new ForwardChainingCrossValidator()
        .setEstimator(trainer)
        .setEstimatorParamMaps(paramMaps)
        .setEvaluator(crossValidationEvaluator)
        .setNumFolds(folds)
        .setOrderCol(orderCol)
        .setMinTrainingSizeRatio(minTrainingSizeRatio.getOrElse(1d / (folds + 1)))

      predictionsProcessor.foreach(validator.setPredictionsProcessor)

      validator
    }
}