package org.incal.spark_ml

import org.incal.spark_ml.models.classification.{Classification, DecisionTree, GradientBoostTree, LinearSupportVectorMachine, LogisticRegression, MultiLayerPerceptron, NaiveBayes, RandomForest}
import org.incal.spark_ml.models.regression.{Regression, GeneralizedLinearRegression => GeneralizedLinearRegressionDef, GradientBoostRegressionTree => GradientBoostRegressionTreeDef, LinearRegression => LinearRegressionDef, RandomRegressionForest => RandomRegressionForestDef, RegressionTree => RegressionTreeDef}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.classification.{LogisticRegression => LogisticRegressionClassifier, NaiveBayes => NaiveBayesClassifier, _}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, RandomForestRegressor, GeneralizedLinearRegression => GeneralizedLinearRegressor, LinearRegression => LinearRegressor}
import org.apache.spark.ml.param._

import org.incal.spark_ml.models.ValueOrSeq._
import org.incal.spark_ml.{ParamGrid, ParamSourceBinder}

object SparkMLEstimatorFactory extends SparkMLEstimatorFactoryHelper {

  def apply[M <: Model[M]](
    model: Classification,
    inputSize: Int,
    outputSize: Int
  ): (Estimator[M], Traversable[ParamGrid[_]]) = {
    val (estimator, paramMaps) = model match {
      case x: LogisticRegression => applyAux(x)
      case x: MultiLayerPerceptron => applyAux(x, inputSize, outputSize)
      case x: DecisionTree => applyAux(x)
      case x: RandomForest => applyAux(x)
      case x: GradientBoostTree => applyAux(x)
      case x: NaiveBayes => applyAux(x)
      case x: LinearSupportVectorMachine => applyAux(x)
    }

    (estimator.asInstanceOf[Estimator[M]], paramMaps)
  }

  def apply[M <: Model[M]](
    model: Regression
  ): (Estimator[M], Traversable[ParamGrid[_]]) = {
    val (estimator, paramMaps) = model match {
      case x: LinearRegressionDef => applyAux(x)
      case x: GeneralizedLinearRegressionDef => applyAux(x)
      case x: RegressionTreeDef => applyAux(x)
      case x: RandomRegressionForestDef => applyAux(x)
      case x: GradientBoostRegressionTreeDef => applyAux(x)
    }

    (estimator.asInstanceOf[Estimator[M]], paramMaps)
  }

  private def applyAux(
    model: LogisticRegression
  ): (LogisticRegressionClassifier, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new LogisticRegressionClassifier())
      .bindValOrSeq(_.aggregationDepth, "aggregationDepth")
      .bindValOrSeq(_.elasticMixingRatio, "elasticNetParam")
      .bind(_.family.map(_.toString), "family")
      .bind(_.fitIntercept, "fitIntercept")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.regularization, "regParam")
      .bindValOrSeq(_.threshold, "threshold")
      .bind(_.thresholds.map(_.toArray), "thresholds")
      .bind(_.standardization, "standardization")
      .bindValOrSeq(_.tolerance, "tol")
      .build

  private def applyAux(
    model: MultiLayerPerceptron,
    inputSize: Int,
    outputSize: Int
  ): (MultilayerPerceptronClassifier, Traversable[ParamGrid[_]]) = {
    val layers = (Seq(inputSize) ++ model.hiddenLayers ++ Seq(outputSize)).toArray

    ParamSourceBinder(model, new MultilayerPerceptronClassifier())
      .bindValOrSeq(_.blockSize, "blockSize")
      .bind(_.seed, "seed")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bind(_.solver.map(_.toString), "solver")
      .bindValOrSeq(_.stepSize, "stepSize")
      .bindValOrSeq(_.tolerance, "tol")
      .bind(o => Some(layers), "layers")
      .build
  }

  private def applyAux(
    model: DecisionTree
  ): (DecisionTreeClassifier, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new DecisionTreeClassifier())
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bind(_.impurity.map(_.toString), "impurity")
      .build

  private def applyAux(
    model: RandomForest
  ): (RandomForestClassifier, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new RandomForestClassifier())
      .bindValOrSeq(_.numTrees, "numTrees")
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bindValOrSeq(_.subsamplingRate, "subsamplingRate")
      .bind(_.impurity.map(_.toString), "impurity")
      .bind(_.featureSubsetStrategy.map(_.toString), "featureSubsetStrategy")
      .build

  private def applyAux(
    model: GradientBoostTree
  ): (GBTClassifier, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new GBTClassifier())
      .bind(_.lossType.map(_.toString), "lossType")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.stepSize, "stepSize")
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bindValOrSeq(_.subsamplingRate, "subsamplingRate")
      //    .bind(_.impurity.map(_.toString), "impurity")
      .build

  private def applyAux(
    model: NaiveBayes
  ): (NaiveBayesClassifier, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new NaiveBayesClassifier())
      .bindValOrSeq(_.smoothing, "smoothing")
      .bind(_.modelType.map(_.toString), "modelType")
      .build

  private def applyAux(
    model: LinearSupportVectorMachine
  ): (LinearSVC, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new LinearSVC())
      .bindValOrSeq(_.aggregationDepth, "aggregationDepth")
      .bind(_.fitIntercept, "fitIntercept")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.regularization, "regParam")
      .bind(_.standardization, "standardization")
      .bindValOrSeq(_.threshold, "threshold")
      .bindValOrSeq(_.tolerance, "tol")
      .build

  private def applyAux(
    model: LinearRegressionDef
  ): (LinearRegressor, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new LinearRegressor())
      .bindValOrSeq(_.aggregationDepth, "aggregationDepth")
      .bindValOrSeq(_.elasticMixingRatio, "elasticNetParam")
      .bind(_.solver.map(_.toString), "solver")
      .bind(_.fitIntercept, "fitIntercept")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.regularization, "regParam")
      .bind(_.standardization, "standardization")
      .bindValOrSeq(_.tolerance, "tol")
      .build

  private def applyAux(
    model: GeneralizedLinearRegressionDef
  ): (GeneralizedLinearRegressor, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new GeneralizedLinearRegressor())
      .bind(_.solver.map(_.toString), "solver")
      .bind(_.family.map(_.toString), "family")
      .bind(_.fitIntercept, "fitIntercept")
      .bind(_.link.map(_.toString), "link")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.regularization, "regParam")
      .bindValOrSeq(_.tolerance, "tol")
      .build

  private def applyAux(
    model: RegressionTreeDef
  ): (DecisionTreeRegressor, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new DecisionTreeRegressor())
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bind(_.impurity.map(_.toString), "impurity")
      .build

  private def applyAux(
    model: RandomRegressionForestDef
  ): (RandomForestRegressor, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new RandomForestRegressor())
      .bind(_.impurity.map(_.toString), "impurity")
      .bindValOrSeq(_.numTrees, "numTrees")
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bindValOrSeq(_.subsamplingRate, "subsamplingRate")
      .bind(_.featureSubsetStrategy.map(_.toString), "featureSubsetStrategy")
      .build

  private def applyAux(
    model: GradientBoostRegressionTreeDef
  ): (GBTRegressor, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(model, new GBTRegressor())
      .bind(_.lossType.map(_.toString), "lossType")
      .bindValOrSeq(_.maxIteration, "maxIter")
      .bindValOrSeq(_.stepSize, "stepSize")
      .bindValOrSeq(_.core.maxDepth, "maxDepth")
      .bindValOrSeq(_.core.maxBins, "maxBins")
      .bindValOrSeq(_.core.minInstancesPerNode, "minInstancesPerNode")
      .bindValOrSeq(_.core.minInfoGain, "minInfoGain")
      .bind(_.core.seed, "seed")
      .bindValOrSeq(_.subsamplingRate, "subsamplingRate")
      //    .bind(_.impurity.map(_.toString), "impurity")
      .build
}

trait SparkMLEstimatorFactoryHelper {

  // helper functions

  protected def setParam[T, M](
    paramValue: Option[T],
    setModelParam: M => (T => M))(
    model: M
  ): M =
    paramValue.map(setModelParam(model)).getOrElse(model)

  protected def setSourceParam[T, S, M](
    source: S)(
    getParamValue: S => Option[T],
    setParamValue: M => (T => M))(
    target: M
  ): M =
    setParam(getParamValue(source), setParamValue)(target)

  protected def chain[T](trans: (T => T)*)(init: T) =
    trans.foldLeft(init){case (a, trans) => trans(a)}
}