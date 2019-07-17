package org.incal.spark_ml

import org.incal.spark_ml.models.classification.{Classifier, DecisionTree, GradientBoostTree, LinearSupportVectorMachine, LogisticRegression, MultiLayerPerceptron, NaiveBayes, RandomForest}
import org.incal.spark_ml.models.regression.{Regressor, GeneralizedLinearRegression => GeneralizedLinearRegressionDef, GradientBoostRegressionTree => GradientBoostRegressionTreeDef, LinearRegression => LinearRegressionDef, RandomRegressionForest => RandomRegressionForestDef, RegressionTree => RegressionTreeDef}
import org.apache.spark.ml.classification.{LogisticRegression => LogisticRegressionClassifier, NaiveBayes => NaiveBayesClassifier, _}
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans, LDA}
import org.incal.spark_ml.models.clustering.{Clustering, BisectingKMeans => BisectingKMeansDef, GaussianMixture => GaussianMixtureDef, KMeans => KMeansDef, LDA => LDADef}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans, LDA}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, RandomForestRegressor, GeneralizedLinearRegression => GeneralizedLinearRegressor, LinearRegression => LinearRegressor}
import org.apache.spark.ml.param._
import org.incal.spark_ml.models.ValueOrSeq._
import org.incal.spark_ml.models.clustering.Clustering

object SparkMLEstimatorFactory extends SparkMLEstimatorFactoryHelper {

  def apply[M <: Model[M]](
    model: Classifier,
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
    model: Regressor
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
//    println(s"Input: $inputSize, output: $outputSize")
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

  def apply[M <: Model[M]](
    model: Clustering
  ): Estimator[M] =
    model match {
      case x: KMeansDef => applyAux(x).asInstanceOf[Estimator[M]]
      case x: LDADef => applyAux(x).asInstanceOf[Estimator[M]]
      case x: BisectingKMeansDef => applyAux(x).asInstanceOf[Estimator[M]]
      case x: GaussianMixtureDef => applyAux(x).asInstanceOf[Estimator[M]]
    }

  private def applyAux(
    model: KMeansDef)
  : KMeans = {
    val (estimator, _) = ParamSourceBinder(model, new KMeans())
      .bind(_.initMode.map(_.toString), "initMode")
      .bind(_.initSteps, "initSteps")
      .bind({o => Some(o.k)}, "k")
      .bind(_.maxIteration, "maxIter")
      .bind(_.tolerance, "tol")
      .bind(_.seed, "seed")
      .build

    estimator
  }

  private def applyAux(
    model: LDADef
  ): LDA = {
    val (estimator, _) = ParamSourceBinder(model, new LDA())
      .bind(_.checkpointInterval, "checkpointInterval")
      .bind(_.keepLastCheckpoint, "keepLastCheckpoint")
      .bind[Array[Double]](_.docConcentration.map(_.toArray), "docConcentration")
      .bind(_.optimizeDocConcentration, "optimizeDocConcentration")
      .bind(_.topicConcentration, "topicConcentration")
      .bind({o => Some(o.k)}, "k")
      .bind(_.learningDecay, "learningDecay")
      .bind(_.learningOffset, "learningOffset")
      .bind(_.maxIteration, "maxIter")
      .bind(_.optimizer.map(_.toString), "optimizer")
      .bind(_.subsamplingRate, "subsamplingRate")
      .bind(_.seed, "seed")
      .build

    estimator
  }

  private def applyAux(
    model: BisectingKMeansDef
  ): BisectingKMeans = {
    val (estimator, _) = ParamSourceBinder(model, new BisectingKMeans())
      .bind({o => Some(o.k)}, "k")
      .bind(_.maxIteration, "maxIter")
      .bind(_.seed, "seed")
      .bind(_.minDivisibleClusterSize, "minDivisibleClusterSize")
      .build

    estimator
  }

  private def applyAux(
    model: GaussianMixtureDef
  ): GaussianMixture = {
    val (estimator, _) = ParamSourceBinder(model, new GaussianMixture())
      .bind({ o => Some(o.k) }, "k")
      .bind(_.maxIteration, "maxIter")
      .bind(_.tolerance, "tol")
      .bind(_.seed, "seed")
      .build

    estimator
  }
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