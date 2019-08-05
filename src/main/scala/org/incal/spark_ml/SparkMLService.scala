package org.incal.spark_ml

import org.apache.spark.sql.Encoders
import org.apache.spark.ml.clustering.{BisectingKMeansModel, GaussianMixtureModel, KMeansModel, LDAModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.{Model, util, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.slf4j.LoggerFactory
import org.incal.spark_ml.transformers._
import org.incal.spark_ml.models.classification.{ClassificationEvalMetric, Classifier}
import org.incal.spark_ml.models.regression.{RegressionEvalMetric, Regressor}
import org.incal.spark_ml.CrossValidatorFactory.{CrossValidatorCreator, CrossValidatorCreatorWithProcessor}
import org.incal.spark_ml.models.result._
import org.incal.core.util.{STuple3, parallelize}
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.models.clustering.Clustering
import org.incal.spark_ml.models.setting._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

trait SparkMLService extends MLBase {

  val rcStatesWindowFactory: RCStatesWindowFactory
  val setting: SparkMLServiceSetting

  protected val logger = LoggerFactory.getLogger("spark_ml")

  // consts
  protected val defaultTrainingTestingSplitRatio = 0.8
  protected val defaultClassificationCrossValidationEvalMetric = ClassificationEvalMetric.accuracy
  protected val defaultRegressionCrossValidationEvalMetric = RegressionEvalMetric.rmse
  protected val seriesOrderCol = "index"

  // settings
  protected lazy val repetitionParallelism = setting.repetitionParallelism.getOrElse(2)
  protected lazy val binaryClassifierInputName = setting.binaryClassifierInputName.getOrElse("probability")
  protected lazy val useConsecutiveOrderForDL = setting.useConsecutiveOrderForDL.getOrElse(false)
  protected lazy val binaryPredictionVectorizer = new IndexVectorizer() {
    setInputCol("prediction"); setOutputCol(binaryClassifierInputName)
  }

  private lazy val binClassificationEvaluators =
    Seq(ClassificationEvalMetric.areaUnderPR, ClassificationEvalMetric.areaUnderROC).map { metric =>
      val evaluator = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol(binaryClassifierInputName)
        .setMetricName(metric.toString)

      EvaluatorWrapper(
        metric,
        evaluator
      )
    }

  def classify(
    df: DataFrame,
    classifier: Classifier,
    setting: ClassificationLearningSetting = ClassificationLearningSetting(),
    replicationDf: Option[DataFrame] = None
  ): Future[ClassificationResultsHolder]  = {

    // k-folds cross validator
    val crossValidatorCreator = setting.crossValidationFolds.map(CrossValidatorFactory.withFolds)

    // data set training / test split
    val split = randomSplit(setting)

    // how to calculate test predictions
    val calcTestPredictions = independentTestPredictions

    // classify with a random split
    classifyAux(
      df, replicationDf, classifier, setting
    )(
      split, calcTestPredictions, crossValidatorCreator, Nil, Nil, Nil
    )
  }

  def classifyTimeSeries(
    df: DataFrame,
    classifier: Classifier,
    setting: TemporalClassificationLearningSetting,
    groupIdColumnName: Option[String] = None,
    replicationDf: Option[DataFrame] = None
  ): Future[ClassificationResultsHolder] = {

    // time series transformers/stages
    val (timeSeriesStages, paramGrids, kernelSize) = createTimeSeriesStagesWithParamGrids(groupIdColumnName, setting)

    // forward-chaining cross validator
    val crossValidatorCreator = setting.core.crossValidationFolds.map(
      CrossValidatorFactory.withForwardChaining(seriesOrderCol, setting.minCrossValidationTrainingSizeRatio)
    )

    // data set training / test split
    val split = seqSplit(setting.core.trainingTestSplitRatio, setting.trainingTestSplitOrderValue)

    // how to calculate test predictions
    val calcTestPredictions = orderDependentTestPredictions(seriesOrderCol)

    // classify with the time-series transformers and a sequential split
    classifyAux(
      df, replicationDf, classifier, setting.core
    )(
      split, calcTestPredictions, crossValidatorCreator, Nil, timeSeriesStages, paramGrids, kernelSize
    )
  }

  protected def classifyAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    classifier: Classifier,
    setting: ClassificationLearningSetting)(
    splitDataSet: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreatorWithProcessor: Option[CrossValidatorCreatorWithProcessor],
    initStages: Seq[() => PipelineStage],
    preTrainingStages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]] = Nil,
    kernelSize: Int => Int = identity
  ): Future[ClassificationResultsHolder] = {
    // stages
    val coreStages = classificationStages(setting)
    val stages = initStages ++ coreStages ++ preTrainingStages

    // classify with the stages
    classifyWithStages(
      df, replicationDf, classifier, setting
    )(
      splitDataSet, calcTestPredictions, crossValidatorCreatorWithProcessor, stages, paramGrids, kernelSize
    )
  }

  protected def classifyWithStages(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    classifier: Classifier,
    setting: ClassificationLearningSetting)(
    splitDataset: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreatorWithProcessor: Option[CrossValidatorCreatorWithProcessor],
    stages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]],
    kernelSize: Int => Int
  ): Future[ClassificationResultsHolder] = {

    // cache the data frames
    df.cache()
    if (replicationDf.isDefined)
      replicationDf.get.cache()

    // CREATE A TRAINER

    val originalFeaturesType = df.schema.fields.find(_.name == "features").get
    val originalInputSize = originalFeaturesType.metadata.getMetadata("ml_attr").getLong("num_attrs").toInt
    val inputSize = kernelSize(setting.pcaDims.getOrElse(originalInputSize))

    logger.debug(s"Input Size: ${inputSize}.")

    val outputLabelType = df.schema.fields.find(_.name == "label").get
    val outputSize = outputLabelType.metadata.getMetadata("ml_attr").getStringArray("vals").length

    val (trainer, trainerParamGrids) = SparkMLEstimatorFactory(classifier, inputSize, outputSize)
    val fullParamMaps = buildParamGrids(trainerParamGrids ++ paramGrids)

    // REPEAT THE TRAINING-TEST CYCLE

    // evaluators
    val evaluators = classificationEvaluators ++ (if (outputSize == 2) binClassificationEvaluators else Nil)

    // cross-validation evaluator
    val crossValidationEvaluator =
      setting.crossValidationEvalMetric.flatMap(metric =>
        evaluators.find(_.metric == metric)
      ).getOrElse(
        evaluators.find(_.metric == defaultClassificationCrossValidationEvalMetric).get
      )

    val count = df.count()

    val resultHoldersFuture = parallelize(1 to setting.repetitions.getOrElse(1), repetitionParallelism) { index =>
      logger.info(s"Execution of repetition $index started for $count rows.")

      val fullTrainer = new Pipeline().setStages((stages.map(_()) ++ Seq(trainer)).toArray)

      // classify and evaluate
      classifyAndEvaluate(
        fullTrainer,
        fullParamMaps,
        evaluators,
        crossValidationEvaluator.evaluator,
        crossValidatorCreatorWithProcessor,
        splitDataset,
        calcTestPredictions,
        outputSize,
        count,
        setting.binCurvesNumBins,
        setting.collectOutputs,
        df,
        replicationDf
      )
    }

    // CREATE FINAL PERFORMANCE RESULTS

    resultHoldersFuture.map { resultHolders =>
      // uncache
      df.unpersist
      if (replicationDf.isDefined)
        replicationDf.get.unpersist

      // create performance results
      val results = resultHolders.flatMap(_.evalResults)
      val performanceResults = results.groupBy(_._1).map { case (evalMetric, results) =>
        ClassificationPerformance(evalMetric, results.map { case (_, trainResult, testResults) =>
          testResults.headOption.map(testResult =>
            (trainResult, Some(testResult): Option[Double], testResults.tail.headOption)
          ).getOrElse(
            (trainResult, None, None)
          )
        })
      }

      // counts
      val counts = resultHolders.map(_.count)

      // curves
      val curves = resultHolders.map { resultHolder =>
        (
          resultHolder.binTrainingCurves,
          resultHolder.binTestCurves.head,
          resultHolder.binTestCurves.tail.headOption.flatten
        )
      }

      // actual vs expected outputs
      val expectedAndActualOutputs = resultHolders.map(_.expectedAndActualOutputs)

      ClassificationResultsHolder(performanceResults, counts, curves, expectedAndActualOutputs)
    }
  }

  def regress(
    df: DataFrame,
    regressor: Regressor,
    setting: RegressionLearningSetting = RegressionLearningSetting(),
    replicationDf: Option[DataFrame] = None
  ): Future[RegressionResultsHolder] = {

    // k-folds cross-validator
    val crossValidatorCreator = setting.crossValidationFolds.map(CrossValidatorFactory.withFolds)

    // data set training / test split
    val split = randomSplit(setting)

    // how to calculate test predictions
    val calcTestPredictions = independentTestPredictions

    // regress
    regressAux(
      df, replicationDf, regressor, setting
    )(
      split, calcTestPredictions, crossValidatorCreator, Nil, Nil, Nil
    )
  }

  def regressTimeSeries(
    df: DataFrame,
    regressor: Regressor,
    setting: TemporalRegressionLearningSetting,
    groupIdColumnName: Option[String] = None,
    replicationDf: Option[DataFrame] = None
  ): Future[RegressionResultsHolder] = {
    // time series transformers/stages
    val (timeSeriesStages, paramMaps, kernelSize) = createTimeSeriesStagesWithParamGrids(groupIdColumnName, setting)
    //    val showDf = SchemaUnchangedTransformer { df: DataFrame => df.orderBy(seriesOrderCol).show(false); df }

    // forward-chaining cross validator
    val crossValidatorCreator = setting.core.crossValidationFolds.map(
      CrossValidatorFactory.withForwardChaining(seriesOrderCol, setting.minCrossValidationTrainingSizeRatio)
    )

    // data set training / test split
    val split = seqSplit(setting.core.trainingTestSplitRatio, setting.trainingTestSplitOrderValue)

    // how to calculate test predictions
    val calcTestPredictions = orderDependentTestPredictions(seriesOrderCol)

    // regress with the time series transformers and a sequential split
    regressAux(
      df, replicationDf, regressor, setting.core
    )(
      split, calcTestPredictions, crossValidatorCreator, Nil, timeSeriesStages, paramMaps
    )
  }

  protected def regressAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    regressor: Regressor,
    setting: RegressionLearningSetting)(
    splitDataSet: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreatorWithProcessor: Option[CrossValidatorCreatorWithProcessor],
    initStages: Seq[() => PipelineStage],
    preTrainingStages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]]
  ): Future[RegressionResultsHolder] = {
    // stages
    val coreStages = regressionStages(setting)
    val stages = initStages ++ coreStages ++ preTrainingStages

    // regress with the stages
    regressWithStages(
      df, replicationDf, regressor, setting
    )(
      splitDataSet, calcTestPredictions, crossValidatorCreatorWithProcessor, stages, paramGrids
    )
  }

  protected def regressWithStages(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    regressor: Regressor,
    setting: RegressionLearningSetting)(
    splitDataset: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreatorWithProcessor: Option[CrossValidatorCreatorWithProcessor],
    stages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]]
  ): Future[RegressionResultsHolder] = {
    // CREATE A TRAINER

    val (trainer, trainerParamGrids) = SparkMLEstimatorFactory(regressor)
    val fullParamMaps = buildParamGrids(trainerParamGrids ++ paramGrids)

    // REPEAT THE TRAINING-TEST CYCLE

    // cross-validation evaluator
    val crossValidationEvaluator =
      setting.crossValidationEvalMetric.flatMap(metric =>
        regressionEvaluators.find(_.metric == metric)
      ).getOrElse(
        regressionEvaluators.find(_.metric == defaultRegressionCrossValidationEvalMetric).get
      )

    val crossValidatorCreator = crossValidatorCreatorWithProcessor.map(_(None))

    val count = df.count()

    val resultHoldersFuture = parallelize(1 to setting.repetitions.getOrElse(1), repetitionParallelism) { index =>
      logger.info(s"Execution of repetition $index started for $count rows.")

      val fullTrainer = new Pipeline().setStages((stages.map(_()) ++ Seq(trainer)).toArray)

      // run the trainer (with folds) with a given split (which will produce training and test data sets) and a replication df (if provided)
      val (trainPredictions, testPredictions, replicationPredictions) = train(
        fullTrainer,
        fullParamMaps,
        crossValidationEvaluator.evaluator,
        crossValidatorCreator,
        splitDataset,
        calcTestPredictions,
        df,
        Seq(replicationDf).flatten
      )

      // evaluate the performance
      val results = evaluate(regressionEvaluators, trainPredictions, Seq(testPredictions) ++ replicationPredictions)

      // collect the actual vs expected outputs (if needed)
      val outputs: Traversable[Seq[(Double, Double)]] =
        if (setting.collectOutputs) {
          val trainingOutputs = collectLabelPredictions(trainPredictions)
          val testOutputs = collectLabelPredictions(testPredictions)
          Seq(trainingOutputs, testOutputs)
        } else
          Nil

      RegressionResultsAuxHolder(results, count, outputs)
    }

    // EVALUATE PERFORMANCE

    resultHoldersFuture.map { resultHolders =>
      // uncache
      df.unpersist
      if (replicationDf.isDefined)
        replicationDf.get.unpersist

      // create performance results
      val results = resultHolders.flatMap(_.evalResults)
      val performanceResults = results.groupBy(_._1).map { case (evalMetric, results) =>
        RegressionPerformance(evalMetric, results.map { case (_, trainResult, testResults) =>
          testResults.headOption.map(testResult =>
            (trainResult, Some(testResult): Option[Double], testResults.tail.headOption)
          ).getOrElse(
            (trainResult, None, None)
          )
        })
      }

      // counts
      val counts = resultHolders.map(_.count)

      // actual vs expected outputs
      val expectedAndActualOutputs = resultHolders.map(_.expectedAndActualOutputs)

      RegressionResultsHolder(performanceResults, counts, expectedAndActualOutputs)
    }
  }

  protected def createTimeSeriesStagesWithParamGrids(
    groupIdColumnName: Option[String],
    setting: TemporalLearningSetting
  ): (Seq[() => PipelineStage], Traversable[ParamGrid[_]], Int => Int) = {
    val slidingWindowUndefined = setting.slidingWindowSize.isLeft && setting.slidingWindowSize.left.get.isEmpty

    if (slidingWindowUndefined && setting.reservoirSetting.isEmpty)
      logger.warn("Sliding window size or reservoir setting should be set for time series transformations.")

    // sliding window transformer
    val swTransformerWithParamGrids =
      if (!slidingWindowUndefined) {
        val swConstructor = if (useConsecutiveOrderForDL)
          SlidingWindowWithConsecutiveOrder.applyInPlace("features", seriesOrderCol, groupIdColumnName)(_)
        else
          SlidingWindow.applyInPlace("features", seriesOrderCol, groupIdColumnName)(_)

        Some(swConstructor(setting.slidingWindowSize))
      } else
        None

    val swTransformer = swTransformerWithParamGrids.map(_._1)
    val swParamGrids = swTransformerWithParamGrids.map(_._2).getOrElse(Nil)

    // reservoir transformer
    if (setting.reservoirSetting.isDefined && groupIdColumnName.isDefined)
      throw new IncalSparkMLException(s"Reservoir processing defined together with a grouping by the column ${groupIdColumnName.get}. This combination is currently unsupported.")

    val rcTransformerWithParamGrids = setting.reservoirSetting.map(rcStatesWindowFactory.applyInPlace("features", seriesOrderCol))

    val rcTransformer = rcTransformerWithParamGrids.map(_._1)
    val rcParamGrids = rcTransformerWithParamGrids.map(_._2).getOrElse(Nil)

    // label shift
    val labelShiftTransformer =
      if (useConsecutiveOrderForDL)
        SeqShiftWithConsecutiveOrder.applyInPlace("label", seriesOrderCol, groupIdColumnName)(setting.predictAhead)
      else
        SeqShift.applyInPlace("label", seriesOrderCol, groupIdColumnName)(setting.predictAhead)

    // put all the transformers together
    val stages = Seq(
      swTransformer.map(() => _),
      rcTransformer.map(() => _),
      Some(() => labelShiftTransformer)
    ).flatten

    // TODO: this works (we extract the (kernel) size) only if it is a constant
    val sizeFun = (inputSize: Int) =>
      if (setting.reservoirSetting.isDefined) {
        setting.reservoirSetting.get.reservoirNodeNum match {
          case Left(value) => value.getOrElse(inputSize)
          case Right(_) => inputSize
        }
      } else {
        setting.slidingWindowSize match {
          case Left(value) => value.map(_ * inputSize).getOrElse(inputSize)
          case Right(_) => inputSize
        }
      }

    (stages, swParamGrids ++ rcParamGrids, sizeFun)
  }

  private def randomSplit(setting: LearningSetting[_]): DataFrame => (DataFrame, DataFrame) =
    randomSplit(setting.trainingTestSplitRatio.getOrElse(defaultTrainingTestingSplitRatio))

  private def seqSplit(
    trainingTestSplitRatio: Option[Double],
    trainingTestSplitOrderValue: Option[Double]
  ): DataFrame => (DataFrame, DataFrame) =
    if (trainingTestSplitOrderValue.isDefined)
      splitByValue(seriesOrderCol)(trainingTestSplitOrderValue.get)
    else if (trainingTestSplitRatio.isDefined)
      seqSplit(seriesOrderCol)(trainingTestSplitRatio.get)
    else
      throw new IncalSparkMLException("trainingTestSplitRatio or trainingTestSplitOrderValue must be defined for a seq split.")

  private def buildParamGrids(
    paramGrids: Traversable[ParamGrid[_]]
  ): Array[ParamMap] = {
    val paramGridBuilder = new ParamGridBuilder()
    paramGrids.foreach{ case ParamGrid(param, values) => paramGridBuilder.addGrid(param, values)}
    paramGridBuilder.build
  }

  private def collectLabelPredictions(dataFrame: DataFrame) = {
    val df =
      if (dataFrame.columns.find(_.equals(seriesOrderCol)).isDefined)
        dataFrame.orderBy(seriesOrderCol)
      else
        dataFrame

    def toDouble(value: Any) =
      value match {
        case x: Double => x
        case x: Int => x.toDouble
        case x: Long => x.toDouble
        case _ => throw new IllegalArgumentException(s"Cannot convert $value of type ${value.getClass.getName} to double.")
      }

    df.select("label", "prediction")
      .collect().toSeq
      .map(row => (toDouble(row.get(0)), toDouble(row.get(1))))
  }

  private def classifyAndEvaluate(
    trainer: Estimator[_],
    paramMaps: Array[ParamMap],
    evaluators: Seq[EvaluatorWrapper[ClassificationEvalMetric.Value]],
    crossValidationEvaluator: Evaluator,
    crossValidatorCreatorWithProcessor: Option[CrossValidatorCreatorWithProcessor],
    splitDataset: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    outputSize: Int,
    count: Long,
    binCurvesNumBins: Option[Int],
    collectOutputs: Boolean,
    mainDf: DataFrame,
    replicationDf: Option[DataFrame]
  ): ClassificationResultsAuxHolder = {

    // run the trainer (with folds) with a given split (which will produce training and test data sets) and a replication df (if provided)
    val predictionsProcessor = withBinaryEvaluationCol(outputSize)
    val (trainPredictions, testPredictions, replicationPredictions) = train(
      trainer,
      paramMaps,
      crossValidationEvaluator,
      crossValidatorCreatorWithProcessor.map(_(Some(predictionsProcessor))),
      splitDataset,
      calcTestPredictions,
      mainDf,
      Seq(replicationDf).flatten
    )

    // evaluate the performance

    // cache the predictions
    trainPredictions.cache()
    testPredictions.cache()
    replicationPredictions.foreach(_.cache())

    val trainingPredictionsExt = predictionsProcessor(trainPredictions)
    val testPredictionsExt = (Seq(testPredictions) ++ replicationPredictions).map(predictionsProcessor)

    val results = evaluate(evaluators, trainingPredictionsExt, testPredictionsExt)

    // generate binary classification curves (roc, pr, etc.) if the output is binary
    val (binTrainingCurves, binTestCurves) =
      if (outputSize == 2) {
        // is binary
        val trainingCurves = binaryMetricsCurves(trainingPredictionsExt, binCurvesNumBins)
        val testCurves = testPredictionsExt.map(binaryMetricsCurves(_, binCurvesNumBins))
        (trainingCurves, testCurves)
      } else
        (None, testPredictionsExt.map(_ => None))

    // collect the actual vs expected outputs (if needed)
    val outputs: Traversable[Seq[(Double, Double)]] =
      if (collectOutputs) {
        val trainingOutputs = collectLabelPredictions(trainPredictions)
        val testOutputs = collectLabelPredictions(testPredictions)
        Seq(trainingOutputs, testOutputs)
      } else
        Nil

    // unpersist and return the results
    trainPredictions.unpersist
    testPredictions.unpersist
    replicationPredictions.foreach(_.unpersist)

    ClassificationResultsAuxHolder(results, count, binTrainingCurves, binTestCurves, outputs)
  }

  private def withBinaryEvaluationCol(outputSize: Int) = { df: DataFrame =>
    if (outputSize == 2 && !df.columns.contains(binaryClassifierInputName)) {
      binaryPredictionVectorizer.transform(df)
    } else
      df
  }

  protected def classificationStages(
    setting: ClassificationLearningSetting
  ): Seq[() => PipelineStage] = {
    // normalize the features
    val normalize = setting.featuresNormalizationType.map(VectorColumnScaler.applyInPlace(_, "features"))

    // reduce the dimensionality if needed
    val reduceDim = setting.pcaDims.map(InPlacePCA(_))

    // keep the label as string for sampling (if needed)
    val keepLabelString = () => IndexToStringIfNeeded("label", "labelString")

    // sampling
    val sample = () => SamplingTransformer(setting.samplingRatios)

    // sequence the stages and return
    val preStages = Seq(normalize, reduceDim).flatten.map(() => _)
    if (setting.samplingRatios.nonEmpty) preStages ++ Seq(keepLabelString, sample) else preStages
  }

  protected def regressionStages(
    setting: RegressionLearningSetting
  ): Seq[() => PipelineStage] = {

    // normalize the features
    val normalizeFeatures = setting.featuresNormalizationType.map(VectorColumnScaler.applyInPlace(_, "features"))

    // normalize the output
    val normalizeOutput = setting.outputNormalizationType.map(NumericColumnScaler.applyInPlace(_, "label"))

    // reduce the dimensionality if needed
    val reduceDim = setting.pcaDims.map(InPlacePCA(_))

    // sequence the stages and return
    Seq(normalizeFeatures, reduceDim, normalizeOutput).flatten.map(() => _)
  }

  protected def evaluate[Q](
    evaluatorWrappers: Traversable[EvaluatorWrapper[Q]],
    trainPredictions: DataFrame,
    testPredictions: Seq[DataFrame]
  ): Traversable[(Q, Double, Seq[Double])] =
    evaluatorWrappers.flatMap { case EvaluatorWrapper(metric, evaluator) =>
      try {
        def evalNonEmpty(df: DataFrame) = if (df.count() > 0) Some(evaluator.evaluate(df)) else None

        val trainValue = evalNonEmpty(trainPredictions)
        val testValues = testPredictions.flatMap(evalNonEmpty)

        trainValue.map(trainValue => (metric, trainValue, testValues))
      } catch {
        case e: Exception =>
          val fieldNamesString = trainPredictions.schema.fieldNames.mkString(", ") + "\n"
          val rowsString = trainPredictions.take(10).map(_.toSeq.mkString(", ")).mkString("\n")

          logger.error(
            s"Evaluation of metric '$metric' failed." +
            s"Train Predictions: ${fieldNamesString + rowsString}"
          )
          None
      }
    }

  protected def verifyRocAndPrResults(predictionDf: DataFrame) = {
    val probabilityMetrics = binaryMetrics(predictionDf, None, "probability")
    val rawPredictionMetrics = binaryMetrics(predictionDf, None, "rawPrediction")

    if (probabilityMetrics.isDefined && rawPredictionMetrics.isDefined) {
      def areMoreLessEqual(val1: Double, val2: Double): Boolean =
        ((val1 == 0 && val2 == 0) || (val2 != 0 && Math.abs((val1 - val2) / val2) < 0.001))

      if (!areMoreLessEqual(probabilityMetrics.get.areaUnderROC(), rawPredictionMetrics.get.areaUnderROC()))
        throw new IncalSparkMLException("ROC values do not match: " + probabilityMetrics.get.areaUnderROC() + " vs " + rawPredictionMetrics.get.areaUnderROC())
      if (!areMoreLessEqual(probabilityMetrics.get.areaUnderPR(), rawPredictionMetrics.get.areaUnderPR()))
        throw new IncalSparkMLException("PR values do not match: " + probabilityMetrics.get.areaUnderPR() + " vs " + rawPredictionMetrics.get.areaUnderPR())
    }
  }

  private def binaryMetricsCurves(
    predictions: DataFrame,
    numBins: Option[Int] = None
  ) =
    binaryMetrics(predictions, numBins).map { metrics =>
      BinaryClassificationCurves(
        metrics.roc().collect(),
        metrics.pr().collect(),
        metrics.fMeasureByThreshold().collect(),
        metrics.precisionByThreshold().collect(),
        metrics.recallByThreshold().collect()
      )
    }

  private def binaryMetrics(
    predictions: DataFrame,
    numBins: Option[Int] = None,
    probabilityCol: String = binaryClassifierInputName,
    labelCol: String = "label"
  ): Option[BinaryClassificationMetrics] =
    if (predictions.count() > 0) {
      val topRow = predictions.select(binaryClassifierInputName).head()
      if (topRow.getAs[Vector](0).size == 2) {
        val metrics = new BinaryClassificationMetrics(
          predictions.select(col(probabilityCol), col(labelCol).cast(DoubleType)).rdd.map {
            case Row(score: Vector, label: Double) => (score(1), label)
          }, numBins.getOrElse(0)
        )
        Some(metrics)
      } else
        None
    } else
      None

  private def train(
    trainer: Estimator[_],
    paramMaps: Array[ParamMap],
    crossValidationEvaluator: Evaluator,
    crossValidatorCreator: Option[CrossValidatorCreator],
    splitDataset: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    mainDf: DataFrame,
    replicationDfs: Seq[DataFrame]
  ): (DataFrame, DataFrame, Seq[DataFrame]) = {
    def trainAux(estimator: Estimator[_]) = {
      // split the main data frame
      val (trainingDf, testDf) = splitDataset(mainDf)

      logger.info("Dataset split into training and test parts as: " + trainingDf.count() + " / " + testDf.count())

      // cache training and test data frames
      trainingDf.cache()
      testDf.cache()

      if (setting.debugMode) {
        logger.debug(s"Training Data Set (# ${trainingDf.count}):\n")
        trainingDf.show(truncate = false)

        logger.debug(s"Test Data Set (# ${testDf.count}):\n")
        testDf.show(truncate = false)
      }

      // fit the model to the training set
      val mlModel = estimator.fit(trainingDf).asInstanceOf[Transformer]

      // get the predictions for the training, test and replication data sets

      val trainingPredictions = mlModel.transform(trainingDf)
      val testPredictions = calcTestPredictions(mlModel, testDf, mainDf)
      val replicationPredictions = replicationDfs.map(mlModel.transform)

      logger.info("Obtained training/test predictions as: " + trainingPredictions.count() + " / " + testPredictions.count())

      if (setting.debugMode) {
        logger.debug(s"Training Predictions (# ${trainingPredictions.count}):\n")
        trainingPredictions.show(truncate = false)

        logger.debug(s"Test Predictions (# ${testPredictions.count}):\n")
        testPredictions.show(truncate = false)
      }

      // unpersist and return the predictions
      trainingDf.unpersist
      testDf.unpersist

      (trainingPredictions, testPredictions, replicationPredictions)
    }

    // use cross-validation if the folds specified together with params to search through, and train
    crossValidatorCreator.map { crossValidatorCreator =>
      val cv = crossValidatorCreator(trainer, paramMaps, crossValidationEvaluator)
      trainAux(cv)
    }.getOrElse(
      trainAux(trainer)
    )
  }

  def cluster(
    df: DataFrame,
    idColumnName: String,
    mlModel: Clustering,
    featuresNormalizationType: Option[VectorScalerType.Value],
    pcaDim: Option[Int]
  ): (DataFrame, Traversable[(String, Int)]) = {
    val trainer = SparkMLEstimatorFactory(mlModel)

    // normalize
    val normalize = featuresNormalizationType.map(VectorColumnScaler.applyInPlace(_, "features"))

    // reduce the dimensionality if needed
    val reduceDim = pcaDim.map(InPlacePCA(_))

    val stages = Seq(normalize, reduceDim).flatten
    val pipeline = new Pipeline().setStages(stages.toArray)
    val dataFrame = pipeline.fit(df).transform(df)

    val cachedDf = dataFrame.cache()
    val classes = fitClustersAndGetClasses(trainer, cachedDf, idColumnName)

    cachedDf.unpersist

    (dataFrame, classes)
  }

  private def fitClustersAndGetClasses[M <: Model[M]](
    estimator: Estimator[M],
    data: DataFrame,
    idColumnName: String
  ): Traversable[(String, Int)] = {
    val (model, predictions) = fit(estimator, data)
    predictions.cache()

    implicit val encoder = Encoders.tuple(Encoders.STRING, Encoders.scalaInt)

    def extractClusterClasses(columnName: String): Traversable[(String, Int)] =
      predictions.select(idColumnName, columnName).map { r =>
        val id = r(0).asInstanceOf[String]
        val clazz = r(1).asInstanceOf[Int]
        (id, clazz + 1)
      }.collect

    def extractClusterClassesFromProbabilities(columnName: String): Traversable[(String, Int)] =
      predictions.select(idColumnName, columnName).map { r =>
        val id = r(0).asInstanceOf[String]
        val clazz = r(1).asInstanceOf[DenseVector].values.zipWithIndex.maxBy(_._1)._2
        (id, clazz + 1)
      }.collect

    val result = model match {
      case _: KMeansModel =>
        extractClusterClasses("prediction")

      case _: LDAModel =>
        extractClusterClassesFromProbabilities("topicDistribution")

      case _: BisectingKMeansModel =>
        extractClusterClasses("prediction")

      case _: GaussianMixtureModel =>
        extractClusterClassesFromProbabilities("probability")
    }

    predictions.unpersist()

    result
  }

  protected def fit[M <: Model[M]](
    estimator: Estimator[M],
    data: DataFrame
  ): (M, DataFrame) = {
    // Fit the model
    val lrModel = estimator.fit(data)

    // Make predictions.
    val predictions = lrModel.transform(data)

    (lrModel, predictions)
  }
}

case class SparkMLServiceSetting(
  repetitionParallelism: Option[Int] = None,
  binaryClassifierInputName: Option[String] = None,
  useConsecutiveOrderForDL: Option[Boolean] = None,
  debugMode: Boolean = false
)