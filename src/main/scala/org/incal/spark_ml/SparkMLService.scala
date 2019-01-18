package org.incal.spark_ml

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.{util, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.slf4j.LoggerFactory
import org.incal.spark_ml.transformers._
import org.incal.spark_ml.models.{LearningSetting, ReservoirSpec}
import org.incal.spark_ml.models.classification.{Classification, ClassificationEvalMetric}
import org.incal.spark_ml.models.regression.{Regression, RegressionEvalMetric}
import org.incal.spark_ml.CrossValidatorFactory.CrossValidatorCreator
import org.incal.spark_ml.MachineLearningUtil._
import org.incal.spark_ml.models.results._
import org.incal.core.util.{STuple3, parallelize}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

trait SparkMLService {

  val rcStatesWindowFactory: RCStatesWindowFactory
  val setting: SparKMLServiceSetting

  protected val logger = LoggerFactory.getLogger("ml")

  // consts
  protected val defaultTrainingTestingSplit = 0.8
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

  private val classificationEvaluators =
    ClassificationEvalMetric.values.filter(metric =>
      metric != ClassificationEvalMetric.areaUnderPR && metric != ClassificationEvalMetric.areaUnderROC
    ).toSeq.map { metric =>
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(metric.toString)

      EvaluatorWrapper(metric, evaluator)
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

  private val regressionEvaluators = RegressionEvalMetric.values.toSeq.map { metric =>
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metric.toString)

    EvaluatorWrapper(metric, evaluator)
  }

  def classify(
    df: DataFrame,
    replicationDf: Option[DataFrame] = None,
    mlModel: Classification,
    setting: LearningSetting[ClassificationEvalMetric.Value],
    binCurvesNumBins: Option[Int] = None
  ): Future[ClassificationResultsHolder]  = {

    // k-folds cross validator
    val crossValidatorCreator = setting.crossValidationFolds.map(CrossValidatorFactory.withFolds)

    // data set training / test split
    val split = randomSplit

    // how to calculate test predictions
    val calcTestPredictions = independentTestPredictions

    // classify with a random split
    classifyAux(df, replicationDf, mlModel, setting, binCurvesNumBins, split, calcTestPredictions, crossValidatorCreator, Nil, Nil, Nil)
  }

  def classifyTimeSeries(
    df: DataFrame,
    replicationDf: Option[DataFrame] = None,
    predictAhead: Int,
    windowSize: Option[Int] = None,
    reservoirSetting: Option[ReservoirSpec] = None,
    mlModel: Classification,
    setting: LearningSetting[ClassificationEvalMetric.Value],
    minCrossValidationTrainingSize: Option[Double] = None,
    binCurvesNumBins: Option[Int] = None,
    groupIdCol: Option[String] = None
  ): Future[ClassificationResultsHolder] = {

    // time series transformers/stages
    val (timeSeriesStages, paramGrids) = createTimeSeriesStagesWithParamGrids(windowSize, reservoirSetting, predictAhead, groupIdCol)

    // forward-chaining cross validator
    val crossValidatorCreator = setting.crossValidationFolds.map(
      CrossValidatorFactory.withForwardChaining(seriesOrderCol, minCrossValidationTrainingSize)
    )

    // data set training / test split
    val split = seqSplit(seriesOrderCol)

    // how to calculate test predictions
    val calcTestPredictions = orderDependentTestPredictions(seriesOrderCol)

    // classify with the time-series transformers and a sequential split
    classifyAux(df, replicationDf, mlModel, setting, binCurvesNumBins, split, calcTestPredictions, crossValidatorCreator, Nil, timeSeriesStages, paramGrids)
  }

  private def classifyAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    mlModel: Classification,
    setting: LearningSetting[ClassificationEvalMetric.Value],
    binCurvesNumBins: Option[Int],
    splitDataSet: Double => (DataFrame => (DataFrame, DataFrame)),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreator: Option[CrossValidatorCreator],
    initStages: Seq[() => PipelineStage],
    preTrainingStages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]] = Nil
  ): Future[ClassificationResultsHolder] = {
    // stages
    val coreStages = classificationStages(setting)
    val stages = initStages ++ coreStages ++ preTrainingStages

    // classify with the stages
    classifyAux(df, replicationDf, mlModel, setting, binCurvesNumBins, splitDataSet, calcTestPredictions, crossValidatorCreator, stages, paramGrids)
  }

  protected def classifyAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    mlModel: Classification,
    setting: LearningSetting[ClassificationEvalMetric.Value],
    binCurvesNumBins: Option[Int],
    splitDataset: Double => (DataFrame => (DataFrame, DataFrame)),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreator: Option[CrossValidatorCreator],
    stages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]]
  ): Future[ClassificationResultsHolder] = {

    // cache the data frames
    df.cache()
    if (replicationDf.isDefined)
      replicationDf.get.cache()

    // CREATE A TRAINER

    val originalFeaturesType = df.schema.fields.find(_.name == "features").get
    val originalInputSize = originalFeaturesType.metadata.getMetadata("ml_attr").getLong("num_attrs").toInt
    val inputSize = setting.pcaDims.getOrElse(originalInputSize)

    val outputLabelType = df.schema.fields.find(_.name == "label").get
    val outputSize = outputLabelType.metadata.getMetadata("ml_attr").getStringArray("vals").length

    val (trainer, trainerParamGrids) = SparkMLEstimatorFactory(mlModel, inputSize, outputSize)
    val fullParamMaps = buildParamGrids(trainerParamGrids ++ paramGrids)

    // REPEAT THE TRAINING-TEST CYCLE

    // split for the data into training and test parts
    val splitRatio = setting.trainingTestingSplit.getOrElse(defaultTrainingTestingSplit)

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
        crossValidatorCreator,
        splitDataset(splitRatio),
        calcTestPredictions,
        outputSize,
        count,
        binCurvesNumBins,
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
          val replicationResult = testResults.tail.headOption
          (trainResult, testResults.head, replicationResult)
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

      ClassificationResultsHolder(performanceResults, counts, curves)
    }
  }

  def regress(
    df: DataFrame,
    replicationDf: Option[DataFrame] = None,
    mlModel: Regression,
    setting: LearningSetting[RegressionEvalMetric.Value]
  ): Future[RegressionResultsHolder] = {

    // k-folds cross-validator
    val crossValidatorCreator = setting.crossValidationFolds.map(CrossValidatorFactory.withFolds)

    // data set training / test split
    val split = randomSplit

    // how to calculate test predictions
    val calcTestPredictions = independentTestPredictions

    // regress
    regressAux(df, replicationDf, mlModel, setting, split, calcTestPredictions, crossValidatorCreator, Nil, Nil, Nil, false)
  }

  def regressTimeSeries(
    df: DataFrame,
    replicationDf: Option[DataFrame] = None,
    predictAhead: Int,
    windowSize: Option[Int] = None,
    reservoirSetting: Option[ReservoirSpec] = None,
    mlModel: Regression,
    setting: LearningSetting[RegressionEvalMetric.Value],
    minCrossValidationTrainingSize: Option[Double] = None,
    groupIdCol: Option[String] = None
  ): Future[RegressionResultsHolder] = {
    // time series transformers/stages
    val (timeSeriesStages, paramMaps) = createTimeSeriesStagesWithParamGrids(windowSize, reservoirSetting, predictAhead, groupIdCol)
    //    val showDf = SchemaUnchangedTransformer { df: DataFrame => df.orderBy(seriesOrderCol).show(false); df }

    // forward-chaining cross validator
    val crossValidatorCreator = setting.crossValidationFolds.map(
      CrossValidatorFactory.withForwardChaining(seriesOrderCol, minCrossValidationTrainingSize)
    )

    // data set training / test split
    val split = seqSplit(seriesOrderCol)

    // how to calculate test predictions
    val calcTestPredictions = orderDependentTestPredictions(seriesOrderCol)

    // regress with the time series transformers and a sequential split
    regressAux(df, replicationDf, mlModel, setting, split, calcTestPredictions, crossValidatorCreator, Nil, timeSeriesStages, paramMaps, true)
  }

  protected def regressAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    mlModel: Regression,
    setting: LearningSetting[RegressionEvalMetric.Value],
    splitDataSet: Double => (DataFrame => (DataFrame, DataFrame)),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreator: Option[CrossValidatorCreator],
    initStages: Seq[() => PipelineStage],
    preTrainingStages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]] = Nil,
    collectOutputs: Boolean = false
  ): Future[RegressionResultsHolder] = {
    // stages
    val coreStages = regressionStages(setting)
    val stages = initStages ++ coreStages ++ preTrainingStages

    // regress with the stages
    regressAux(df, replicationDf, mlModel, setting, splitDataSet, calcTestPredictions, crossValidatorCreator, stages, paramGrids, collectOutputs)
  }

  private def regressAux(
    df: DataFrame,
    replicationDf: Option[DataFrame],
    mlModel: Regression,
    setting: LearningSetting[RegressionEvalMetric.Value],
    splitDataset: Double => (DataFrame => (DataFrame, DataFrame)),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    crossValidatorCreator: Option[CrossValidatorCreator],
    stages: Seq[() => PipelineStage],
    paramGrids: Traversable[ParamGrid[_]],
    collectOutputs: Boolean
  ): Future[RegressionResultsHolder] = {
    // CREATE A TRAINER

    val (trainer, trainerParamGrids) = SparkMLEstimatorFactory(mlModel)
    val fullParamMaps = buildParamGrids(trainerParamGrids ++ paramGrids)

    // REPEAT THE TRAINING-TEST CYCLE

    // split ratio for the data into training and test parts
    val splitRatio = setting.trainingTestingSplit.getOrElse(defaultTrainingTestingSplit)

    // cross-validation evaluator
    val crossValidationEvaluator =
      setting.crossValidationEvalMetric.flatMap(metric =>
        regressionEvaluators.find(_.metric == metric)
      ).getOrElse(
        regressionEvaluators.find(_.metric == defaultRegressionCrossValidationEvalMetric).get
      )

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
        splitDataset(splitRatio),
        calcTestPredictions,
        df,
        Seq(replicationDf).flatten
      )

      // evaluate the performance
      val results = evaluate(regressionEvaluators, trainPredictions, Seq(testPredictions) ++ replicationPredictions)

      // collect the actual vs expected outputs (if needed)
      val outputs: Traversable[Seq[(Double, Double)]] =
        if (collectOutputs) {
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
          val replicationResult = testResults.tail.headOption
          (trainResult, testResults.head, replicationResult)
        })
      }

      // counts
      val counts = resultHolders.map(_.count)

      // actual vs expected outputs
      val expectedAndActualOutputs = resultHolders.map(_.expectedAndActualOutputs)

      RegressionResultsHolder(performanceResults, counts, expectedAndActualOutputs)
    }
  }

  private def createTimeSeriesStagesWithParamGrids(
    windowSize: Option[Int],
    reservoirSetting: Option[ReservoirSpec],
    labelShift: Int,
    groupCol: Option[String] = None
  ): (Seq[() => PipelineStage], Traversable[ParamGrid[_]]) = {
    if (windowSize.isEmpty && reservoirSetting.isEmpty)
      logger.warn("Window size or reservoir setting should be set for time series transformations.")

    val dlTransformer = windowSize.map(
      if (useConsecutiveOrderForDL)
        SlidingWindowWithConsecutiveOrder.applyInPlace("features", seriesOrderCol, groupCol)
      else
        SlidingWindow.applyInPlace("features", seriesOrderCol, groupCol)
    )

    if (reservoirSetting.isDefined && groupCol.isDefined)
      throw new IncalSparkMLException(s"Reservoir processing defined together with a grouping by the column ${groupCol.get}, which is currently unsupported.")

    val rcTransformerWithParamGrids = reservoirSetting.map(rcStatesWindowFactory.applyInPlace("features", seriesOrderCol))

    val rcTransformer = rcTransformerWithParamGrids.map(_._1)
    val paramGrids = rcTransformerWithParamGrids.map(_._2).getOrElse(Nil)

    val labelShiftTransformer =
      if (useConsecutiveOrderForDL)
        SeqShiftWithConsecutiveOrder.applyInPlace("label", seriesOrderCol, groupCol)(labelShift)
      else
        SeqShift.applyInPlace("label", seriesOrderCol, groupCol)(labelShift)

    val stages = Seq(
      dlTransformer.map(() => _),
      rcTransformer.map(() => _),
      Some(() => labelShiftTransformer)
    ).flatten

    (stages, paramGrids)
  }

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
    crossValidatorCreator: Option[CrossValidatorCreator],
    splitDataset: DataFrame => (DataFrame, DataFrame),
    calcTestPredictions: (Transformer, Dataset[_], Dataset[_]) => DataFrame,
    outputSize: Int,
    count: Long,
    binCurvesNumBins: Option[Int],
    mainDf: DataFrame,
    replicationDf: Option[DataFrame]
  ): ClassificationResultsAuxHolder = {

    // run the trainer (with folds) with a given split (which will produce training and test data sets) and a replication df (if provided)
    val (trainPredictions, testPredictions, replicationPredictions) = train(
      trainer,
      paramMaps,
      crossValidationEvaluator,
      crossValidatorCreator,
      splitDataset,
      calcTestPredictions,
      mainDf,
      Seq(replicationDf).flatten
    )

    // evaluate the performance

    def withBinaryEvaluationCol(df: DataFrame) =
      if (outputSize == 2 && !df.columns.contains(binaryClassifierInputName)) {
        binaryPredictionVectorizer.transform(df)
      } else
        df

    // cache the predictions
    trainPredictions.cache()
    testPredictions.cache()
    replicationPredictions.foreach(_.cache())

    val trainingPredictionsExt = withBinaryEvaluationCol(trainPredictions)
    val testPredictionsExt = (Seq(testPredictions) ++ replicationPredictions).map(withBinaryEvaluationCol)

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

    // unpersist and return the results
    trainPredictions.unpersist
    testPredictions.unpersist
    replicationPredictions.foreach(_.unpersist)

    ClassificationResultsAuxHolder(results, count, binTrainingCurves, binTestCurves)
  }

  private def classificationStages(
    setting: LearningSetting[_]
  ): Seq[() => PipelineStage] = {
    // normalize the features
    val normalize = setting.featuresNormalizationType.map(VectorColumnScalerNormalizer.applyInPlace(_, "features"))

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

  private def regressionStages(
    setting: LearningSetting[_]
  ): Seq[() => PipelineStage] = {
    // normalize the features
    val normalize = setting.featuresNormalizationType.map(VectorColumnScalerNormalizer.applyInPlace(_, "features"))

    // reduce the dimensionality if needed
    val reduceDim = setting.pcaDims.map(InPlacePCA(_))

    // sequence the stages and return
    Seq(normalize, reduceDim).flatten.map(() => _)
  }

  private def evaluate[Q](
    evaluatorWrappers: Traversable[EvaluatorWrapper[Q]],
    trainPredictions: DataFrame,
    testPredictions: Seq[DataFrame]
  ): Traversable[(Q, Double, Seq[Double])] =
    evaluatorWrappers.flatMap { case EvaluatorWrapper(metric, evaluator) =>
      try {
        val trainValue = evaluator.evaluate(trainPredictions)
        val testValues = testPredictions.map(evaluator.evaluate)
        Some((metric, trainValue, testValues))
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
  ): Option[BinaryClassificationMetrics] = {
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
  }

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

      // fit the model to the training set
      val mlModel = estimator.fit(trainingDf).asInstanceOf[Transformer]

      // get the predictions for the training, test and replication data sets

      val trainPredictions = mlModel.transform(trainingDf)
      val testPredictions = calcTestPredictions(mlModel, testDf, mainDf)
      val replicationPredictions = replicationDfs.map(mlModel.transform)

      logger.info("Obtained training/test predictions as: " + trainPredictions.count() + " / " + testPredictions.count())
//      println("Training predictions min index  : " + trainPredictions.agg(min(trainPredictions("index"))).head.getInt(0))
//      println("Training predictions max index  : " + trainPredictions.agg(max(trainPredictions("index"))).head.getInt(0))
//      println("Test predictions min index      : " + testPredictions.agg(min(testPredictions("index"))).head.getInt(0))
//      println("Test predictions max index      : " + testPredictions.agg(max(testPredictions("index"))).head.getInt(0))

//      trainPredictions.show(false)
//      testPredictions.show(false)

      // unpersist and return the predictions
      trainingDf.unpersist
      testDf.unpersist

      (trainPredictions, testPredictions, replicationPredictions)
    }

    // use cross-validation if the folds specified together with params to search through, and train
    crossValidatorCreator.map { crossValidatorCreator =>
      val cv = crossValidatorCreator(trainer, paramMaps, crossValidationEvaluator)
      trainAux(cv)
    }.getOrElse(
      trainAux(trainer)
    )
  }

  case class EvaluatorWrapper[Q](metric: Q, evaluator: Evaluator)
}

case class SparKMLServiceSetting(
  repetitionParallelism: Option[Int] = None,
  binaryClassifierInputName: Option[String] = None,
  useConsecutiveOrderForDL: Option[Boolean] = None
)
