package org.apache.spark.ml.tuning

import java.util.{List => JList}

import com.github.fommil.netlib.F2jBLAS
import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions.{max, min}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.StructType
import org.incal.spark_ml.{IncalSparkMLException, MLBase}

import scala.collection.JavaConverters._
import org.json4s.DefaultFormats
import org.slf4j.LoggerFactory

// TODO: could be substantially simplified if CrossValidator provides "splits" function that can be overridden, plus the model would need to be generic
class ForwardChainingCrossValidator(override val uid: String)
  extends Estimator[ForwardChainingCrossValidatorModel]
    with MLBase
    with CrossValidatorParams with MLWritable with Logging {

  private val logger = LoggerFactory.getLogger("spark_ml")

  @Since("1.2.0")
  def this() = this(Identifiable.randomUID("forward_cv"))

  private val f2jBLAS = new F2jBLAS

  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  def setNumFolds(value: Int): this.type = set(numFolds, value)

  def setSeed(value: Long): this.type = set(seed, value)

  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")

  def setOrderCol(value: String): this.type = set(orderCol, value)

  protected final val minTrainingSizeRatio: Param[Double] = new Param[Double](this, "minTrainingSizeRatio", "min training size ratio", ParamValidators.inRange(0, 1))

  def setMinTrainingSizeRatio(value: Double): this.type = set(minTrainingSizeRatio, value)

  protected final val predictionsProcessor: Param[DataFrame => DataFrame] = new Param[DataFrame => DataFrame](this, "predictionsProcessor", "predictions processor")

  def setPredictionsProcessor(value: DataFrame => DataFrame): this.type = set(predictionsProcessor, value)

  // can be removed
  private def splitFolds(dataset: Dataset[_]): Array[(Dataset[_], Dataset[_])] = {
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    splits.map { case (training, validation) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema)
      val validationDataset = sparkSession.createDataFrame(validation, schema)
      (trainingDataset, validationDataset)
    }
  }

  protected def splitForward(dataset: Dataset[_]):  Array[(Dataset[_], Dataset[_], Dataset[_])] = {
    val stepSize = (1 - $(minTrainingSizeRatio)) / $(numFolds)
    val maxOrderValue = dataset.agg(max($(orderCol))).head.getInt(0)

    val splitValues = for (i <- 0 until $(numFolds)) yield
      dataset.stat.approxQuantile($(orderCol), Array($(minTrainingSizeRatio) + i * stepSize), 0.001)(0)

    splitValues.zip(splitValues.tail ++ Seq(maxOrderValue: Double)).map { case (trainingSplitValue, validationSplitValue) =>
      logger.info(s"Splitting forward as: training <= $trainingSplitValue, validation > $trainingSplitValue && <= $validationSplitValue")
      val trainingSet = dataset.where(dataset($(orderCol)) <= trainingSplitValue)
      val validationSet = dataset.where(dataset($(orderCol)) > trainingSplitValue and dataset($(orderCol)) <= validationSplitValue)
      val fullSet = dataset.where(dataset($(orderCol)) <= validationSplitValue)
      (trainingSet, validationSet, fullSet)
    }.toArray
  }

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ForwardChainingCrossValidatorModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)

    val est = $(estimator)
    val eval = $(evaluator)
    val processor = get(predictionsProcessor).getOrElse(identity[DataFrame](_))
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)

    val instr = Instrumentation.create(this, dataset)
    instr.logParams(numFolds, seed)
    logTuningParams(instr)

    // this is the only line we had to change
    val splitDatasets = splitForward(dataset)

    // function to calculate test predictions for ordered data sets
    val calcTestPredictions = orderDependentTestPredictionsWithParams($(orderCol))

    splitDatasets.zipWithIndex.foreach { case ((trainingDataset, validationDataset, fullDataset), splitIndex) =>
      trainingDataset.cache()
      validationDataset.cache()

      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
//        val validationPredictions = models(i).transform(validationDataset, epm(i))
        val validationPredictions = calcTestPredictions(models(i), validationDataset, fullDataset, epm(i))
        if (validationPredictions.count() == 0)
          throw new IncalSparkMLException(s"Got no validation predictions for a forward-chaining cross validation. Perhaps the validation set with ${validationDataset.count} rows is too short.")

        val metric = eval.evaluate(processor(validationPredictions))
        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric
        i += 1
      }
      validationDataset.unpersist()
    }

    // choosing best model (params)
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)

    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    instr.logSuccess(bestModel)
    copyValues(new ForwardChainingCrossValidatorModel(uid, bestModel, metrics).setParent(this))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  @Since("1.4.0")
  override def copy(extra: ParamMap): ForwardChainingCrossValidator = {
    val copied = defaultCopy(extra).asInstanceOf[ForwardChainingCrossValidator]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  // Currently, this only works if all [[Param]]s in [[estimatorParamMaps]] are simple types.
  // E.g., this may fail if a [[Param]] is an instance of an [[Estimator]].
  // However, this case should be unusual.
  @Since("1.6.0")
  override def write: MLWriter = new ForwardChainingCrossValidator.ForwardChainingCrossValidatorWriter(this)
}

object ForwardChainingCrossValidator extends MLReadable[ForwardChainingCrossValidator] {

  override def read: MLReader[ForwardChainingCrossValidator] = new ForwardChainingCrossValidatorReader

  override def load(path: String): ForwardChainingCrossValidator = super.load(path)

  private[ForwardChainingCrossValidator] class ForwardChainingCrossValidatorWriter(instance: ForwardChainingCrossValidator) extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

  private class ForwardChainingCrossValidatorReader extends MLReader[ForwardChainingCrossValidator] {

    /** Checked against metadata when loading model */
    private val className = classOf[ForwardChainingCrossValidator].getName

    override def load(path: String): ForwardChainingCrossValidator = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val numFolds = (metadata.params \ "numFolds").extract[Int]
      val seed = (metadata.params \ "seed").extract[Long]
      val orderCol = (metadata.params \ "orderCol").extract[String]
      val minTrainingSize = (metadata.params \ "minTrainingSize").extract[Double]

      new ForwardChainingCrossValidator(metadata.uid)
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(estimatorParamMaps)
        .setSeed(seed)
        .setNumFolds(numFolds)
        .setOrderCol(orderCol)
        .setMinTrainingSizeRatio(minTrainingSize)
    }
  }
}

class ForwardChainingCrossValidatorModel private[ml] (
  @Since("1.4.0") override val uid: String,
  @Since("1.2.0") val bestModel: Model[_],
  @Since("1.5.0") val avgMetrics: Array[Double]
) extends Model[ForwardChainingCrossValidatorModel] with CrossValidatorParams with MLWritable {

  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")
  protected final val minTrainingSize: Param[Double] = new Param[Double](this, "minTrainingSize", "min training size")

  /** A Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, bestModel: Model[_], avgMetrics: JList[Double]) = {
    this(uid, bestModel, avgMetrics.asScala.toArray)
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): ForwardChainingCrossValidatorModel = {
    val copied = new ForwardChainingCrossValidatorModel(
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      avgMetrics.clone()
    )

    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new ForwardChainingCrossValidatorModel.ForwardChainingCrossValidatorModelWriter(this)
}

@Since("1.6.0")
object ForwardChainingCrossValidatorModel extends MLReadable[ForwardChainingCrossValidatorModel] {

  @Since("1.6.0")
  override def read: MLReader[ForwardChainingCrossValidatorModel] = new ForwardChainingCrossValidatorModelReader

  @Since("1.6.0")
  override def load(path: String): ForwardChainingCrossValidatorModel = super.load(path)

  private[ForwardChainingCrossValidatorModel] class ForwardChainingCrossValidatorModelWriter(instance: ForwardChainingCrossValidatorModel) extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit = {
      import org.json4s.JsonDSL._
      val extraMetadata = "avgMetrics" -> instance.avgMetrics.toSeq
      ValidatorParams.saveImpl(path, instance, sc, Some(extraMetadata))
      val bestModelPath = new Path(path, "bestModel").toString
      instance.bestModel.asInstanceOf[MLWritable].save(bestModelPath)
    }
  }

  private class ForwardChainingCrossValidatorModelReader extends MLReader[ForwardChainingCrossValidatorModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[ForwardChainingCrossValidatorModel].getName

    override def load(path: String): ForwardChainingCrossValidatorModel = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val numFolds = (metadata.params \ "numFolds").extract[Int]
      val seed = (metadata.params \ "seed").extract[Long]
      val orderCol = (metadata.params \ "orderCol").extract[String]
      val minTrainingSize = (metadata.params \ "minTrainingSize").extract[Double]

      val bestModelPath = new Path(path, "bestModel").toString
      val bestModel = DefaultParamsReader.loadParamsInstance[Model[_]](bestModelPath, sc)
      val avgMetrics = (metadata.metadata \ "avgMetrics").extract[Seq[Double]].toArray

      val model = new ForwardChainingCrossValidatorModel(metadata.uid, bestModel, avgMetrics)
      model.set(model.estimator, estimator)
        .set(model.evaluator, evaluator)
        .set(model.estimatorParamMaps, estimatorParamMaps)
        .set(model.numFolds, numFolds)
        .set(model.seed, seed)
        .set(model.orderCol, orderCol)
        .set(model.minTrainingSize, minTrainingSize)
    }
  }
}