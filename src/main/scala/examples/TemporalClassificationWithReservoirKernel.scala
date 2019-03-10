package examples

import com.banda.math.domain.rand.RandomDistribution
import com.banda.network.domain.ActivationFunctionType
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.BooleanType
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import org.incal.spark_ml.models.{ReservoirSpec, VectorScalerType}
import org.incal.spark_ml.models.classification.{ClassificationEvalMetric, LogisticModelFamily, LogisticRegression, MultiLayerPerceptron}
import org.incal.spark_ml.models.result.ClassificationResultsHolder
import org.incal.spark_ml.models.setting.{ClassificationLearningSetting, TemporalClassificationLearningSetting}
import org.incal.spark_ml.transformers.BooleanLabelIndexer
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}

import scala.concurrent.ExecutionContext.Implicits.global

object TemporalClassificationWithReservoirKernel extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val index,AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4,eyeDetection,eyeDetectionBool = Value
  }

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val outputColumnName = Column.eyeDetectionBool.toString
  val orderColumnName = Column.index.toString
  val featureColumnNames = columnNames.filter(name => name != outputColumnName && name != orderColumnName)

  // read a csv and create a data frame with given column names
  val url = "https://in-cal.org/data/EEG_Eye_State_by_DBWH.csv"
  val df = remoteCsvToDataFrame(url, true)(session)

  val df2 = df.withColumn(outputColumnName, df(Column.eyeDetection.toString).cast(BooleanType))

  // turn the data frame into ML-ready one with features and a label
  val df3 = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df2)
  val finalDf = BooleanLabelIndexer(Some(outputColumnName)).transform(df3)

  // logistic regression spec
  val logisticRegressionSpec = LogisticRegression(
    family = Some(LogisticModelFamily.Binomial),
    regularization = Right(Seq(1, 0.1)), // Right(Seq(10, 1, 0.1, 0.01, 0.001)),
    elasticMixingRatio = Right(Seq(0, 0.5, 1))
  )

  // multi-layer perceptron spec
  val multiLayerPerceptronSpec = MultiLayerPerceptron(
    hiddenLayers = Seq(5, 5),
    blockSize = Right(Seq(32,64,128))
  )

  // learning setting
  val classificationLearningSetting = ClassificationLearningSetting(
    trainingTestSplitRatio = Some(0.75),
    featuresNormalizationType = Some(VectorScalerType.StandardScaler),
    crossValidationEvalMetric = Some(ClassificationEvalMetric.areaUnderROC),
    crossValidationFolds = Some(5)
  )

  val reservoirSpec = ReservoirSpec(
    inputNodeNum = featureColumnNames.size,
    bias = 1,
    nonBiasInitial = 0,
    reservoirNodeNum = Left(Some(20)), // Right(Seq(10,20,50)),
    reservoirInDegree = Left(Some(20)),
    inputReservoirConnectivity = Left(Some(1)),
    weightDistribution = RandomDistribution.createNormalDistribution(classOf[java.lang.Double], 0d, 1d),
    reservoirSpectralRadius = Right(Seq(0.9, 0.95, 0.99)),
    reservoirFunctionType = ActivationFunctionType.Tanh
    //  reservoirEdgesNum: ValueOrSeq[Int] = Left(None),
    //  reservoirInDegreeDistribution: Option[RandomDistribution[Integer]] = None,
    //  reservoirCircularInEdges: Option[Seq[Int]] = None,
    //  reservoirPreferentialAttachment: Boolean = false,
    //  reservoirFunctionParams: Seq[Double] = Nil,
    //  washoutPeriod: ValueOrSeq[Int] = Left(None)
  )

  val temporalLearningSetting = TemporalClassificationLearningSetting(
    core = classificationLearningSetting,
    predictAhead = 100, // roughly 0.78 sec ahead
    reservoirSetting = Some(reservoirSpec)
  )

  // aux function to get a mean training and test accuracy and AUROC
  def calcMeanAccuracyAndAUROC(results: ClassificationResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingAccuracy, Some(testAccuracy), _) = metricStatsMap.get(ClassificationEvalMetric.accuracy).get
    val (trainingAUROC, Some(testAUROC), _) = metricStatsMap.get(ClassificationEvalMetric.areaUnderROC).get

    ((trainingAccuracy.mean, testAccuracy.mean), (trainingAUROC.mean, testAUROC.mean))
  }

  for {
    // run the logistic regression and get results
    lrResults <- mlService.classifyTimeSeries(finalDf, logisticRegressionSpec, temporalLearningSetting)

    // run the multi-layer perceptron and get results
    mlpResults <- mlService.classifyTimeSeries(finalDf, multiLayerPerceptronSpec, temporalLearningSetting)
  } yield {
    val ((lrTrainingAccuracy, lrTestAccuracy), (lrTrainingAUROC, lrTestAUROC)) = calcMeanAccuracyAndAUROC(lrResults)
    val ((mlpTrainingAccuracy, mlpTestAccuracy), (mlpTrainingAUROC, mlpTestAUROC)) = calcMeanAccuracyAndAUROC(mlpResults)

    println(s"Logistic Regression    Accuracy: $lrTrainingAccuracy / $lrTestAccuracy")
    println(s"Logistic Regression       AUROC: $lrTrainingAUROC / $lrTestAUROC")

    println(s"Multi-layer Perceptron Accuracy: $mlpTrainingAccuracy / $mlpTestAccuracy")
    println(s"Multi-layer Perceptron    AUROC: $mlpTrainingAUROC / $mlpTestAUROC")
  }
})