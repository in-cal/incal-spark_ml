package examples

import com.bnd.math.domain.rand.RandomDistribution
import com.bnd.network.domain.ActivationFunctionType
import org.apache.spark.sql.SparkSession
import org.incal.core.{PlotSetting, PlotlyPlotter}
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.{ReservoirSpec, TreeCore}
import org.incal.spark_ml.models.regression._
import org.incal.spark_ml.models.result.RegressionResultsHolder
import org.incal.spark_ml.models.setting.{RegressionLearningSetting, TemporalRegressionLearningSetting}
import org.incal.spark_ml.{MLResultUtil, SparkMLApp, SparkMLService}

import scala.concurrent.ExecutionContext.Implicits.global

object TemporalRegressionWithReservoirKernel extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val index, Date, SP500, Dividend, Earnings, ConsumerPriceIndex, LongInterestRate, RealPrice, RealDividend, RealEarnings, PE10,
    SP500Change, DividendChange, EarningsChange, ConsumerPriceIndexChange, LongInterestRateChange, RealPriceChange, RealDividendChange, RealEarningsChange, PE10Change = Value
  }

  val featureColumnNames = Seq(
    Column.SP500Change, Column.DividendChange, Column.EarningsChange, Column.ConsumerPriceIndexChange,
    Column.LongInterestRateChange, Column.RealPriceChange, Column.RealDividendChange, Column.RealEarningsChange, Column.PE10Change
  ).map(_.toString)
  val outputColumnName = Column.SP500Change.toString

  // read a csv and create a data frame with given column names
  val url = "https://bit.ly/2OmhfOD" // SAP
  val df = remoteCsvToDataFrame(url, true)(session)

  // turn the data frame into ML-ready one with features and a label
  val finalDf = prepFeaturesDataFrame(featureColumnNames.toSet, Some(outputColumnName))(df)

  // linear regression spec
  val linearRegressionSpec = LinearRegression(
    maxIteration = Left(Some(200)),
    regularization = Right(Seq(1, 0.1, 0.01, 0.001)),
    elasticMixingRatio = Right(Seq(0, 0.5, 1))
  )

  // random regression forest spec
  val randomRegressionForestSpec = RandomRegressionForest(
    core = TreeCore(maxDepth = Right(Seq(4,5,6)))
  )

  // gradient-boost regression tree spec
  val gradientBoostRegressionTreeSpec = GradientBoostRegressionTree(
    core = TreeCore(maxDepth = Right(Seq(4,5,6))),
    maxIteration = Left(Some(50))
  )

  // learning setting
  val regressionLearningSetting = RegressionLearningSetting(
    trainingTestSplitRatio = Some(0.8),
//    featuresNormalizationType = Some(VectorScalerType.StandardScaler),
    crossValidationEvalMetric = Some(RegressionEvalMetric.rmse),
    crossValidationFolds = Some(5),
    collectOutputs = true
  )

  // reservoir kernel
  val reservoirSpec = ReservoirSpec(
    inputNodeNum = featureColumnNames.size,
    bias = 1,
    nonBiasInitial = 0,
    reservoirNodeNum = Left(Some(20)), // Right(Seq(10,20,50)),
    reservoirInDegree = Left(Some(20)),
    inputReservoirConnectivity = Left(Some(1)),
    weightDistribution = RandomDistribution.createNormalDistribution(classOf[java.lang.Double], 0d, 1d),
    reservoirSpectralRadius = Right(Seq(0.9, 0.95, 0.99)),
    reservoirFunctionType = ActivationFunctionType.Tanh,
    washoutPeriod = Left(Some(50))
  )

  val temporalLearningSetting = TemporalRegressionLearningSetting(
    core = regressionLearningSetting,
    predictAhead = 1,
    reservoirSetting = Some(reservoirSpec)
//    slidingWindowSize = Right(Seq(5,10,20))
  )

  // aux function to get a mean training and test RMSE and MAE
  def calcMeanRMSEAndMAE(results: RegressionResultsHolder) = {
    val metricStatsMap = MLResultUtil.calcMetricStats(results.performanceResults)
    val (trainingRMSE, Some(testRMSE), _) = metricStatsMap.get(RegressionEvalMetric.rmse).get
    val (trainingMAE, Some(testMAE), _) = metricStatsMap.get(RegressionEvalMetric.mae).get

    ((trainingRMSE.mean, testRMSE.mean), (trainingMAE.mean, testMAE.mean))
  }

  // aux function to export outputs using GNU plot (must be installed)
  def exportOutputs(results: RegressionResultsHolder, fileName: String, size: Int) = {
    val outputs = results.expectedActualOutputs.head
    val trainingOutputs = outputs._1
    val testOutputs = outputs._2

    export(trainingOutputs, "training")
    export(testOutputs, "test")

    def export(outputsx: Seq[(Double, Double)], prefix: String) = {
      val y = outputsx.map { case (_, y) => y }.take(size)
      val yhat = outputsx.map { case (yhat, _) => yhat }.take(size)

      PlotlyPlotter.plotLines(
        data = Seq(y, yhat),
        setting = PlotSetting(
          title = Some("Outputs"),
          xLabel = Some("Time"),
          yLabel = Some("Value"),
          showLegend = true,
          captions = Seq("Actual Output", "Expected Output")
        ),
        outputFileName = prefix + "-" + fileName
      )
    }
  }

  for {
    // run the linear regression and get results
    lrResults <- mlService.regressTimeSeries(finalDf, linearRegressionSpec, temporalLearningSetting)

    // run the random regression forest and get results
    rrfResults <- mlService.regressTimeSeries(finalDf, randomRegressionForestSpec, temporalLearningSetting)

    // run the gradient boost regression and get results
    gbrtResults <- mlService.regressTimeSeries(finalDf, gradientBoostRegressionTreeSpec, temporalLearningSetting)
  } yield {
    val ((lrTrainingRMSE, lrTestRMSE), (lrTrainingMAE, lrTestMAE)) = calcMeanRMSEAndMAE(lrResults)
    val ((rrfTrainingRMSE, rrfTestRMSE), (rrfTrainingMAE, rrfTestMAE)) = calcMeanRMSEAndMAE(rrfResults)
    val ((gbrtTrainingRMSE, gbrtTestRMSE), (gbrtTrainingMAE, gbrtTestMAE)) = calcMeanRMSEAndMAE(gbrtResults)

    println(s"Linear Regression         RMSE: $lrTrainingRMSE / $lrTestRMSE")
    println(s"Linear Regression          MAE: $lrTrainingMAE / $lrTestMAE")
    println(s"Random Regression Forest  RMSE: $rrfTrainingRMSE / $rrfTestRMSE")
    println(s"Random Regression Forest   MAE: $rrfTrainingMAE / $rrfTestMAE")
    println(s"Gradient Boost Regression RMSE: $gbrtTrainingRMSE / $gbrtTestRMSE")
    println(s"Gradient Boost Regression  MAE: $gbrtTrainingMAE / $gbrtTestMAE")

    exportOutputs(lrResults, "lrOutputs.html", 300)
    exportOutputs(rrfResults, "rrfOutputs.html", 300)
    exportOutputs(gbrtResults, "gbrtOutputs.html", 300)
  }
})