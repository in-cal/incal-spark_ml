package examples

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.incal.spark_ml.SparkUtil._
import org.incal.spark_ml.models.clustering.{BisectingKMeans, KMeans}
import org.incal.spark_ml.{SparkMLApp, SparkMLService}
import org.incal.core.util.{GroupMapList3, nonAlphanumericToUnderscore, toHumanReadableCamel}
import org.incal.core.{PlotSetting, PlotlyPlotter}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

object SimpleClustering extends SparkMLApp((session: SparkSession, mlService: SparkMLService) => {

  object Column extends Enumeration {
    val sepalLength, sepalWidth, petalLength, petalWidth, clazz = Value
  }

  val firstResultColumn = Column.sepalLength.toString
  val secondResultColumn = Column.sepalWidth.toString

  val columnNames = Column.values.toSeq.sortBy(_.id).map(_.toString)
  val featureColumnNames = columnNames.filter(_ != Column.clazz.toString) // all except clazz name

  // read a csv and create a data frame with given column names
  val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  val df = remoteCsvToDataFrame(url, false)(session).toDF(columnNames :_*)

  // turn the data frame into ML-ready one with features and a label
  val finalDf = prepFeaturesDataFrame(featureColumnNames.toSet, None, false)(df)

  // k-means spec
  val kMeansSpec = KMeans(k = 3)

  // bisecting k-means
  val bisKMeansSpec = BisectingKMeans(k = 3)

  // helper function to plot the results
  def plotResults(resultsDf: DataFrame, title: String) = {
    val results = resultsDf.select(firstResultColumn, secondResultColumn, "cluster").map { r =>
      val length = r(0).asInstanceOf[Double]
      val width = r(1).asInstanceOf[Double]
      val clusterClass = r(2).asInstanceOf[Int]
      (clusterClass + 1, length, width)
    }.collect.toSeq.toGroupMap.toSeq.sortBy(_._1)

    PlotlyPlotter.plotScatter(
      results.map(_._2),
      PlotSetting(
        title = Some(title),
        xLabel = Some(toHumanReadableCamel(firstResultColumn)),
        yLabel = Some(toHumanReadableCamel(secondResultColumn)),
        captions = results.map(res => "Cluster " + res._1.toString).toSeq
      ),
      outputFileName = nonAlphanumericToUnderscore(title.toLowerCase) + ".html"
    )
  }

  Future {
    // run k-means
    val kMeansResultsDf = mlService.cluster(finalDf, kMeansSpec)

    // run bisecting k-means
    val bisKMeansResultsDf = mlService.cluster(finalDf, bisKMeansSpec)

    plotResults(kMeansResultsDf, "IRIS k-Means Clusters")
    plotResults(bisKMeansResultsDf, "IRIS Bisecting k-Means Clusters")
  }
})