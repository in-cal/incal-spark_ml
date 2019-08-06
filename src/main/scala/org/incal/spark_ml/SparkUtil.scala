package org.incal.spark_ml

import examples.SimpleClassification._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.functions.{monotonically_increasing_id, struct, udf}
import org.apache.spark.sql.types.{StructField, StructType}

import scala.reflect.ClassTag
import org.incal.spark_ml.transformers.{FixedOrderStringIndexer, SchemaUnchangedTransformer}

import scala.collection.mutable.ArrayBuilder
import scala.io.BufferedSource
import scala.util.Random

object SparkUtil {

  import scala.collection.JavaConversions._

  def transposeVectors(
    session: SparkSession,
    columnNames: Traversable[String],
    df: DataFrame
  ): DataFrame = {
    def vectorElement(i: Int) = udf { r: Row => r.getAs[Vector](0)(i) }

    // vectors for different columns are expected to have same size, otherwise transpose wouldn't work
    val vectorSize = df.select(columnNames.head).head().getAs[Vector](0).size

    val columns = columnNames.map { columnName =>
      for (i <- 0 until vectorSize) yield {
        vectorElement(i)(struct(columnName)).as(columnName + "_" + i)
      }
    }.flatten

    val newDf = df.select(columns.toSeq :_ *)

    val rows = for (i <- 0 until vectorSize) yield {
      val vectors = columnNames.map { columnName =>
        val values = newDf.select(columnName + "_" + i).collect().map(_.getDouble(0))
        Vectors.dense(values)
      }

      Row.fromSeq(vectors.toSeq)
    }

    val columnTypes = columnNames.map(columnName => df.schema(columnName))
    session.createDataFrame(rows, StructType(columnTypes.toSeq))
  }

  // adapted from org.apache.spark.ml.feature.VectorAssembler.assemble
  def assembleVectors(vv: Seq[Vector]): Vector = {
    val indices = ArrayBuilder.make[Int]
    val values = ArrayBuilder.make[Double]
    var cur = 0

    vv.foreach { vec: Vector =>
      vec.foreachActive { case (i, v) =>
        if (v != 0.0) {
          indices += cur + i
          values += v
        }
      }
      cur += vec.size
    }
    Vectors.sparse(cur, indices.result(), values.result()).compressed
  }

  def transformInPlace(
    outputColumnToStage: String => PipelineStage,
    inputOutputCol: String
  ): Estimator[PipelineModel] = {
    val tempOutputCol = inputOutputCol + Random.nextLong()
    val stage = outputColumnToStage(tempOutputCol)

    val renameColumn = SchemaUnchangedTransformer(
      _.drop(inputOutputCol).withColumnRenamed(tempOutputCol, inputOutputCol)
    )

    new Pipeline().setStages(Array(stage, renameColumn))
  }

  def transformInPlaceWithParamGrids(
    outputColumnToStage: String => (PipelineStage, Traversable[ParamGrid[_]]),
    inputOutputCol: String
  ): (Estimator[PipelineModel], Traversable[ParamGrid[_]]) = {
    val tempOutputCol = inputOutputCol + Random.nextLong()
    val (stage, paramGrids) = outputColumnToStage(tempOutputCol)

    val renameColumn = SchemaUnchangedTransformer(
      _.drop(inputOutputCol).withColumnRenamed(tempOutputCol, inputOutputCol)
    )

    val inPlaceTransformer = new Pipeline().setStages(Array(stage, renameColumn))
    (inPlaceTransformer, paramGrids)
  }

  def joinByOrder(df1: DataFrame, df2: DataFrame) = {
    val joinColumnName = "_id" + Random.nextLong()

    // aux function
    def withOrderColumn(df: DataFrame) =
      df.withColumn(joinColumnName, monotonically_increasing_id())

    withOrderColumn(df1).join(withOrderColumn(df2), joinColumnName).drop(joinColumnName)
  }

  implicit class VectorMap(vector: Vector) {
    def map(fun: Double => Double) =
      vector match {
        case DenseVector(vs) =>
          val values = vs.map(fun)
          Vectors.dense(values)

        case SparseVector(size, indeces, vs) =>
          val values = vs.map(fun)
          Vectors.sparse(size, indeces, values)

        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }

    def mapWithIndex(fun: (Double, Int) => Double) = {
      val tuppledFun = (fun(_, _)).tupled
      vector match {
        case DenseVector(vs) =>
          val values = vs.zipWithIndex.map(tuppledFun)
          Vectors.dense(values)

        case SparseVector(size, indeces, vs) =>
          val values = vs.zip(indeces).map(tuppledFun)
          Vectors.sparse(size, indeces, values)

        case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
      }
    }
  }

  implicit def kryoEncoder[A](implicit ct: ClassTag[A]) = Encoders.kryo[A](ct)

  def prepFeaturesDataFrame(
    featureFieldNames: Set[String],
    outputFieldName: Option[String],
    dropFeatureCols: Boolean = true,
    dropNaValues: Boolean = true)(
    df: DataFrame
  ): DataFrame = {
    // drop null values
    val nonNullDf = if (dropNaValues) df.na.drop else df

    val existingFeatureCols = nonNullDf.columns.filter(featureFieldNames.contains)

    val assembler = new VectorAssembler()
      .setInputCols(existingFeatureCols)
      .setOutputCol("features")

    val featuresDf = assembler.transform(nonNullDf)

    val finalDf = outputFieldName.map(
      featuresDf.withColumnRenamed(_, "label")
    ).getOrElse(
      featuresDf
    )

    if (dropFeatureCols) {
      val columnsToDrop = outputFieldName.map { outputFieldName =>
        existingFeatureCols.filterNot(_.equals(outputFieldName))
      }.getOrElse(
        existingFeatureCols
      )
      finalDf.drop(columnsToDrop: _ *)
    } else
      finalDf
  }

  def indexStringCols(
    columnNameWithEnumLabels: Seq[(String, Seq[String])])(
    df: DataFrame
  ) =
    columnNameWithEnumLabels.foldLeft(df){ case (newDf, (columnName, enumLabels)) =>
      val tempCol = columnName + Random.nextLong()

      // if enum labels provided create an fixed-order string indexer, otherwise use a standard one, which index values based on their frequencies
      val indexer = if (enumLabels.nonEmpty) {
        new FixedOrderStringIndexer().setLabels(enumLabels.toArray).setInputCol(columnName).setOutputCol(tempCol).setHandleInvalid("skip")
      } else
        new StringIndexer().setInputCol(columnName).setOutputCol(tempCol).setHandleInvalid("skip")

      indexer.fit(newDf).transform(newDf).drop(columnName).withColumnRenamed(tempCol, columnName)
    }

  def localCsvToDataFrame(fileName: String, header: Boolean = true)(session: SparkSession) = {
    val src = scala.io.Source.fromFile(fileName)
    csvSourceToDf(src, header)(session)
  }

  def remoteCsvToDataFrame(url: String, header: Boolean = true)(session: SparkSession) = {
    val src = scala.io.Source.fromURL(url)
    csvSourceToDf(src, header)(session)
  }

  private def csvSourceToDf(src: BufferedSource, header: Boolean = true)(session: SparkSession) = {
    import session.sqlContext.implicits._

    val csvData: Dataset[String] = session.sparkContext.parallelize(src.mkString.stripMargin.lines.toList).toDS()
    session.read.option("header", header).option("inferSchema", true).csv(csvData)
  }
}