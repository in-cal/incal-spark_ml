package org.incal.spark_ml

import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.functions.{monotonically_increasing_id, struct, udf}
import org.apache.spark.sql.types.StructType

import scala.reflect.ClassTag
import org.apache.spark.sql.Encoders
import org.incal.spark_ml.transformers.SchemaUnchangedTransformer

import scala.collection.mutable.ArrayBuilder
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
}