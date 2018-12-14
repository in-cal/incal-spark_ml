package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{collect_list, _}
import org.incal.spark_ml.SparkUtil.{assembleVectors, transformInPlace}

private class SlidingWindow(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("sliding_window"))

  protected final val windowSize: Param[Int] = new Param[Int](this, "windowSize", "Sliding window size")
  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setWindowSize(value: Int): this.type = set(windowSize, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOrderCol(value: String): this.type = set(orderCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val flattenVectors = udf {
    assembleVectors(_: Seq[Vector])
  }

  private def seqSizeEq(size: Int) = udf { seq: Seq[_] => seq.size == size }

  override def transform(dataset: Dataset[_]): DataFrame = {
    require($(windowSize) > 0, "Window size must be a positive integer.")

    val inputType = dataset.schema($(inputCol)).dataType

    // data frame with a sliding window
    val windowSpec = Window.orderBy($(orderCol)).rowsBetween(1 - $(windowSize), 0)
    val windowDf = dataset.withColumn($(outputCol), collect_list(dataset($(inputCol))).over(windowSpec))

    // remove init. entries with a fewer elements than the window size
    val smallerDf = windowDf.where(seqSizeEq($(windowSize))(windowDf($(outputCol))))

    // flatten vectors if of type
    inputType.typeName match {
      case "vector" => smallerDf.withColumn($(outputCol), flattenVectors(smallerDf($(outputCol))))
      case _ => smallerDf
    }
  }

  override def copy(extra: ParamMap): SlidingWindow = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    val existingFields = schema.fields

    val inputField = schema(inputColName)

    val outputType = inputField.dataType.typeName match {
      case "vector" => inputField.dataType
      case _ => ArrayType(inputField.dataType)
    }

    require(!existingFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    schema.add(StructField(outputColName, outputType, true))
  }
}

object SlidingWindow {

  def apply(
    inputCol: String,
    orderCol: String,
    outputCol: String)(
    windowSize: Int
  ): Transformer = new SlidingWindow().setWindowSize(windowSize).setInputCol(inputCol).setOrderCol(orderCol).setOutputCol(outputCol)

  def applyInPlace(
    inputOutputCol: String,
    orderCol: String)(
    windowSize: Int
  ): Estimator[PipelineModel] =
    transformInPlace(
      apply(inputOutputCol, orderCol, _)(windowSize),
      inputOutputCol
    )
}