package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.incal.spark_ml.SparkUtil
import org.incal.spark_ml.models.VectorScalerType

private class VectorHead(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("vector_head"))

  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val vectorHead = udf { vector: Vector => vector(0) }

  override def transform(dataset: Dataset[_]): DataFrame =
    if (!dataset.columns.contains($(outputCol))) {
      dataset.withColumn($(outputCol), vectorHead(dataset($(inputCol))))
    } else {
      dataset.toDF()
    }

  override def copy(extra: ParamMap): VectorHead = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    require(schema(inputColName).dataType.typeName == "vector",
      s"Input column must be of type Vector but got ${schema(inputColName).dataType}")

    val existingFields = schema.fields

    require(!existingFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    schema.add(StructField(outputColName, DoubleType, true))
  }
}

object VectorHead {

  def apply(
    inputCol: String,
    outputCol: String
  ): Transformer = new VectorHead().setInputCol(inputCol).setOutputCol(outputCol)

  def applyInPlace(
    inputOutputCol: String
  ): Estimator[PipelineModel] =
    SparkUtil.transformInPlace(
      apply(inputOutputCol, _),
      inputOutputCol
    )
}