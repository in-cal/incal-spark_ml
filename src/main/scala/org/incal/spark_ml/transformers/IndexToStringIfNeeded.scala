package org.incal.spark_ml.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class IndexToStringIfNeeded(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("index_to_string_if_needed"))

  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame =
    if (!dataset.columns.contains($(outputCol))) {
      val indexer = new IndexToString().setInputCol($(inputCol)).setOutputCol($(outputCol))
      indexer.transform(dataset)
    } else {
      dataset.toDF()
    }

  override def copy(extra: ParamMap): IndexToStringIfNeeded = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val labelField = NominalAttribute.defaultAttr.withName($(outputCol)).toStructField()
    val outputFields = schema.fields :+ labelField
    StructType(outputFields)
  }
}

object IndexToStringIfNeeded {

  def apply(
    inputCol: String,
    outputCol: String
  ): Transformer = new IndexToStringIfNeeded().setInputCol(inputCol).setOutputCol(outputCol)
}