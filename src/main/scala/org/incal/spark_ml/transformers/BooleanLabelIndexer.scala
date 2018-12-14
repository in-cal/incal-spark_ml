package org.incal.spark_ml.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{BooleanType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.Random

private class BooleanLabelIndexer(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("boolean_label_indexer"))

  protected final val stringLabelCol: Param[Option[String]] = new Param[Option[String]](this, "stringLabelCol", "string label column name")

  def setStringLabelCol(value: Option[String]): this.type = set(stringLabelCol, value)

  private val indexTempCol = "label_" + Random.nextInt()
  private val indexer = new FixedOrderStringIndexer().setLabels(Array("false", "true")).setInputCol("label").setOutputCol(indexTempCol)

  override def transform(dataset: Dataset[_]): DataFrame =
    dataset.schema("label").dataType match {
      case BooleanType =>
        val metadata = dataset.schema("label").metadata
        val stringDf = dataset.withColumn("label", dataset("label").cast(StringType).as("", metadata))

        val indexedDf = indexer.fit(stringDf).transform(stringDf)
        val indexedStringDf = $(stringLabelCol).map { stringCol =>
          indexedDf.withColumnRenamed("label", stringCol)
        }.getOrElse(
          indexedDf.drop("label")
        )
        indexedStringDf.withColumnRenamed(indexTempCol, "label")

      case _ => dataset.toDF()
    }

  override def copy(extra: ParamMap): BooleanLabelIndexer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val labelField = NominalAttribute.defaultAttr.withName("label").toStructField()
    val outputFields = schema.fields.filter(_.name != "label") :+ labelField
    StructType(outputFields)
  }
}

object BooleanLabelIndexer {

  def apply(stringLabelCol: Option[String] = None): Transformer = new BooleanLabelIndexer().setStringLabelCol(stringLabelCol)
}