package org.incal.spark_ml.transformers

import java.util.UUID

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

trait SchemaUnchangedTransformer extends Transformer with DefaultParamsWritable {

  protected def transformDataFrame(dataFrame: DataFrame): DataFrame

  override def transform(dataset: Dataset[_]): DataFrame =
    transformDataFrame(dataset.asInstanceOf[DataFrame])

  override def copy(extra: ParamMap): Transformer =
    this

  override def transformSchema(schema: StructType): StructType =
    schema

  override val uid: String =
    UUID.randomUUID.toString
}

private class SchemaUnchangedTransformerAdapter(_transform: DataFrame => DataFrame) extends SchemaUnchangedTransformer {

  override protected def transformDataFrame(dataFrame: DataFrame) = _transform(dataFrame)
}

object SchemaUnchangedTransformer {

  def apply(_transform: DataFrame => DataFrame): Transformer = new SchemaUnchangedTransformerAdapter(_transform)
}
