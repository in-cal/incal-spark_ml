package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, PipelineModel, Transformer}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{col, last}
import org.incal.spark_ml.SparkUtil.transformInPlace

/**
  * Moves each input column value of a data frame with an order column to a next row defined by a given shift.
  *
  * @author Peter Banda
  * @since 2018
  */
private class SeqShift(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("seq_shift"))

  protected final val shift: Param[Int] = new Param[Int](this, "shift", "shift", ParamValidators.gt(0))
  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  protected final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name")

  def setShift(value: Int): this.type = set(shift, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOrderCol(value: String): this.type = set(orderCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setGroupCol(value: Option[String]) = value.map(set(groupCol, _)).getOrElse(SeqShift.this)

  override def transform(dataset: Dataset[_]): DataFrame = {

    // data frame with a sliding window
    val windowBaseSpec = get(groupCol) match {
      case Some(groupCol) => Window.partitionBy(groupCol).orderBy($(orderCol))
      case None => Window.orderBy($(orderCol))
    }

    val windowSpec = windowBaseSpec.rowsBetween(0, $(shift))
    val shiftedDf = dataset.withColumn($(outputCol), last(dataset($(inputCol))).over(windowSpec))

    // drop the last "shift" items
    val orders = get(groupCol) match {
      case Some(_) => dataset.select(col($(orderCol))).distinct
      case None => dataset.select(col($(orderCol)))
    }

    val cutOrderValue = orders
      .orderBy(dataset($(orderCol)).desc)
      .limit($(shift))
      .collect().last.getInt(0)

    shiftedDf.where(dataset($(orderCol)) < cutOrderValue)
  }

  override def copy(extra: ParamMap): SeqShift = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    val existingFields = schema.fields

    val inputField = schema(inputColName)
    val outputField = inputField.copy(name = outputColName)

    require(!existingFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    schema.add(outputField)
  }
}

object SeqShift {

  def apply(
    inputCol: String,
    orderCol: String,
    outputCol: String,
    groupCol: Option[String] = None)(
    shift: Int
  ): Transformer = new SeqShift().setShift(shift).setInputCol(inputCol).setOrderCol(orderCol).setOutputCol(outputCol).setGroupCol(groupCol)

  def applyInPlace(
    inputOutputCol: String,
    orderCol: String,
    groupCol: Option[String] = None)(
    shift: Int
  ): Estimator[PipelineModel] =
    transformInPlace(
      apply(inputOutputCol, orderCol, _, groupCol)(shift),
      inputOutputCol
    )
}