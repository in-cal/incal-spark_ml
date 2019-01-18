package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, PipelineModel, Transformer}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.incal.spark_ml.SparkUtil.transformInPlace

/**
  * Alternative implementation of <code>SeqShift</code> that assumes the order column contains strictly consecutive values.
  *
  * @author Peter Banda
  * @since 2018
  */
private class SeqShiftWithConsecutiveOrder(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("seq_shift_with_consecutive_order"))

  protected final val shift: Param[Int] = new Param[Int](this, "shift", "shift", ParamValidators.gt(0))
  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  protected final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name")

  def setShift(value: Int): this.type = set(shift, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOrderCol(value: String): this.type = set(orderCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setGroupCol(value: Option[String]) = value.map(set(groupCol, _)).getOrElse(SeqShiftWithConsecutiveOrder.this)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    val inputOrderDf = get(groupCol) match {
      case Some(groupCol) => df.select(df($(orderCol)), df($(inputCol)).as($(outputCol)), df(groupCol))
      case None => df.select(df($(orderCol)), df($(inputCol)).as($(outputCol)))
    }

    val shiftOrderDf = inputOrderDf.withColumn($(orderCol), df($(orderCol)) - $(shift))

    get(groupCol) match {
      case Some(groupCol) => df.join(shiftOrderDf, Seq(groupCol, $(orderCol)))
      case None => df.join(shiftOrderDf, $(orderCol))
    }
  }

  override def copy(extra: ParamMap): SeqShiftWithConsecutiveOrder = defaultCopy(extra)

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

object SeqShiftWithConsecutiveOrder {

  def apply(
    inputCol: String,
    orderCol: String,
    outputCol: String,
    groupCol: Option[String] = None)(
    shift: Int
  ): Transformer = new SeqShiftWithConsecutiveOrder().setShift(shift).setInputCol(inputCol).setOrderCol(orderCol).setOutputCol(outputCol).setGroupCol(groupCol)

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