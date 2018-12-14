package org.incal.spark_ml.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.min
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.incal.spark_ml.ParamGrid
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq

class DropSeriesLeft(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("drop_left"))

  protected[spark_ml] final val count: Param[Int] = new Param[Int](this, "count", "Number of elements to drop", ParamValidators.gtEq(0))

  protected final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")

  def setCount(value: Int): this.type = set(count, value)
  def setOrderCol(value: String): this.type = set(orderCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    println("Drop left count: " + get(count))
    val minOrder = dataset.agg(min($(orderCol))).head.getInt(0)
    dataset.where(dataset($(orderCol)) > minOrder + $(count)).toDF()
  }

  override def copy(extra: ParamMap): DropSeriesLeft = defaultCopy(extra)

  override def transformSchema(schema: StructType) = schema
}

object DropSeriesLeft {

  def apply(
    orderCol: String)(
    count: ValueOrSeq[Int] = Left(None)
  ): (Transformer, Traversable[ParamGrid[_]]) = {
    val transformer = new DropSeriesLeft().setOrderCol(orderCol)

    val paramGrids = count match {
      case Left(value) => value.foreach(transformer.setCount); Nil
      case Right(values) => Seq(ParamGrid(transformer.count, values))
    }

    (transformer, paramGrids)
  }
}