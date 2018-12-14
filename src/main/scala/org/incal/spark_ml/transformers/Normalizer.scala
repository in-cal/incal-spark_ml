package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.linalg.{DenseVector, SQLDataTypes, SparseVector, Vector}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.incal.spark_ml.SparkUtil.VectorMap

import scala.util.Random

private class Normalizer(override val uid: String) extends Estimator[NormalizerModel] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("normalizer"))

  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  protected final val p = new DoubleParam(this, "p", "the p norm value", ParamValidators.gtEq(1))

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setP(value: Double): this.type = set(p, value)

  override def fit(dataset: Dataset[_]): NormalizerModel = {
    transformSchema(dataset.schema, logging = true)

    val norm = VectorNorm(dataset.toDF(), $(inputCol), $(p))
    copyValues(new NormalizerModel(uid, norm).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputDataType = schema($(inputCol)).dataType

    require(inputDataType.equals(SQLDataTypes.VectorType),
      s"Column ${$(inputCol)} must be of type vector but was actually ${inputDataType}")

    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    val outputFields = schema.fields :+ StructField($(outputCol), SQLDataTypes.VectorType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): Normalizer = defaultCopy(extra)
}

class NormalizerModel(override val uid: String, norms: Vector) extends Model[NormalizerModel] with DefaultParamsWritable {

  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private def vectorDiv(divs: Vector) =
    udf {
      vector: Vector => vector.mapWithIndex { case (value, index) =>
        val div = divs(index)
        if (div != 0.0)
          value / div
        else
          value
      }
    }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    dataset.withColumn($(outputCol), vectorDiv(norms)(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputDataType = schema($(inputCol)).dataType

    require(inputDataType.equals(SQLDataTypes.VectorType),
      s"Column ${$(inputCol)} must be of type vector but was actually ${inputDataType}")

    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    val outputFields = schema.fields :+ StructField($(outputCol), SQLDataTypes.VectorType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): NormalizerModel = {
    val copied = new NormalizerModel(uid, norms)
    copyValues(copied, extra).setParent(parent)
  }
}

object Normalizer {

  def apply(
    p: Double,
    inputCol: String,
    outputCol: String
  ): Estimator[NormalizerModel] = new Normalizer().setP(p).setInputCol(inputCol).setOutputCol(outputCol)
}

object VectorNorm {

  private val vectorUdf =
    (fun: Double => Double) =>
      udf { vector: Vector => vector.map(fun) }

  private val vectorAbs = vectorUdf(math.abs)
  private val vectorSquare = vectorUdf(value => value * value)
  private def vectorPow(p: Double) = vectorUdf(value => math.pow(value, p))

  def apply(df: DataFrame, inputCol: String, p: Double): Vector = {
    require(p >= 1.0, "To compute the p-norm of the vector, we require that you specify a p>=1. " +
      s"You specified p=$p.")

    val tempCol = inputCol + Random.nextLong()

    val absDf = df.select(df(inputCol)).withColumn(tempCol, vectorAbs(df(inputCol)))

    // aux function to calc sum
    def calcSums(dataFrame: DataFrame) = {
      val result = dataFrame.agg(VectorSum(dataFrame(tempCol)))
      result.head.getAs[Vector](0)
    }

    p match {
      case 1 =>
        calcSums(absDf)

      case 2 =>
        val squareDf = absDf.withColumn(tempCol, vectorSquare(absDf(tempCol)))
        val sums = calcSums(squareDf)
        sums.map(math.sqrt)

      case Double.PositiveInfinity =>
        absDf.agg(VectorMax(absDf(tempCol))).head.getAs[Vector](0)

      case _ =>
        val powDf = absDf.withColumn(tempCol, vectorPow(p)(absDf(tempCol)))
        val sums = calcSums(powDf)
        sums.map(math.pow(_, 1.0 / p))
    }
  }
}