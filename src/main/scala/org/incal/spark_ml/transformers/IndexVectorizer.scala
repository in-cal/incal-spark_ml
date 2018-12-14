package org.incal.spark_ml.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

class IndexVectorizer(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("vectorizer"))

  protected final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    require(schema(inputColName).dataType.isInstanceOf[NumericType],
      s"Input column must be of type NumericType but got ${schema(inputColName).dataType}")

    val inputFields = schema.fields

    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    val outputFields = inputFields ++ Seq(StructField(outputColName, ArrayType(DoubleType), true))
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    // If the number of attributes is unknown, we check the values from the input column.
    val numAttrs = dataset.select(col(inputColName).cast(DoubleType)).rdd.map(_.getDouble(0))
      .aggregate(0.0)(
        (m, x) => {
          assert(x <= Int.MaxValue,
            s"OneHotEncoder only supports up to ${Int.MaxValue} indices, but got $x")
          assert(x >= 0.0 && x == x.toInt,
            s"Values from column $inputColName must be indices, but got $x.")
          math.max(m, x)
        },
        (m0, m1) => {
          math.max(m0, m1)
        }
      ).toInt + 1

    // data transformation
    val size = numAttrs

    val oneValue = Array(1.0)
    val emptyValues = Array.empty[Double]
    val emptyIndices = Array.empty[Int]
    val encode = udf { label: Double =>
      if (label < size) {
        Vectors.sparse(size, Array(label.toInt), oneValue)
      } else {
        Vectors.sparse(size, emptyIndices, emptyValues)
      }
    }

    dataset.select(col("*"), encode(col(inputColName).cast(DoubleType)).as(outputColName, Metadata.empty))
  }

  override def copy(extra: ParamMap): IndexVectorizer = defaultCopy(extra)
}

object IndexVectorizer extends DefaultParamsReadable[IndexVectorizer] {

  override def load(path: String): IndexVectorizer = super.load(path)
}