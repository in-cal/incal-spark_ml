package org.incal.spark_ml.transformers

import org.apache.spark.ml.linalg.{DenseVector, SQLDataTypes, SparseVector, Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}

import scala.collection.mutable.WrappedArray

protected abstract class VectorAgg extends UserDefinedAggregateFunction {

  protected def updateBuffer(
    buff: WrappedArray[Double],
    inputWithIndeces: Traversable[(Double, Int)],
    inputSize: Int
  ): WrappedArray[Double]

  protected def mergeBuffers(
    buff1: WrappedArray[Double],
    buff2: WrappedArray[Double]
  ): WrappedArray[Double]

  override def inputSchema = StructType(StructField("vec", SQLDataTypes.VectorType) :: Nil)

  override def bufferSchema = StructType(StructField("agg", ArrayType(DoubleType)) :: Nil)

  override def dataType = SQLDataTypes.VectorType

  override def deterministic: Boolean = true

  def initialize(buffer: MutableAggregationBuffer) =
    buffer.update(0, Array.empty[DoubleType])

  def update(buffer: MutableAggregationBuffer, input: Row) = {
    if (!input.isNullAt(0)) {
      val inputVector = input.getAs[Vector](0)
      val buff = buffer.getAs[WrappedArray[Double]](0)

      val newBuff = inputVector match {
        case DenseVector(values) => updateBuffer(buff, values.zipWithIndex, inputVector.size)

        case SparseVector(_, indices, values) => updateBuffer(buff, values.zip(indices), inputVector.size)
      }

      buffer.update(0, newBuff)
    }
  }

  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    val buff1 = buffer1.getAs[WrappedArray[Double]](0)
    val buff2 = buffer2.getAs[WrappedArray[Double]](0)

    val newBuff = mergeBuffers(buff1, buff2)
    buffer1.update(0, newBuff)
  }

  def evaluate(buffer: Row) =  Vectors.dense(buffer.getAs[WrappedArray[Double]](0).toArray)
}