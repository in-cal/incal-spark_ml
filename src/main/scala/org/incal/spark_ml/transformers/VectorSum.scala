package org.incal.spark_ml.transformers

import scala.collection.mutable.WrappedArray

object VectorSum extends PerElementAccumVectorAgg {
  override protected def initValue = 0d

  override protected def updateBufferElement(
    buff: WrappedArray[Double],
    index: Int,
    input: Double
  ) =
    buff(index) += input
}

object VectorMax extends PerElementAccumVectorAgg {
  override protected def initValue = Double.MinValue

  override protected def updateBufferElement(
    buff: WrappedArray[Double],
    index: Int,
    input: Double
  ) =
    buff(index) = math.max(buff(index), input)
}

object VectorMin extends PerElementAccumVectorAgg {
  override protected def initValue = Double.MaxValue

  override protected def updateBufferElement(
    buff: WrappedArray[Double],
    index: Int,
    input: Double
  ) =
    buff(index) = math.min(buff(index), input)
}