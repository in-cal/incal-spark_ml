package org.incal.spark_ml.transformers

import scala.collection.mutable.WrappedArray

protected abstract class PerElementAccumVectorAgg extends VectorAgg {

  protected def updateBufferElement(
    buff: WrappedArray[Double],
    index: Int,
    input: Double
  ): Unit

  protected def initValue: Double

  protected def newInitBuffer(size: Int): WrappedArray[Double] =
    WrappedArray.make(Array.fill(size)(initValue))

  override protected def updateBuffer(
    buff: WrappedArray[Double],
    inputWithIndeces: Traversable[(Double, Int)],
    inputSize: Int
  ) = {
    // initialize buffer if empty
    val initBuff = if (buff.isEmpty) newInitBuffer(inputSize) else buff

    // update buffer elements with the inputs
    inputWithIndeces.foreach { case (input, index) => updateBufferElement(initBuff, index, input) }

    // return a new buffer
    initBuff
  }

  override protected def mergeBuffers(
    buff1: WrappedArray[Double],
    buff2: WrappedArray[Double]
  ) =
    updateBuffer(buff1, buff2.zipWithIndex, buff2.size)
}
