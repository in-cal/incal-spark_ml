package org.incal.spark_ml.transformers

import scala.collection.mutable.WrappedArray

protected abstract class StateFunctionVectorAgg extends VectorAgg {

  protected def newStateFun(
    inputWithIndeces: Traversable[(Double, Int)]
  ): Array[Double]

  override protected def updateBuffer(
    buff: WrappedArray[Double],
    inputWithIndeces: Traversable[(Double, Int)],
    inputSize: Int
  ) =
    newStateFun(inputWithIndeces)

  override protected def mergeBuffers(
    buff1: WrappedArray[Double],
    buff2: WrappedArray[Double]
  ) = throw new IllegalArgumentException("Cannot merge buffers for StateFunctionVectorAgg because of statefulness.")
}

protected abstract class ScalarStateFunctionVectorAgg extends StateFunctionVectorAgg {

  protected def newScalarStateFun(
    inputWithIndeces: Traversable[(Double, Int)]
  ): Double

  override protected def newStateFun(
    inputWithIndeces: Traversable[(Double, Int)]
  ) = Array(newScalarStateFun(inputWithIndeces))
}

protected class ScalarStateFunctionVectorAggAdapter(
  initState: Double)(
  updateState: (Double, Traversable[Double]) => Double
) extends ScalarStateFunctionVectorAgg {
  val stateHolder = StateHolder[Double, Traversable[Double]](initState) {
    case (state, inputs) => updateState(state, inputs)
  }

  override protected def newScalarStateFun(inputWithIndeces: Traversable[(Double, Int)]): Double =
    stateHolder.update(inputWithIndeces.map(_._1))
}

object ScalarStateFunctionVectorAgg {

  val min: VectorAgg =
    new ScalarStateFunctionVectorAggAdapter(Double.MaxValue)( (min, inputs) => Math.min(min, inputs.min) )

  val max: VectorAgg =
    new ScalarStateFunctionVectorAggAdapter(Double.MinValue)( (max, inputs) => Math.max(max, inputs.max) )

  val sum: VectorAgg =
    new ScalarStateFunctionVectorAggAdapter(0d)( (sum, inputs) => sum + inputs.sum )
}

case class StateHolder[S, IN](initState: S)(updateState: (S, IN) => S) {
  var state = initState

  def update(input: IN): S = {
    state = updateState(state, input)
    state
  }
}