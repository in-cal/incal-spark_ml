package org.incal.spark_ml.transformers

import java.{lang => jl}

import com.bnd.network.business.NetworkRunnableFactoryUtil.NetworkRunnable
import com.bnd.network.domain.TopologicalNode

protected class NetworkStateVectorAgg(
  networkRunnable: NetworkRunnable[jl.Double],
  inputNodes: Seq[TopologicalNode],
  outputNodes: Seq[TopologicalNode]
) extends StateFunctionVectorAgg {

  override protected def newStateFun(
    inputWithIndeces: Traversable[(Double, Int)]
  ): Array[Double] = {
    // set the inputs
    inputWithIndeces.foreach { case (input, index) =>
      val inputNode = inputNodes(index)
      networkRunnable.setState(inputNode, input)
    }

    // run for one step
    networkRunnable.runFor(1)

    // get the output states
    outputNodes.map(outputNode => networkRunnable.getState(outputNode): Double).toArray
  }
}