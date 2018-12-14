package org.incal.spark_ml.models

import java.{lang => jl}

import com.banda.math.domain.rand.RandomDistribution
import com.banda.network.domain.ActivationFunctionType
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq

case class ReservoirSpec(
  inputNodeNum: Int,
  bias: Double,
  nonBiasInitial: Double,
  reservoirNodeNum: ValueOrSeq[Int] = Left(None),
  reservoirInDegree: ValueOrSeq[Int] = Left(None),
  reservoirEdgesNum: ValueOrSeq[Int] = Left(None),
  reservoirInDegreeDistribution: Option[RandomDistribution[Integer]] = None,
  reservoirCircularInEdges: Option[Seq[Int]] = None,
  reservoirPreferentialAttachment: Boolean = false,
  reservoirBias: Boolean = false,
  reservoirAllowSelfEdges: Boolean = true,
  reservoirAllowMultiEdges: Boolean = false,
  inputReservoirConnectivity: ValueOrSeq[Double] = Left(None),
  weightDistribution: RandomDistribution[jl.Double],
  reservoirSpectralRadius: ValueOrSeq[Double] = Left(None),
  reservoirFunctionType: ActivationFunctionType,
  reservoirFunctionParams: Seq[Double] = Nil,
  perNodeReservoirFunctionWithParams: Option[Stream[(ActivationFunctionType, Seq[jl.Double])]] = None,
  washoutPeriod: ValueOrSeq[Int] = Left(None)
)