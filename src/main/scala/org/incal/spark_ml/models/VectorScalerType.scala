package org.incal.spark_ml.models

object VectorScalerType extends Enumeration {
  val L1Normalizer, L2Normalizer, StandardScaler, MinMaxZeroOneScaler, MinMaxPlusMinusOneScaler, MaxAbsScaler = Value
}
