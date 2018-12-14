package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, PipelineModel, Transformer}
import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler, StandardScaler}
import org.incal.spark_ml.models.VectorScalerType
import org.incal.spark_ml.SparkUtil

object VectorColumnScalerNormalizer {

  def apply(
    transformType: VectorScalerType.Value
  ): Estimator[_  <: Transformer] =
    apply(transformType, "features", "scaledFeatures")

  def applyInPlace(
    transformType: VectorScalerType.Value,
    inputOutputCol: String
  ): Estimator[PipelineModel] =
    SparkUtil.transformInPlace(
      apply(transformType, inputOutputCol, _),
      inputOutputCol
    )

  def apply(
    transformType: VectorScalerType.Value,
    inputCol: String,
    outputCol: String
  ): Estimator[_  <: Transformer] =
    transformType match {
      case VectorScalerType.L1Normalizer =>
        Normalizer(1, inputCol, outputCol)

      case VectorScalerType.L2Normalizer =>
        Normalizer(2, inputCol, outputCol)

      case VectorScalerType.StandardScaler =>
        new StandardScaler()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
          .setWithStd(true)
          .setWithMean(true)

      case VectorScalerType.MinMaxPlusMinusOneScaler =>
        new MinMaxScaler()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
          .setMin(-1)
          .setMax(1)

      case VectorScalerType.MinMaxZeroOneScaler =>
        new MinMaxScaler()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
          .setMin(0)
          .setMax(1)

      case VectorScalerType.MaxAbsScaler =>
        new MaxAbsScaler()
          .setInputCol(inputCol)
          .setOutputCol(outputCol)
    }
}