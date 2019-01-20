package org.incal.spark_ml.transformers

import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.incal.spark_ml.SparkUtil
import org.incal.spark_ml.models.VectorScalerType

object NumericColumnScaler {

  def apply(
    scalerType: VectorScalerType.Value,
    inputCol: String,
    outputCol: String
  ): Estimator[PipelineModel] = {
    val vectorize = new VectorAssembler()
      .setInputCols(Array(inputCol))
      .setOutputCol(outputCol)

    val scaler = VectorColumnScaler.applyInPlace(scalerType, outputCol)

    val vectorHead = VectorHead.applyInPlace(outputCol)

    new Pipeline().setStages(Array(vectorize, scaler, vectorHead))
  }

  def applyInPlace(
    scalerType: VectorScalerType.Value,
    inputOutputCol: String
  ): Estimator[PipelineModel] =
    SparkUtil.transformInPlace(
      apply(scalerType, inputOutputCol, _),
      inputOutputCol
    )
}
