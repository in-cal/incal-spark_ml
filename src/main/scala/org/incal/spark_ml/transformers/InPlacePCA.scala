package org.incal.spark_ml.transformers

import org.apache.spark.ml._
import org.apache.spark.ml.feature.PCA
import org.incal.spark_ml.SparkUtil

object InPlacePCA {

  private val inputOutputCol = "features"

  def apply(k: Int): Estimator[PipelineModel] = {

    val createPCA = (outputCol: String) => new PCA()
      .setInputCol(inputOutputCol)
      .setOutputCol(outputCol)
      .setK(k)

    SparkUtil.transformInPlace(createPCA, inputOutputCol)
  }
}
