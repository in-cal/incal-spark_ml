package org.incal.spark_ml.models.classification

object ClassificationEvalMetric extends Enumeration {
  val f1, weightedPrecision, weightedRecall, accuracy, areaUnderROC, areaUnderPR = Value
}
