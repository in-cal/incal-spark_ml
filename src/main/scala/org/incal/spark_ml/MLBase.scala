package org.incal.spark_ml

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.min
import org.incal.spark_ml.models.classification.ClassificationEvalMetric
import org.incal.spark_ml.models.regression.RegressionEvalMetric

trait MLBase {

  // Training/Test Splits

  protected val randomSplit = (splitRatio: Double) => (dataFrame: DataFrame) => {
    val Array(training, test) = dataFrame.randomSplit(Array(splitRatio, 1 - splitRatio))
    (training, test)
  }

  protected val seqSplit = (orderColumn: String) => (splitRatio: Double) => (df: DataFrame) => {
    val splitValue = df.stat.approxQuantile(orderColumn, Array(splitRatio), 0.001)(0)
    val headDf = df.where(df(orderColumn) <= splitValue)
    val tailDf = df.where(df(orderColumn) > splitValue)
    (headDf, tailDf)
  }

  protected val splitByValue = (orderColumn: String) => (splitValue: Double) => (df: DataFrame) => {
    val headDf = df.where(df(orderColumn) <= splitValue)
    val tailDf = df.where(df(orderColumn) > splitValue)
    (headDf, tailDf)
  }

  // Predictions

  protected val independentTestPredictions =
    (mlModel: Transformer, testDf: Dataset[_], _: Dataset[_]) => mlModel.transform(testDf)

  protected val orderDependentTestPredictions = (orderColumn: String) =>
    (mlModel: Transformer, testDf: Dataset[_], mainDf: Dataset[_]) => {
      val allPredictions = mlModel.transform(mainDf)

      // either extract a min test index value or if empty return +infinity which will produce empty predictions
      val minTestIndexRow = testDf.agg(min(testDf(orderColumn))).head()
      val minTestIndexVal = if (!minTestIndexRow.isNullAt(0)) minTestIndexRow.getInt(0) else Int.MaxValue

      allPredictions.where(allPredictions(orderColumn) >= minTestIndexVal)
    }

  protected val orderDependentTestPredictionsWithParams = (orderColumn: String) =>
    (mlModel: Transformer, testDf: Dataset[_], mainDf: Dataset[_], paramMap: ParamMap) => {
      val allPredictions = mlModel.transform(mainDf, paramMap)

      // either extract a min test index value or if empty return +infinity which will produce empty predictions
      val minTestIndexRow = testDf.agg(min(testDf(orderColumn))).head()
      val minTestIndexVal = if (!minTestIndexRow.isNullAt(0)) minTestIndexRow.getInt(0) else Int.MaxValue

      allPredictions.where(allPredictions(orderColumn) >= minTestIndexVal)
    }

  // Performance Evaluators

  case class EvaluatorWrapper[Q](metric: Q, evaluator: Evaluator)

  protected val classificationEvaluators =
    ClassificationEvalMetric.values.filter(metric =>
      metric != ClassificationEvalMetric.areaUnderPR && metric != ClassificationEvalMetric.areaUnderROC
    ).toSeq.map { metric =>
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName(metric.toString)

      EvaluatorWrapper(metric, evaluator)
    }

  protected val regressionEvaluators = RegressionEvalMetric.values.toSeq.map { metric =>
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metric.toString)

    EvaluatorWrapper(metric, evaluator)
  }
}