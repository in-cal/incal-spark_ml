package org.incal.spark_ml.transformers

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class FixedOrderStringIndexer(override val uid: String) extends StringIndexer {

  def this() = this(Identifiable.randomUID("fixed_strIdx"))

  protected val labels: Param[Array[String]] = new Param[Array[String]](this, "labels", "Fixed-order labels to use")
  def setLabels(value: Array[String]): this.type = set(labels, value)

  override def fit(dataset: Dataset[_]): StringIndexerModel = {
    require($(labels).nonEmpty, s"No String labels provided for the fixed-order indexer.")

    transformSchema(dataset.schema, logging = true)
    copyValues(new StringIndexerModel(uid, $(labels)).setParent(this))
  }

  override def copy(extra: ParamMap): FixedOrderStringIndexer = defaultCopy(extra)
}