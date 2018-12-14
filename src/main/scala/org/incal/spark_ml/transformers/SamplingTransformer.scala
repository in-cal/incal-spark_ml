package org.incal.spark_ml.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.slf4j.LoggerFactory

import scala.util.Random

private class SamplingTransformer(override val uid: String) extends SchemaUnchangedTransformer {

  private val logger = LoggerFactory.getLogger("spark_ml")

  def this() = this(Identifiable.randomUID("sampling"))

  protected final val samplingRatios: Param[Seq[(String, Double)]] = new Param[Seq[(String, Double)]](this, "samplingRatios", "List of pairs - label and sampling ratio")
  protected final val seed: Param[Long] = new Param[Long](this, "seed", "Seed for random sampling")

  def setSamplingRatios(value: Seq[(String, Double)]): this.type = set(samplingRatios, value)
  def setSeed(value: Long): this.type = set(seed, value)
  setDefault(seed, Random.nextLong)

  protected def transformDataFrame(df: DataFrame) = {
    val aliasMap = extractAliasMap(df)

    val sampledDfsWithLabels = $(samplingRatios).map { case (label, samplingRatio) =>
      val labelOrAlias = aliasMap.getOrElse(label, label)

      val pdf = df.filter(df.col("labelString") === labelOrAlias)

      val newPdf = pdf.sample(false, samplingRatio, $(seed))
      logger.info("sampling " + labelOrAlias + " : " + pdf.count() + " -> " + newPdf.count())
      (newPdf, labelOrAlias)
    }

    val labels = sampledDfsWithLabels.map(_._2)
    val sampledDfs = sampledDfsWithLabels.map(_._1)

    val nonSampledDf = df.filter(!col("labelString").isin(labels: _*))
//    logger.info("rest non-sampled : " + nonSampledDf.count())

    val finalDf = sampledDfs.foldLeft(nonSampledDf)(_.union(_))

    logger.info("# after sampling : " + finalDf.count() + " with a seed: " + $(seed))

    finalDf
  }

  private def extractAliasMap(df: DataFrame): Map[String, String] = {
    val labelStringMetadata = df.schema.fields.find(_.name == "labelString").get.metadata

    if (labelStringMetadata.contains("ml_attr")) {
      val mlAttribute = labelStringMetadata.getMetadata("ml_attr")
      if (mlAttribute.contains("aliases")) {
        val aliases = mlAttribute.getMetadata("aliases")

        val from = aliases.getStringArray("from")
        val to = aliases.getStringArray("to")

        from.zip(to).toMap
      } else
        Map[String, String]()
    } else
      Map[String, String]()
  }

  override def copy(extra: ParamMap): SamplingTransformer = defaultCopy(extra)
}

object SamplingTransformer {

  def apply(
    samplingRatios: Seq[(String, Double)]
  ): Transformer = new SamplingTransformer().setSamplingRatios(samplingRatios)

  def apply(
    samplingRatios: Seq[(String, Double)],
    seed: Long
  ): Transformer = new SamplingTransformer().setSamplingRatios(samplingRatios).setSeed(seed)
}
