package org.incal.spark_ml.models.classification

import java.util.Date
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class LogisticRegression(
  _id: Option[BSONObjectID] = None,
  regularization: ValueOrSeq[Double] = Left(None),
  elasticMixingRatio: ValueOrSeq[Double] = Left(None),
  maxIteration: ValueOrSeq[Int] = Left(None),
  tolerance: ValueOrSeq[Double] = Left(None),
  fitIntercept: Option[Boolean] = None,
  family: Option[LogisticModelFamily.Value] = None,
  standardization: Option[Boolean] = None,
  aggregationDepth: ValueOrSeq[Int] = Left(None),
  threshold: ValueOrSeq[Double] = Left(None),
  thresholds: Option[Seq[Double]] = None,  // used for multinomial logistic regression
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Classification

object LogisticModelFamily extends Enumeration {
  val Auto = Value("auto")
  val Binomial = Value("binomial")
  val Multinomial = Value("multinomial")
}