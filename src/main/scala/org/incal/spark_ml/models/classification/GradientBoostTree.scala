package org.incal.spark_ml.models.classification

import java.util.Date

import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class GradientBoostTree(
  _id: Option[BSONObjectID] = None,
  core: TreeCore = TreeCore(),
  maxIteration: ValueOrSeq[Int] = Left(None),
  stepSize: ValueOrSeq[Double] = Left(None),
  subsamplingRate: ValueOrSeq[Double] = Left(None),
  lossType: Option[GBTClassificationLossType.Value] = None,
  //  impurity: Option[Impurity.Value] = None
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Classification

object GBTClassificationLossType extends Enumeration {
  val logistic = Value
}