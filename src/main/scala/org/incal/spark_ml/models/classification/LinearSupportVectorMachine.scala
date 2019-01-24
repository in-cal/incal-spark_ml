package org.incal.spark_ml.models.classification

import java.util.Date
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class LinearSupportVectorMachine(
  _id: Option[BSONObjectID] = None,
  aggregationDepth: ValueOrSeq[Int] = Left(None),
  fitIntercept: Option[Boolean],
  maxIteration: ValueOrSeq[Int] = Left(None),
  regularization: ValueOrSeq[Double] = Left(None),
  standardization: Option[Boolean],
  threshold: ValueOrSeq[Double] = Left(None),
  tolerance: ValueOrSeq[Double] = Left(None),
  // TODO weightColumn: String
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends ClassificationModel