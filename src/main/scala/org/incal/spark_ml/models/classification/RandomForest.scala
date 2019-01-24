package org.incal.spark_ml.models.classification

import java.util.Date

import org.incal.spark_ml.models.TreeCore
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class RandomForest(
  _id: Option[BSONObjectID] = None,
  core: TreeCore = TreeCore(),
  numTrees: ValueOrSeq[Int] = Left(None),
  subsamplingRate: ValueOrSeq[Double] = Left(None),
  impurity: Option[DecisionTreeImpurity.Value] = None,
  featureSubsetStrategy: Option[RandomForestFeatureSubsetStrategy.Value] = None,
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends ClassificationModel

object RandomForestFeatureSubsetStrategy extends Enumeration {
  val auto, all, onethird, sqrt, log2 = Value
}