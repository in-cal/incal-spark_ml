package org.incal.spark_ml.models.regression

import java.util.Date
import org.incal.spark_ml.models.TreeCore
import reactivemongo.bson.BSONObjectID

case class RegressionTree(
  _id: Option[BSONObjectID] = None,
  core: TreeCore = TreeCore(),
  impurity: Option[RegressionTreeImpurity.Value] = None,
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends RegressionModel

object RegressionTreeImpurity extends Enumeration {
  val variance = Value
}