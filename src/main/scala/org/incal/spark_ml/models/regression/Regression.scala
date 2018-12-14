package org.incal.spark_ml.models.regression

import java.util.Date
import reactivemongo.bson.BSONObjectID

abstract class Regression {
  val _id: Option[BSONObjectID]
  val name: Option[String]
  val createdById: Option[BSONObjectID]
  val timeCreated: Date
}
