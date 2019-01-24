package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

case class IOSpec(
  inputFieldNames: Seq[String],
  outputFieldName: String,
  filterId: Option[BSONObjectID] = None,
  replicationFilterId: Option[BSONObjectID] = None
) extends AbstractIOSpec {
  override val allFieldNames = (inputFieldNames ++ Seq(outputFieldName)).toSet.toSeq
}

case class TemporalGroupIOSpec(
  inputFieldNames: Seq[String],
  outputFieldName: String,
  groupIdFieldName: String,
  orderFieldName: String,
  orderedStringValues: Seq[String] = Nil,
  filterId: Option[BSONObjectID] = None,
  replicationFilterId: Option[BSONObjectID] = None
) extends AbstractIOSpec {
  override val allFieldNames = (inputFieldNames ++ Seq(outputFieldName, groupIdFieldName, orderFieldName)).toSet.toSeq
}

trait AbstractIOSpec {
  def allFieldNames: Seq[String]
}