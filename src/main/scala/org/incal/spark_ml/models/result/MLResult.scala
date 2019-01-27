package org.incal.spark_ml.models.result

import org.incal.spark_ml.models.setting.RunSpec
import reactivemongo.bson.BSONObjectID
import java.{util => ju}

trait MLResult {
  type R <: RunSpec

  val _id: Option[BSONObjectID]
  val runSpec: R
  val timeCreated: ju.Date

  def mlModelId = runSpec.mlModelId
  def ioSpec = runSpec.ioSpec
  def inputFieldNames = ioSpec.inputFieldNames
  def outputFieldName = ioSpec.outputFieldName
  def filterId = ioSpec.filterId
}