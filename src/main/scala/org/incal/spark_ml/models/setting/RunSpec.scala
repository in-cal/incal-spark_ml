package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

trait RunSpec {

  type IO <: AbstractIOSpec
  type S

  val ioSpec: IO
  val learningSetting: S
  val mlModelId: BSONObjectID
}
