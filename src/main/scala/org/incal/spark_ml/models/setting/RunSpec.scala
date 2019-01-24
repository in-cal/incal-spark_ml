package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

trait RunSpec[IO <: AbstractIOSpec, S] {
  val ioSpec: IO
  val learningSetting: S
  val mlModelId: BSONObjectID
}
