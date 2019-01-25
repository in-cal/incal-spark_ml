package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

case class RegressionRunSpec(
  ioSpec: IOSpec,
  mlModelId: BSONObjectID,
  learningSetting: RegressionLearningSetting
) extends RunSpec {
  type IO = IOSpec
  type S = RegressionLearningSetting
}

case class TemporalRegressionRunSpec(
  ioSpec: TemporalGroupIOSpec,
  mlModelId: BSONObjectID,
  learningSetting: TemporalRegressionLearningSetting
) extends RunSpec {
  type IO = TemporalGroupIOSpec
  type S = TemporalRegressionLearningSetting
}
