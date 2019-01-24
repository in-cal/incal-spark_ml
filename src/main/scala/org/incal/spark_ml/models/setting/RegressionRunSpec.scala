package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

case class RegressionRunSpec(
  ioSpec: IOSpec,
  mlModelId: BSONObjectID,
  learningSetting: RegressionLearningSetting
)

case class TemporalRegressionRunSpec(
  ioSpec: TemporalGroupIOSpec,
  mlModelId: BSONObjectID,
  learningSetting: TemporalRegressionLearningSetting
)
