package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

case class ClassificationRunSpec(
  ioSpec: IOSpec,
  mlModelId: BSONObjectID,
  learningSetting: ClassificationLearningSetting
)

case class TemporalClassificationRunSpec(
  ioSpec: TemporalGroupIOSpec,
  mlModelId: BSONObjectID,
  learningSetting: TemporalClassificationLearningSetting
)