package org.incal.spark_ml.models.setting

import reactivemongo.bson.BSONObjectID

case class ClassificationRunSpec(
  ioSpec: IOSpec,
  mlModelId: BSONObjectID,
  learningSetting: ClassificationLearningSetting
) extends RunSpec[IOSpec, ClassificationLearningSetting]

case class TemporalClassificationRunSpec(
  ioSpec: TemporalGroupIOSpec,
  mlModelId: BSONObjectID,
  learningSetting: TemporalClassificationLearningSetting
) extends RunSpec[TemporalGroupIOSpec, TemporalClassificationLearningSetting]