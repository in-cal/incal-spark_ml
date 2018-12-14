package org.incal.spark_ml.models

import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq

case class TreeCore(
  maxDepth: ValueOrSeq[Int] = Left(None),
  maxBins: ValueOrSeq[Int] = Left(None),
  minInstancesPerNode: ValueOrSeq[Int] = Left(None),
  minInfoGain: ValueOrSeq[Double] = Left(None),
  seed: Option[Long] = None
)