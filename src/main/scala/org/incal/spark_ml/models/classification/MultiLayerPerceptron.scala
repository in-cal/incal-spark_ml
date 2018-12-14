package org.incal.spark_ml.models.classification

import java.util.Date
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class MultiLayerPerceptron(
  _id: Option[BSONObjectID] = None,
  hiddenLayers: Seq[Int],
  maxIteration: ValueOrSeq[Int] = Left(None),
  tolerance: ValueOrSeq[Double] = Left(None),
  blockSize: ValueOrSeq[Int] = Left(None),
  solver: Option[MLPSolver.Value] = None,
  seed: Option[Long] = None,
  stepSize: ValueOrSeq[Double] = Left(None),
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Classification

object MLPSolver extends Enumeration {
  val LBFGS = Value("l-bfgs")
  val GD = Value("gd")
}