package org.incal.spark_ml.models.regression

import java.util.Date
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class LinearRegression(
  _id: Option[BSONObjectID] = None,
  regularization: ValueOrSeq[Double] = Left(None),
  elasticMixingRatio: ValueOrSeq[Double] = Left(None),
  maxIteration: ValueOrSeq[Int] = Left(None),
  tolerance: ValueOrSeq[Double] = Left(None),
  fitIntercept: Option[Boolean] = None,
  solver: Option[RegressionSolver.Value] = None,
  standardization: Option[Boolean] = None,
  aggregationDepth: ValueOrSeq[Int] = Left(None),
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Regression

object RegressionSolver extends Enumeration {
  val Auto = Value("auto")
  val LBFGS = Value("l-bfgs")
  val Normal = Value("normal")
}