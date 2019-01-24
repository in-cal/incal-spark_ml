package org.incal.spark_ml.models.regression

import java.util.Date
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq
import reactivemongo.bson.BSONObjectID

case class GeneralizedLinearRegression(
  _id: Option[BSONObjectID] = None,
  regularization: ValueOrSeq[Double] = Left(None),
  link: Option[GeneralizedLinearRegressionLinkType.Value] = None,
  maxIteration: ValueOrSeq[Int] = Left(None),
  tolerance: ValueOrSeq[Double] = Left(None),
  fitIntercept: Option[Boolean] = None,
  family: Option[GeneralizedLinearRegressionFamily.Value] = None,
  solver: Option[GeneralizedLinearRegressionSolver.Value] = None,
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends RegressionModel

object GeneralizedLinearRegressionFamily extends Enumeration {
  val Gaussian = Value("gaussian")
  val Binomial = Value("binomial")
  val Poisson = Value("poisson")
  val Gamma = Value("gamma")
}

object GeneralizedLinearRegressionLinkType extends Enumeration {
  val Identity = Value("identity")
  val Log = Value("log")
  val Logit = Value("logit")
  val Probit = Value("probit")
  val CLogLog = Value("cloglog")
  val Sqrt = Value("sqrt")
  val Inverse = Value("inverse")
}

object GeneralizedLinearRegressionSolver extends Enumeration {
  val IRLS = Value("irls")
}