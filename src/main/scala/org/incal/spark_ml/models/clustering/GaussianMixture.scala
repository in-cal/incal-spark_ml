package org.incal.spark_ml.models.clustering

import java.util.Date

import reactivemongo.bson.BSONObjectID

case class GaussianMixture(
  _id: Option[BSONObjectID] = None,
  k: Int,
  maxIteration: Option[Int] = None,
  tolerance: Option[Double] = None,
  seed: Option[Long] = None,
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Clustering
