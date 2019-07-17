package org.incal.spark_ml.models.clustering

import java.util.Date

import reactivemongo.bson.BSONObjectID

case class BisectingKMeans(
  _id: Option[BSONObjectID],
  k: Int,
  maxIteration: Option[Int] = None,
  seed: Option[Long] = None,
  minDivisibleClusterSize: Option[Double] = None,
  name: Option[String] = None,
  createdById: Option[BSONObjectID] = None,
  timeCreated: Date = new Date()
) extends Clustering
