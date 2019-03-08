package org.incal.spark_ml

import com.banda.network.business.NetworkModule
import com.google.inject.Guice
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Dataset, SparkSession}
import net.codingwell.scalaguice.InjectorExtensions._

import scala.concurrent.{Future, Await}
import scala.concurrent.duration._

class SparkMLApp(execute: (SparkSession, SparkMLService) => Future[Unit]) extends App {

  protected def conf = new SparkConf(false)
    .setMaster("local[*]")
    .setAppName("Test-ML")
    .set("spark.logConf", "true")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .set("spark.worker.cleanup.enabled", "true")
    .set("spark.worker.cleanup.interval", "900")

  private def createSession = SparkSession.builder().config(conf).getOrCreate()

  private def createMLService = {
    val injector = Guice.createInjector(new NetworkModule())
    val factory = injector.instance[SparkMLServiceFactory]
    factory(SparkMLServiceSetting())
  }

  // run the execute method with a newly create ML service
  Await.ready(execute(createSession, createMLService), 1.hour)
}