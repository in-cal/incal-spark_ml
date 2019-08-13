package org.incal.spark_ml

import com.bnd.network.business.NetworkModule
import com.google.inject.Guice
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Dataset, SparkSession}
import net.codingwell.scalaguice.InjectorExtensions._
import org.slf4j.LoggerFactory

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}

class SparkMLApp(execute: (SparkSession, SparkMLService) => Future[Unit]) extends App {

  protected val logger = LoggerFactory.getLogger("Spark-ML-App")

  protected def conf = new SparkConf(false)
    .setMaster("local[*]")
    .setAppName("Test-ML")
    .set("spark.logConf", "true")
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .set("spark.worker.cleanup.enabled", "true")
    .set("spark.worker.cleanup.interval", "900")

  private def createSession = SparkSession.builder().config(conf).getOrCreate()

  protected def mlServiceSetting = SparkMLServiceSetting(debugMode = false)

  private def createMLService = {
    val injector = Guice.createInjector(new NetworkModule())
    val factory = injector.instance[SparkMLServiceFactory]
    factory(mlServiceSetting)
  }

  // run the execute method with a newly created ML service
  private val future = execute(createSession, createMLService)

  future.onComplete {
    case Success(_) => logger.info("Spark app finished successfully.")
    case Failure(t) => logger.error("An error has occurred: " + t.getMessage)
  }

  Await.ready(future, 1.hour)
}