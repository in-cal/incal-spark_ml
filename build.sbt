organization := "org.in-cal"

name := "incal-spark_ml"

version := "0.0.9"

description := "Extension of Spark ML library for the temporal domain with a lot of handy transformers, classification and regression models, and a convenient customizable pipeline execution."

isSnapshot := false

scalaVersion := "2.11.12"

resolvers ++= Seq(
  Resolver.mavenLocal,
  "bnd libs" at "https://peterbanda.net/maven2"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.spark" %% "spark-mllib" % "2.2.0",
  "com.banda.network" % "banda-network-business" % "0.5.6.1",
  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.in-cal" %% "incal-core" % "0.0.10",
  "org.reactivemongo" %% "reactivemongo-bson" % "0.12.6"    // BSON ids should be removed together with this lib
)

// POM settings for Sonatype
publishMavenStyle := true

developers := List(Developer("bnd", "Peter Banda", "peter.banda@protonmail.com", url("https://peterbanda.net")))

licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")

publishMavenStyle := true

// publishTo := sonatypePublishTo.value

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
