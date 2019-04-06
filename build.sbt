organization := "org.in-cal"

name := "incal-spark_ml"

version := "0.1.3"

description := "Spark ML library extension primarily for the temporal domain with a lot of handy transformers, classification and regression models, and a convenient customizable pipeline execution."

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
  "com.banda.network" % "banda-network-business-guice" % "0.5.6.1",
  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.in-cal" %% "incal-core" % "0.1.4",
  "org.reactivemongo" %% "reactivemongo-bson" % "0.12.6"    // BSON ids should be removed together with this lib
)

// POM settings for Sonatype
homepage := Some(url("https://ada.parkinson.lu"))

publishMavenStyle := true

scmInfo := Some(ScmInfo(url("https://github.com/peterbanda/incal-spark_ml"), "scm:git@github.com:peterbanda/incal-spark_ml.git"))

developers := List(Developer("bnd", "Peter Banda", "peter.banda@protonmail.com", url("https://peterbanda.net")))

licenses += "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
