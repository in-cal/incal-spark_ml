import com.typesafe.sbt.license.{DepModuleInfo, LicenseInfo}

organization := "org.in-cal"

name := "incal-spark_ml"

version := "0.2.2.RC.1"

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
  "com.banda.network" % "banda-network-business-guice" % "0.5.6.1" exclude("com.googlecode.efficient-java-matrix-library", "ejml") exclude("jep", "jep") exclude("com.panayotis", "javaplot"), // ejml (version < 0.25) is LGPL, jep is GPLv3, and JavaPlot is LGPL 2.0
  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.in-cal" %% "incal-core" % "0.2.1",
  "org.reactivemongo" %% "reactivemongo-bson" % "0.18.1"    // BSON ids should be removed together with this lib
)

// For licenses not automatically downloaded (need to list them manually)
licenseOverrides := {
  case
    DepModuleInfo("org.apache.commons", _, _)
  | DepModuleInfo("org.apache.curator", _, _)
  | DepModuleInfo("org.apache.directory.api", _, _)
  | DepModuleInfo("org.apache.directory.server", _, _)
  | DepModuleInfo("org.apache.httpcomponents", _, _)
  | DepModuleInfo("org.apache.hadoop", _, _)
  | DepModuleInfo("org.apache.parquet", _, _)
  | DepModuleInfo("org.apache.avro", _, _)
  | DepModuleInfo("commons-beanutils", "commons-beanutils", _)
  | DepModuleInfo("commons-beanutils", "commons-beanutils-core", _)
  | DepModuleInfo("commons-cli", "commons-cli", _)
  | DepModuleInfo("commons-codec", "commons-codec", _)
  | DepModuleInfo("commons-collections", "commons-collections", _)
  | DepModuleInfo("commons-io", "commons-io", _)
  | DepModuleInfo("commons-lang", "commons-lang", _)
  | DepModuleInfo("commons-logging", "commons-logging", _)
  | DepModuleInfo("commons-net", "commons-net", _)
  | DepModuleInfo("com.google.guava", "guava", _)
  | DepModuleInfo("com.google.inject", "guice", _)
  | DepModuleInfo("com.google.inject.extensions", "guice-multibindings", _)
  | DepModuleInfo("io.dropwizard.metrics", _, _) =>
    LicenseInfo(LicenseCategory.Apache, "Apache License v2.0", "http://www.apache.org/licenses/LICENSE-2.0")

  case
    DepModuleInfo("org.glassfish.hk2", "hk2-api", "2.4.0-b34")
  | DepModuleInfo("org.glassfish.hk2", "hk2-locator", "2.4.0-b34")
  | DepModuleInfo("org.glassfish.hk2", "hk2-utils", "2.4.0-b34")
  | DepModuleInfo("org.glassfish.hk2", "osgi-resource-locator", "1.0.1")
  | DepModuleInfo("org.glassfish.hk2.external", "aopalliance-repackaged", "2.4.0-b34")
  | DepModuleInfo("org.glassfish.hk2.external", "javax.inject", "2.4.0-b34")
  | DepModuleInfo("org.glassfish.jersey.bundles.repackaged", "jersey-guava", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.containers", "jersey-container-servlet", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.containers", "jersey-container-servlet-core", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.core", "jersey-client", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.core", "jersey-common", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.core", "jersey-server", "2.22.2")
  | DepModuleInfo("org.glassfish.jersey.media", "jersey-media-jaxb", "2.22.2")
  =>
    LicenseInfo(LicenseCategory.GPLClasspath, "CDDL + GPLv2 with classpath exception", "https://javaee.github.io/glassfish/LICENSE")

  case DepModuleInfo("org.slf4j", _, _) =>
    LicenseInfo(LicenseCategory.MIT, "MIT", "http://opensource.org/licenses/MIT")
}

// POM settings for Sonatype
publishMavenStyle := true

scmInfo := Some(ScmInfo(url("https://github.com/peterbanda/incal-spark_ml"), "scm:git@github.com:peterbanda/incal-spark_ml.git"))

developers := List(Developer("bnd", "Peter Banda", "peter.banda@protonmail.com", url("https://peterbanda.net")))

licenses += "Apache 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
