import com.typesafe.sbt.license.{DepModuleInfo, LicenseInfo}

name := "incal-spark_ml"

version := "0.2.2.RC.4"

description := "Spark ML library extension primarily for the temporal domain with a lot of handy transformers, classification and regression models, and a convenient customizable pipeline execution."

isSnapshot := false

scalaVersion := "2.11.12"

resolvers ++= Seq(
  Resolver.mavenLocal
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.spark" %% "spark-mllib" % "2.2.0",
  "com.bnd-lib" % "bnd-network-guice" % "0.7.0",
  "tech.tablesaw" % "tablesaw-jsplot" % "0.34.1",
  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.in-cal" %% "incal-core" % "0.2.2.RC.3",
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
  | DepModuleInfo("io.dropwizard.metrics", _, _)
  | DepModuleInfo("org.apache.xbean", "xbean-asm5-shaded", "4.4")
  | DepModuleInfo("org.apache.ivy", "ivy", "2.4.0")
  | DepModuleInfo("org.apache.zookeeper", "zookeeper", "3.4.6")
  | DepModuleInfo("com.fasterxml.jackson.module", "jackson-module-paranamer", "2.6.5")
  | DepModuleInfo("io.netty", "netty-all", "4.0.43.Final")
  | DepModuleInfo("com.bnd-lib", "bnd-core", "0.7.0")
  | DepModuleInfo("com.bnd-lib", "bnd-function", "0.7.0")
  | DepModuleInfo("com.bnd-lib", "bnd-math", "0.7.0")
  | DepModuleInfo("com.bnd-lib", "bnd-network", "0.7.0")
  | DepModuleInfo("com.bnd-lib", "bnd-network-guice", "0.7.0")
  | DepModuleInfo("org.codehaus.jettison", "jettison", "1.1")
  | DepModuleInfo("org.htrace", "htrace-core", "3.0.4")
  | DepModuleInfo("org.mortbay.jetty", "jetty-util", "6.1.26")
  | DepModuleInfo("org.objenesis", "objenesis", "2.1")
  | DepModuleInfo("oro", "oro", "2.0.8")
  | DepModuleInfo("xerces", "xercesImpl", "2.9.1")
  =>
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

  case
    DepModuleInfo("javax.mail", "mail", "1.4.7")
  =>
    LicenseInfo(LicenseCategory.GPLClasspath, "CDDL + GPLv2 with classpath exception", "https://javaee.github.io/javamail/LICENSE")

  case
    DepModuleInfo("com.esotericsoftware", "kryo-shaded", "3.0.3")
  =>
    LicenseInfo(LicenseCategory.BSD, "BSD 2-clause", "https://opensource.org/licenses/BSD-2-Clause")

  case
    DepModuleInfo("com.github.fommil.netlib", "core", "1.1.2")
  | DepModuleInfo("org.antlr", "antlr4-runtime", "4.5.3")
  | DepModuleInfo("org.fusesource.leveldbjni", "leveldbjni-all", "1.8")
  | DepModuleInfo("org.hamcrest", "hamcrest-core", "1.3")
  =>
    LicenseInfo(LicenseCategory.BSD, "BSD 3-clause", "https://opensource.org/licenses/BSD-3-Clause")

  case
    DepModuleInfo("com.thoughtworks.paranamer", "paranamer", "2.6")
  =>
    LicenseInfo(LicenseCategory.BSD, "BSD License", "http://www.opensource.org/licenses/bsd-license.php")

  case
    DepModuleInfo("org.codehaus.janino", "commons-compiler", "3.0.0")
  | DepModuleInfo("org.codehaus.janino", "janino", "3.0.0")
  =>
    LicenseInfo(LicenseCategory.BSD, "New BSD License", "http://www.opensource.org/licenses/bsd-license.php")

  case DepModuleInfo("org.slf4j", _, _) =>
    LicenseInfo(LicenseCategory.MIT, "MIT", "http://opensource.org/licenses/MIT")
}
organization := "org.in-cal"

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
