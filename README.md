# InCal Spark ML Library [![version](https://img.shields.io/badge/version-1.0.1-yellow.svg)](https://semver.org)

This is an extension of Spark ML library (version 2.2.0) providing:

* Integrated service with configurable classification and regression execution, cross-validation, and pre-processing
* Several handy transformers and evaluators
* Extension of classification and regression for temporal domain especially by served by a sliding window (delay line) and a reservoir computing kernel
* Convenient customizable pipeline execution.
* Summary evaluation metrics 


#### Installation

All you need is **Scala 2.11**. To pull the library you need to add the following dependency to build.sbt

```
"org.in-cal" %% "incal-spark_ml" % "0.1.0"
```

or pom.xml (if you use maven)

```
<dependency>
    <groupId>org.in-cal</groupId>
    <artifactId>incal-spark_ml_2.11</artifactId>
    <version>0.1.0</version>
</dependency>
```

#### Examples

Once you have incal-spark_ml on your classpath you are ready to go.
To conveniently launch Spark-ML based (command line) apps a class *SparkMLApp* with automatically created resources: SparkSession and SparkMLService. You can explore and run the following examples demonstrating the basic functionality:

* [Simple classification](src/main/scala/examples/SimpleClassification.scala)
* [ClassificationWithCustomSparkConf](src/main/scala/examples/ClassificationWithCustomSparkConf.scala)
* [ClassificationWithCrossValidation](src/main/scala/examples/ClassificationWithCrossValidation.scala)
* [SimpleRegression](src/main/scala/examples/SimpleRegression.scala)
* [TemporalClassificationWithSlidingWindow](src/main/scala/examples/TemporalClassificationWithSlidingWindow.scala)
* [TemporalClassificationWithReservoirKernel](src/main/scala/examples/TemporalClassificationWithReservoirKernel.scala)
* [TemporalRegressionWithSlidingWindow](src/main/scala/examples/TemporalRegressionWithSlidingWindow.scala)
* [TemporalRegressionWithReservoirKernel](src/main/scala/examples/TemporalRegressionWithReservoirKernel.scala) 

#### Acknowledgement

Development of this library has been significantly supported by a one-year MJFF Grant (2018-2019):
*Scalable Machine Learning And Reservoir Computing Platform for Analyzing Temporal Data Sets in the Context of Parkinson’s Disease and Biomedicine*

![alt text](https://in-cal.org/mjff_logo.png)