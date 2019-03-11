# InCal Spark ML Library [![version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://ada.parkinson.lu)

This is an extension of Spark ML library (version *2.2.0*) providing:

* Integrated service with a configurable classification and regression execution, cross-validation, and pre-processing.
* Several handy transformers and evaluators.
* Extension of classification and regression for the temporal domain mainly by two kernels (can be combined): a sliding window (delay line) and a reservoir computing network with various topologies and activiation functions.
* Convenient customizable pipeline execution.
* Summary evaluation metrics 

#### Installation

All you need is **Scala 2.11**. To pull the library you need to add the following dependency to *build.sbt*

```
"org.in-cal" %% "incal-spark_ml" % "0.1.0"
```

or to *pom.xml* (if you use maven)

```
<dependency>
    <groupId>org.in-cal</groupId>
    <artifactId>incal-spark_ml_2.11</artifactId>
    <version>0.1.0</version>
</dependency>
```

#### Examples

Once you have the *incal-spark_ml* lib on your classpath you are ready to go. To conveniently launch Spark-ML based (command line) apps the  *SparkMLApp* class with automatically created/injected resources: SparkSession and SparkMLService, can be used. You can explore and run the following examples demonstrating the basic functionality (all data is public):

* [Simple classification](src/main/scala/examples/SimpleClassification.scala) - for Iris data set
* [Classification with a custom Spark confing](src/main/scala/examples/ClassificationWithCustomSparkConf.scala) - for Iris data set
* [Classification with cross-validation](src/main/scala/examples/ClassificationWithCrossValidation.scala) - for Iris data set
* [Simple regression](src/main/scala/examples/SimpleRegression.scala) - for Abalone data set

as well as example classifications and regressions for temporal problems:

* [Temporal classification with sliding window (delay line)](src/main/scala/examples/TemporalClassificationWithSlidingWindow.scala) - for EEG eye movement time series
* [Temporal classification with a reservoir kernel](src/main/scala/examples/TemporalClassificationWithReservoirKernel.scala)  - for EEG eye movement time series
* [Temporal regression with a sliding window (delay line)](src/main/scala/examples/TemporalRegressionWithSlidingWindow.scala) - for S&P time series
* [Temporal regression with a reservoir kernel](src/main/scala/examples/TemporalRegressionWithReservoirKernel.scala) - for S&P time series

#### Acknowledgement

Development of this library has been significantly supported by a one-year MJFF Grant (2018-2019):
*Scalable Machine Learning And Reservoir Computing Platform for Analyzing Temporal Data Sets in the Context of Parkinson’s Disease and Biomedicine*

![alt text](https://in-cal.org/mjff_logo.png)