package org.incal.spark_ml

import org.apache.spark.ml.param.Param

case class ParamGrid[T](param: Param[T], values: Iterable[T])
