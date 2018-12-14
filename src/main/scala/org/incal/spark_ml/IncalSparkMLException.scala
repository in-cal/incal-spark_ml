package org.incal.spark_ml

class IncalSparkMLException(message: String, cause: Throwable) extends RuntimeException(message, cause) {
  def this(message: String) = this(message, null)
}
