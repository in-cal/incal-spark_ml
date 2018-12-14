package org.incal.spark_ml.models

object ValueOrSeq {
  type ValueOrSeq[T] = Either[Option[T], Seq[T]]

  def toValue[T](valueOrSeq: ValueOrSeq[T]): Option[T] = valueOrSeq match {
    case Left(value) => value
    case Right(values) => None
  }
}
