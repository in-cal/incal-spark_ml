package org.incal.spark_ml

import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.incal.spark_ml.models.ValueOrSeq.ValueOrSeq

import scala.collection.mutable.Buffer

case class ParamSourceBinder[S, T <: Params](source: S, model: T) {
  private var paramValueSetters: Buffer[ParamValueSetter[S, _]] = Buffer[ParamValueSetter[S, _]]()

  def bind[V](value: S => Option[V], paramName: String): this.type =
    bindAux(x => Left(value(x)), model.getParam(paramName))

  def bindP[V](value: S => Option[V], param: T => Param[V]): this.type =
    bindAux(x => Left(value(x)), param(model))

  def bindDef[V](value: S => V, paramName: String): this.type =
    bind(x => Some(value(x)), paramName)

  def bindDefP[V](value: S => V, param: T => Param[V]): this.type =
    bindP(x => Some(value(x)), param)

  def bindConst[V](value: V, paramName: String): this.type =
    bind(_ => Some(value), paramName)

  def bindConstP[V](value: V, param: T => Param[V]): this.type =
    bindP(_ => Some(value), param)

  def bindValOrSeq[V](value: S => ValueOrSeq[V], paramName: String): this.type =
    bindAux(value, model.getParam(paramName))

  def bindValOrSeqP[V](value: S => ValueOrSeq[V], param: T => Param[V]): this.type =
    bindAux(value, param(model))

  private def bindAux[V](values: S => Either[Option[V], Iterable[V]], param: Param[V]): this.type = {
    paramValueSetters.append(ParamValueSetter(param, values))
    this
  }

  def build: (T, Traversable[ParamGrid[_]]) = {
    val paramGrids = paramValueSetters.map(_.set(model, source): Option[ParamGrid[_]])
    (model, paramGrids.flatten)
  }

  def buildWithMaps: (T, Array[ParamMap]) = {
    val (model, paramGrids) = build
    val paramGridBuilder = new ParamGridBuilder()
    paramGrids.foreach{ case ParamGrid(param, values) => paramGridBuilder.addGrid(param, values)}
    (model, paramGridBuilder.build)
  }
}

case class ParamValueSetter[S, T](
  val param: Param[T],
  value: S => (Either[Option[T], Iterable[T]])
) {
  def set(
    params: Params,
    source: S
  ): Option[ParamGrid[T]] = value(source) match {
    case Left(valueOption) => valueOption.foreach(params.set(param, _)); None
    case Right(values) => Some(ParamGrid[T](param, values))
  }
}
