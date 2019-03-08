package org.incal.spark_ml

import javax.inject.{Inject, Singleton}
import com.google.inject.ImplementedBy
import org.incal.spark_ml.transformers.RCStatesWindowFactory

@ImplementedBy(classOf[SparkMLServiceFactoryImpl])
trait SparkMLServiceFactory {
  def apply(setting: SparkMLServiceSetting = SparkMLServiceSetting()): SparkMLService
}

@Singleton
private class SparkMLServiceFactoryImpl @Inject()(rcStatesWindowFactory: RCStatesWindowFactory) extends SparkMLServiceFactory {

  override def apply(setting: SparkMLServiceSetting) = new SparkMLServiceImpl(rcStatesWindowFactory, setting)
}

private class SparkMLServiceImpl(val rcStatesWindowFactory: RCStatesWindowFactory, val setting: SparkMLServiceSetting) extends SparkMLService