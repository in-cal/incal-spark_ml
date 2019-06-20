package org.incal.spark_ml.transformers

import java.{lang => jl}
import javax.inject.{Inject, Singleton}

import com.banda.math.domain.rand.RandomDistribution
import com.banda.network.business.learning.ReservoirRunnableFactory
import com.banda.network.domain.{ActivationFunctionType, ReservoirSetting}
import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.ml._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._
import org.incal.spark_ml.{ParamGrid, ParamSourceBinder}
import org.incal.spark_ml.SparkUtil.transformInPlaceWithParamGrids
import org.incal.spark_ml.models.ReservoirSpec

private class RCStatesWindow(override val uid: String, reservoirRunnableFactory: ReservoirRunnableFactory) extends Transformer with DefaultParamsWritable {

  def this(reservoirRunnableFactory: ReservoirRunnableFactory) = this(
    Identifiable.randomUID("rc_states_window"),
    reservoirRunnableFactory
  )

  protected[spark_ml] final val inputNodeNum: Param[Int] = new Param[Int](this, "inputNodeNum", "# reservoir nodes")
  protected[spark_ml] final val bias: Param[Double] = new Param[Double](this, "bias", "Bias")
  protected[spark_ml] final val nonBiasInitial: Param[Double] = new Param[Double](this, "nonBiasInitial", "Non-bias Initial")

  protected[spark_ml] final val reservoirNodeNum: Param[Int] = new Param[Int](this, "reservoirNodeNum", "# reservoir nodes")
  setDefault(reservoirNodeNum, 10)

  protected[spark_ml] final val reservoirInDegree = new Param[Int](this, "reservoirInDegree", "Reservoir in-degree")

  protected[spark_ml] final val reservoirInDegreeDistribution = new Param[RandomDistribution[Integer]](this, "reservoirInDegreeDistribution", "Reservoir in-degree distribution")

  protected[spark_ml] final val reservoirEdgesNum = new Param[Int](this, "reservoirEdgesNum", "# reservoir edges")

  protected[spark_ml] final val reservoirPreferentialAttachment = new Param[Boolean](this, "reservoirPreferentialAttachment", "Use preferential attachment to generate a reservoir?")
  setDefault(reservoirPreferentialAttachment, false)

  protected[spark_ml] final val reservoirBias = new Param[Boolean](this, "reservoirBias", "Use bias (weight) for a reservoir?")
  setDefault(reservoirBias, false)

  protected[spark_ml] final val reservoirCircularInEdges = new Param[Seq[Int]](this, "reservoirCircularInEdges", "# reservoir circular in-edges per node if spatial/toroidal topology is to be generated")

  protected[spark_ml] final val reservoirAllowSelfEdges = new Param[Boolean](this, "reservoirAllowSelfEdges", "Allow self-edges for a reservoir")
  setDefault(reservoirAllowSelfEdges, true)

  protected[spark_ml] final val reservoirAllowMultiEdges = new Param[Boolean](this, "reservoirAllowMultiEdges", "Allow multi edges for a reservoir")
  setDefault(reservoirAllowMultiEdges, false)

  protected[spark_ml] final val inputReservoirConnectivity = new Param[Double](this, "inputReservoirConnectivity", "Input reservoir connectivity")

  protected[spark_ml] final val weightDistribution = new Param[RandomDistribution[jl.Double]](this, "weightDistribution", "Weight distribution")

  protected[spark_ml] final val reservoirSpectralRadius = new Param[Double](this, "reservoirSpectralRadius", "Reservoir spectral radius")

  protected[spark_ml] final val reservoirFunctionType = new Param[ActivationFunctionType](this, "reservoirFunctionType", "Reservoir function type")

  protected[spark_ml] final val reservoirFunctionParams = new Param[Seq[Double]](this, "reservoirFunctionParams", "Reservoir function params")
  setDefault(reservoirFunctionParams, Nil)

  protected[spark_ml] final val perNodeReservoirFunctionWithParams = new Param[Stream[(ActivationFunctionType, Seq[jl.Double])]](this, "perNodeReservoirFunctionWithParams", "Per-node reservoir function with params")

  protected[spark_ml] final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  protected[spark_ml] final val orderCol: Param[String] = new Param[String](this, "orderCol", "order column name")
  protected[spark_ml] final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  def setReservoirNodeNum(value: Int): this.type = set(reservoirNodeNum, value)
  def setReservoirInDegree(value: Int): this.type = set(reservoirInDegree, value)
  def setReservoirInDegreeDistribution(value: RandomDistribution[Integer]): this.type = set(reservoirInDegreeDistribution, value)
  def setReservoirEdgesNum(value: Int): this.type = set(reservoirEdgesNum, value)
  def setReservoirPreferentialAttachment(value: Boolean): this.type = set(reservoirPreferentialAttachment, value)
  def setReservoirBias(value: Boolean): this.type = set(reservoirBias, value)
  def setReservoirCircularInEdges(value: Seq[Int]): this.type = set(reservoirCircularInEdges, value)
  def setReservoirAllowSelfEdges(value: Boolean): this.type = set(reservoirAllowSelfEdges, value)
  def setReservoirAllowMultiEdges(value: Boolean): this.type = set(reservoirAllowMultiEdges, value)
  def setInputReservoirConnectivity(value: Double): this.type = set(inputReservoirConnectivity, value)
  def setWeightDistribution(value: RandomDistribution[jl.Double]): this.type = set(weightDistribution, value)
  def setReservoirSpectralRadius(value: Double): this.type = set(reservoirSpectralRadius, value)
  def setReservoirFunctionType(value: ActivationFunctionType): this.type = set(reservoirFunctionType, value)
  def setReservoirFunctionParams(value: Seq[Double]): this.type = set(reservoirFunctionParams, value)
  def setPerNodeReservoirFunctionWithParams(value: Stream[(ActivationFunctionType, Seq[jl.Double])]): this.type = set(perNodeReservoirFunctionWithParams, value)

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOrderCol(value: String): this.type = set(orderCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  // create RC network runnable with input nodes and reservoir nodes
  protected lazy val (networkRunnable, inputNodes, reservoirNodes) = {
    reservoirRunnableFactory(ReservoirSetting(
      inputNodeNum = $(inputNodeNum),
      bias = $(bias),
      nonBiasInitial = $(nonBiasInitial),
      reservoirNodeNum = $(reservoirNodeNum),
      reservoirInDegree = get(reservoirInDegree),
      reservoirInDegreeDistribution = get(reservoirInDegreeDistribution),
      reservoirEdgesNum = get(reservoirEdgesNum),
      reservoirPreferentialAttachment = $(reservoirPreferentialAttachment),
      reservoirBias = $(reservoirBias),
      reservoirCircularInEdges = get(reservoirCircularInEdges),
      reservoirAllowSelfEdges = $(reservoirAllowSelfEdges),
      reservoirAllowMultiEdges = $(reservoirAllowMultiEdges),
      inputReservoirConnectivity = $(inputReservoirConnectivity),
      weightDistribution = $(weightDistribution),
      reservoirSpectralRadius = get(reservoirSpectralRadius),
      reservoirFunctionType = $(reservoirFunctionType),
      reservoirFunctionParams = $(reservoirFunctionParams),
      perNodeReservoirFunctionWithParams = get(perNodeReservoirFunctionWithParams)
    ))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
//    println("Network Time              : " + networkRunnable.currentTime)
//    println("Network Hash Code         : " + networkRunnable.hashCode())
//    val reservoirStates = reservoirNodes.map(outputNode => networkRunnable.getState(outputNode): Double).mkString(", ")

    // create a network state agg fun
    val rcAggFun = new NetworkStateVectorAgg(networkRunnable, inputNodes, reservoirNodes)

    // data frame with a sliding window with the RC network state agg function
    dataset.withColumn($(outputCol), rcAggFun(dataset($(inputCol))).over(Window.orderBy($(orderCol))))
  }

  override def copy(extra: ParamMap): RCStatesWindow = {
    val that = new RCStatesWindow(uid, reservoirRunnableFactory)
    copyValues(that, extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    val existingFields = schema.fields

    require(!existingFields.exists(_.name == $(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    require(existingFields.exists(_.name == $(orderCol)),
      s"Order column ${$(orderCol)} doesn't exist.")

    schema.add(StructField($(outputCol), SQLDataTypes.VectorType, true))
  }
}

@Singleton
class RCStatesWindowFactory @Inject() (reservoirRunnableFactory: ReservoirRunnableFactory) {

  def apply(
    inputCol: String,
    orderCol: String,
    outputCol: String)(
    spec: ReservoirSpec
  ): (PipelineStage, Traversable[ParamGrid[_]]) = {
    val (rcTransformer, paramGrids) = applyWoWashout(inputCol, orderCol, outputCol)(spec)

    // if washout period is defined add a drop-left transformer and create a mini pipeline
    if (!spec.washoutPeriod.isLeft || spec.washoutPeriod.left.get.nonEmpty) {
      val (dropLeftTransformer, dropLeftParamGrids) = DropSeriesLeft(orderCol)(spec.washoutPeriod)
      val rcWithWashout = new Pipeline().setStages(Array(rcTransformer, dropLeftTransformer))
      (rcWithWashout, paramGrids ++ dropLeftParamGrids)
    } else
      (rcTransformer, paramGrids)
  }

  def applyWoWashout(
    inputCol: String,
    orderCol: String,
    outputCol: String)(
    spec: ReservoirSpec
  ): (Transformer, Traversable[ParamGrid[_]]) =
    ParamSourceBinder(spec, new RCStatesWindow(reservoirRunnableFactory))
      .bindConstP(inputCol, _.inputCol)
      .bindConstP(orderCol, _.orderCol)
      .bindConstP(outputCol, _.outputCol)
      .bindDefP(_.inputNodeNum, _.inputNodeNum)
      .bindDefP(_.bias, _.bias)
      .bindDefP(_.nonBiasInitial, _.nonBiasInitial)
      .bindValOrSeqP(_.reservoirNodeNum, _.reservoirNodeNum)
      .bindValOrSeqP(_.reservoirInDegree, _.reservoirInDegree)
      .bindP(_.reservoirInDegreeDistribution, _.reservoirInDegreeDistribution)
      .bindValOrSeqP(_.reservoirEdgesNum, _.reservoirEdgesNum)
      .bindP(_.reservoirCircularInEdges, _.reservoirCircularInEdges)
      .bindDefP(_.reservoirPreferentialAttachment, _.reservoirPreferentialAttachment)
      .bindDefP(_.reservoirBias, _.reservoirBias)
      .bindDefP(_.reservoirAllowSelfEdges, _.reservoirAllowSelfEdges)
      .bindDefP(_.reservoirAllowMultiEdges, _.reservoirAllowMultiEdges)
      .bindValOrSeqP(_.inputReservoirConnectivity, _.inputReservoirConnectivity)
      .bindDefP(_.weightDistribution, _.weightDistribution)
      .bindValOrSeqP(_.reservoirSpectralRadius, _.reservoirSpectralRadius)
      .bindDefP(_.reservoirFunctionType, _.reservoirFunctionType)
      .bindDefP(_.reservoirFunctionParams, _.reservoirFunctionParams)
      .bindP(_.perNodeReservoirFunctionWithParams, _.perNodeReservoirFunctionWithParams)
      .build

  def applyInPlace(
    inputOutputCol: String,
    orderCol: String)(
    spec: ReservoirSpec
  ): (PipelineStage, Traversable[ParamGrid[_]]) =
    transformInPlaceWithParamGrids(
      apply(inputOutputCol, orderCol, _)(spec),
      inputOutputCol
    )
}