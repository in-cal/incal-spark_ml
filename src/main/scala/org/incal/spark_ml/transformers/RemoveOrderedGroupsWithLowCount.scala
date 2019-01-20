package org.incal.spark_ml.transformers

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

import scala.util.Random

private class FilterOrderedGroupsWithCount(override val uid: String) extends SchemaUnchangedTransformer {

  def this() = this(Identifiable.randomUID("filter_ordered_groups_with_count"))

  protected final val expectedCount: Param[Int] = new Param[Int](this, "expectedCount", "expected count")
  protected final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name")

  def setExpectedCount(value: Int): this.type = set(expectedCount, value)
  def setGroupCol(value: String) = set(groupCol, value)

  protected def transformDataFrame(df: DataFrame) = {
    val countOnWindow = count("*").over(Window.partitionBy(col($(groupCol))))

    val tempCountCol = "count_" + Random.nextInt()

    df.withColumn(tempCountCol, countOnWindow)
      .where(col(tempCountCol) === $(expectedCount))
      .drop(tempCountCol)
  }

  override def copy(extra: ParamMap): FilterOrderedGroupsWithCount = defaultCopy(extra)
}

object FilterOrderedGroupsWithCount {
  def apply(
    groupCol: String,
    expectedCount: Int
  ): Transformer = new FilterOrderedGroupsWithCount().setGroupCol(groupCol).setExpectedCount(expectedCount)
}

object FilterOrderedGroupsWithCountTest extends App {

  val conf = new SparkConf().setAppName("FilterOrderedGroupsWithCountTest").setMaster("local[*]")
  val session = SparkSession.builder().config(conf).getOrCreate()

  import session.sqlContext.implicits._

  val df = Seq(
    ("Peter", 0, 1000),
    ("John", 0, 1130),
    ("Zuzana", 2, 2121),
    ("John", 1, 1111),
    ("John", 2, 4323),
    ("Zuzana", 0, 8321),
//    ("Zuzana", 1, 1932),
    ("Cecilia", 0, 1220),
    ("Robert", 0, 9201),
//    ("Robert", 1, 8490),
    ("Robert", 2, 4100),
    ("Cecilia", 1, 2010),
    ("Cecilia", 2, 3012),
    ("Peter", 1, 2000),
    ("Peter", 2, 3000)
  ).toDF("name", "index", "salary")

  println("Original")
  df.show(numRows = 50)

  val newDf = FilterOrderedGroupsWithCount("name", 3).transform(df)

  println("New")
  newDf.show(numRows = 50)
}

