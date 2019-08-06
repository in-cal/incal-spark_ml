package org.incal.core

import org.incal.core.util.writeStringAsStream
import tech.tablesaw.plotly.components.{Axis, Figure, Layout, Page}
import tech.tablesaw.plotly.traces.ScatterTrace
import java.{lang => jl}

object PlotlyPlotter {

  def plotScatter(
    data: Traversable[Traversable[(Double, Double)]],
    setting: PlotSetting = PlotSetting(),
    outputFileName: String = "output.html"
  ): Unit =
    plotScatterAux(data, ScatterTrace.Mode.MARKERS, setting, outputFileName)

  def plotLines(
    data: Traversable[Traversable[Double]],
    xValues: Traversable[Double] = Nil,
    setting: PlotSetting = PlotSetting(),
    outputFileName: String = "output.html"
  ): Unit = {
    val xValuesInit = xValues match {
      case Nil => Stream.from(1).map(_.toDouble)
      case _ => xValues.toSeq
    }
    plotXYLines(
      data.map(series => series.toSeq.zip(xValuesInit).map(_.swap)),
      setting,
      outputFileName
    )
  }

  def plotXYLines(
    data: Traversable[Traversable[(Double, Double)]],
    setting: PlotSetting = PlotSetting(),
    outputFileName: String = "output.html"
  ): Unit =
    plotScatterAux(data, ScatterTrace.Mode.LINE, setting, outputFileName)

  private def plotScatterAux(
    data: Traversable[Traversable[(Double, Double)]],
    mode: ScatterTrace.Mode,
    setting: PlotSetting = PlotSetting(),
    outputFileName: String = "output.html"
  ) = {
    val missingCaptionsCount = data.size - setting.captions.size
    val captionsInit = setting.captions ++ Seq.fill(Math.max(missingCaptionsCount, 0))("")

    val traces = data.toSeq.zip(captionsInit).map { case (xySeries, caption) =>
      val x = xySeries.map(_._1)
      val y = xySeries.map(_._2)

      ScatterTrace.builder(x.toArray, y.toArray)
        .showLegend(setting.showLegend)
        .name(caption)
        .mode(mode)
        .build()
    }

    def buildAxis(
      label: Option[String],
      min: Option[Double],
      max: Option[Double]
    ) = {
      val axis = Axis.builder.title(label.getOrElse(""))

      if (min.isDefined || max.isDefined) {
        axis.range(min.map(new jl.Double(_)).getOrElse(null), max.map(new jl.Double(_)).getOrElse(null))
        axis.autoRange(Axis.AutoRange.FALSE)
      }

      axis.build()
    }

    val layout = Layout.builder(setting.title.getOrElse(""))
      .width(700).height(450)
      .xAxis(buildAxis(setting.xLabel, setting.xMin, setting.xMax))
      .yAxis(buildAxis(setting.yLabel, setting.yMin, setting.yMax))
      .showLegend(setting.showLegend)
      .build

    val page = Page.pageBuilder(new Figure(layout, traces:_*), "target").build
    val output = page.asJavascript

    writeStringAsStream(output, new java.io.File(outputFileName))
  }
}

case class PlotSetting(
  title: Option[String] = None,
  xLabel: Option[String] = None,
  yLabel: Option[String] = None,
  xMin: Option[Double] = None,
  xMax: Option[Double] = None,
  yMin: Option[Double] = None,
  yMax: Option[Double] = None,
  showLegend: Boolean = true,
  captions: Seq[String] = Nil
)

object PlotlyPlotterTest extends App {

  val x: Seq[Double] = Seq(1, 2, 3, 4, 5, 6)
  val y: Seq[Double] = Seq(0, 1, 6, 14, 25, 39)

  val x2: Seq[Double] = Seq(2, 4, 5, 6, 10)
  val y2: Seq[Double] = Seq(1, 2, 3, 4, -1)

  val setting = PlotSetting(
    title = Some("Test"),
    xLabel = Some("x"),
    yLabel = Some("y"),
    xMin = Some(0),
    xMax = Some(20),
    yMin = None,
    yMax = None,
    false,
    Seq("First", "Second")
  )

  PlotlyPlotter.plotXYLines(
    Seq(x.zip(y), x2.zip(y2)),
    setting,
    "test-xylines.html"
  )

  PlotlyPlotter.plotLines(
    Seq(x, y),
    Nil,
    setting,
    "test-lines.html"
  )

  PlotlyPlotter.plotScatter(
    Seq(x.zip(y), x2.zip(y2)),
    setting,
    "test-scatter.html"
  )
}