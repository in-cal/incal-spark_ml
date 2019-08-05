package examples

import tech.tablesaw.plotly.components.{Axis, Figure, Layout, Page}
import org.incal.core.util.writeStringAsStream
import tech.tablesaw.plotly.traces.ScatterTrace
import java.{lang => jl}

object Lala extends App {

  val x: Array[Double] = Array(1, 2, 3, 4, 5, 6)
  val y: Array[Double] = Array(0, 1, 6, 14, 25, 39)

  val x2: Array[Double] = Array(2, 4, 5, 6, 10)
  val y2: Array[Double] = Array(1, 2, 3, 4, -1)

  PlotlyPlotter.plotSeries(
    Seq(x.zip(y), x2.zip(y2)), //
    PlotSetting(
      title = None, // Some("Lala"),
      xLabel = None, // Some("x"),
      yLabel = Some("y"),
      xMin = Some(0),
      xMax = Some(20),
      yMin = None,
      yMax = None,
      false,
      Seq("First", "Second")
    ),
    "lala.html"
  )
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

object PlotlyPlotter {

  def plotSeries(
    data: Seq[Seq[(Double, Double)]],
    setting: PlotSetting = PlotSetting(),
    outputFile: String = "output.html"
  ) = {
    val traces = data.zip(setting.captions).map { case (xySeries, caption) =>
      val x = xySeries.map(_._1)
      val y = xySeries.map(_._2)

      ScatterTrace.builder(x.toArray, y.toArray)
        .showLegend(setting.showLegend)
        .name(caption)
        .mode(ScatterTrace.Mode.LINE)
        .build()
    }

    val layout = Layout.builder(setting.title.getOrElse(""))
      .width(700).height(450)
      .xAxis {
        val axis = Axis.builder.title(setting.xLabel.getOrElse(""))

        if (setting.xMin.isDefined || setting.xMax.isDefined) {
          axis.range(setting.xMin.map(new jl.Double(_)).getOrElse(null), setting.xMax.map(new jl.Double(_)).getOrElse(null))
        }

        axis.autoRange(Axis.AutoRange.FALSE)

        axis.build()
      }
      .yAxis {
        val axis = Axis.builder.title(setting.yLabel.getOrElse(""))

        if (setting.yMin.isDefined || setting.yMax.isDefined) {
          axis.range(setting.yMin.map(new jl.Double(_)).getOrElse(null), setting.yMax.map(new jl.Double(_)).getOrElse(null))
        }

        axis.autoRange(Axis.AutoRange.FALSE)

        axis.build
      }
      .showLegend(setting.showLegend)
      .build

    val page = Page.pageBuilder(new Figure(layout, traces:_*), "target").build
    val output = page.asJavascript

    writeStringAsStream(output, new java.io.File(outputFile))
  }
}