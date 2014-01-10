"""
<name>PCA</name>
<description>Perform Principal Component Analysis</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
<icon>icons/PCA.svg</icon>
<tags>pca,principal,component,projection</tags>
<priority>3050</priority>

"""
import sys

import numpy as np

from PyQt4.Qwt5 import QwtPlot, QwtPlotCurve, QwtSymbol
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange
import Orange.projection.linear as plinear

from OWWidget import *
from OWGraph import OWGraph

import OWGUI


def plot_curve(title=None, pen=None, brush=None, style=QwtPlotCurve.Lines,
               symbol=QwtSymbol.Ellipse, legend=True, antialias=True,
               auto_scale=True, xaxis=QwtPlot.xBottom, yaxis=QwtPlot.yLeft):
    curve = QwtPlotCurve(title or "")
    return configure_curve(curve, pen=pen, brush=brush, style=style,
                           symbol=symbol, legend=legend, antialias=antialias,
                           auto_scale=auto_scale, xaxis=xaxis, yaxis=yaxis)


def configure_curve(curve, title=None, pen=None, brush=None,
          style=QwtPlotCurve.Lines, symbol=QwtSymbol.Ellipse,
          legend=True, antialias=True, auto_scale=True,
          xaxis=QwtPlot.xBottom, yaxis=QwtPlot.yLeft):
    if title is not None:
        curve.setTitle(title)
    if pen is not None:
        curve.setPen(pen)

    if brush is not None:
        curve.setBrush(brush)

    if not isinstance(symbol, QwtSymbol):
        symbol_ = QwtSymbol()
        symbol_.setStyle(symbol)
        symbol = symbol_

    curve.setStyle(style)
    curve.setSymbol(QwtSymbol(symbol))
    curve.setRenderHint(QwtPlotCurve.RenderAntialiased, antialias)
    curve.setItemAttribute(QwtPlotCurve.Legend, legend)
    curve.setItemAttribute(QwtPlotCurve.AutoScale, auto_scale)
    curve.setAxis(xaxis, yaxis)
    return curve


class PlotTool(QObject):
    """
    A base class for Plot tools that operate on QwtPlot's canvas
    widget by installing itself as its event filter.

    """
    cursor = Qt.ArrowCursor

    def __init__(self, parent=None, graph=None):
        QObject.__init__(self, parent)
        self.__graph = None
        self.__oldCursor = None
        self.setGraph(graph)

    def setGraph(self, graph):
        """
        Install this tool to operate on ``graph``.
        """
        if self.__graph is graph:
            return

        if self.__graph is not None:
            self.uninstall(self.__graph)

        self.__graph = graph

        if graph is not None:
            self.install(graph)

    def graph(self):
        return self.__graph

    def install(self, graph):
        canvas = graph.canvas()
        canvas.setMouseTracking(True)
        canvas.installEventFilter(self)
        canvas.destroyed.connect(self.__on_destroyed)
        self.__oldCursor = canvas.cursor()
        canvas.setCursor(self.cursor)

    def uninstall(self, graph):
        canvas = graph.canvas()
        canvas.removeEventFilter(self)
        canvas.setCursor(self.__oldCursor)
        canvas.destroyed.disconnect(self.__on_destroyed)
        self.__oldCursor = None

    def eventFilter(self, obj, event):
        if obj is self.__graph.canvas():
            return self.canvasEvent(event)
        return False

    def canvasEvent(self, event):
        """
        Main handler for a canvas events.
        """
        if event.type() == QEvent.MouseButtonPress:
            return self.mousePressEvent(event)
        elif event.type() == QEvent.MouseButtonRelease:
            return self.mouseReleaseEvent(event)
        elif event.type() == QEvent.MouseButtonDblClick:
            return self.mouseDoubleClickEvent(event)
        elif event.type() == QEvent.MouseMove:
            return self.mouseMoveEvent(event)
        elif event.type() == QEvent.Leave:
            return self.leaveEvent(event)
        elif event.type() == QEvent.Enter:
            return self.enterEvent(event)
        return False

    # These are actually event filters (note the return values)
    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def enterEvent(self, event):
        return False

    def leaveEvent(self, event):
        return False

    def keyPressEvent(self, event):
        return False

    def transform(self, point, xaxis=QwtPlot.xBottom, yaxis=QwtPlot.yLeft):
        """
        Transform a QPointF from plot coordinates to canvas local coordinates.
        """
        x = self.__graph.transform(xaxis, point.x())
        y = self.__graph.transform(yaxis, point.y())
        return QPoint(x, y)

    def invTransform(self, point, xaxis=QwtPlot.xBottom, yaxis=QwtPlot.yLeft):
        """
        Transform a QPoint from canvas local coordinates to plot coordinates.
        """
        x = self.__graph.invTransform(xaxis, point.x())
        y = self.__graph.invTransform(yaxis, point.y())
        return QPointF(x, y)

    @Slot()
    def __on_destroyed(self, obj):
        obj.removeEventFilter(self)


class CutoffControler(PlotTool):

    class CutoffCurve(QwtPlotCurve):
        pass

    cutoffChanged = Signal(float)
    cutoffMoved = Signal(float)
    cutoffPressed = Signal()
    cutoffReleased = Signal()

    NoState, Drag = 0, 1

    def __init__(self, parent=None, graph=None):
        self.__curve = None
        self.__range = (0, 1)
        self.__cutoff = 0
        super(CutoffControler, self).__init__(parent, graph)
        self._state = self.NoState

    def install(self, graph):
        super(CutoffControler, self).install(graph)
        assert self.__curve is None
        self.__curve = CutoffControler.CutoffCurve("")
        configure_curve(self.__curve, symbol=QwtSymbol.NoSymbol, legend=False)
        self.__curve.setData([self.__cutoff, self.__cutoff], [0.0, 1.0])
        self.__curve.attach(graph)

    def uninstall(self, graph):
        super(CutoffControler, self).uninstall(graph)
        self.__curve.detach()
        self.__curve = None

    def _toRange(self, value):
        minval, maxval = self.__range
        return max(min(value, maxval), minval)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            cut = self.invTransform(event.pos()).x()
            self.setCutoff(cut)
            self.cutoffPressed.emit()
            self._state = self.Drag
        return True

    def mouseMoveEvent(self, event):
        if self._state == self.Drag:
            cut = self._toRange(self.invTransform(event.pos()).x())
            self.setCutoff(cut)
            self.cutoffMoved.emit(cut)
        else:
            cx = self.transform(QPointF(self.cutoff(), 0)).x()
            if abs(cx - event.pos().x()) < 2:
                self.graph().canvas().setCursor(Qt.SizeHorCursor)
            else:
                self.graph().canvas().setCursor(self.cursor)
        return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._state == self.Drag:
            cut = self._toRange(self.invTransform(event.pos()).x())
            self.setCutoff(cut)
            self.cutoffReleased.emit()
            self._state = self.NoState
        return True

    def setCutoff(self, cutoff):
        minval, maxval = self.__range
        cutoff = max(min(cutoff, maxval), minval)
        if self.__cutoff != cutoff:
            self.__cutoff = cutoff
            if self.__curve is not None:
                self.__curve.setData([cutoff, cutoff], [0.0, 1.0])
            self.cutoffChanged.emit(cutoff)
            if self.graph() is not None:
                self.graph().replot()

    def cutoff(self):
        return self.__cutoff

    def setRange(self, minval, maxval):
        maxval = max(minval, maxval)
        if self.__range != (minval, maxval):
            self.__range = (minval, maxval)
            self.setCutoff(max(min(self.cutoff(), maxval), minval))


class Graph(OWGraph):
    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)
        self.gridCurve.attach(self)

    # bypass the OWGraph event handlers
    def mousePressEvent(self, event):
        QwtPlot.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        QwtPlot.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        QwtPlot.mouseReleaseEvent(self, event)


class OWPCA(OWWidget):
    settingsList = ["standardize", "max_components", "variance_covered",
                    "use_generalized_eigenvectors", "auto_commit"]

    def __init__(self, parent=None, signalManager=None, title="PCA"):
        OWWidget.__init__(self, parent, signalManager, title, wantGraph=True)

        self.inputs = [("Input Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Transformed Data", Orange.data.Table, Default),
                        ("Eigen Vectors", Orange.data.Table)]

        self.standardize = True
        self.max_components = 0
        self.variance_covered = 100.0
        self.use_generalized_eigenvectors = False
        self.auto_commit = False

        self.loadSettings()

        self.data = None
        self.changed_flag = False

        #####
        # GUI
        #####
        grid = QGridLayout()
        box = OWGUI.widgetBox(self.controlArea, "Components Selection",
                              orientation=grid)

        label1 = QLabel("Max components", box)
        grid.addWidget(label1, 1, 0)

        sb1 = OWGUI.spin(box, self, "max_components", 0, 1000,
                         tooltip="Maximum number of components",
                         callback=self.on_update,
                         addToLayout=False,
                         keyboardTracking=False
                         )
        self.max_components_spin = sb1.control
        self.max_components_spin.setSpecialValueText("All")
        grid.addWidget(sb1.control, 1, 1)

        label2 = QLabel("Variance covered", box)
        grid.addWidget(label2, 2, 0)

        sb2 = OWGUI.doubleSpin(box, self, "variance_covered", 1.0, 100.0, 1.0,
                               tooltip="Percent of variance covered.",
                               callback=self.on_update,
                               decimals=1,
                               addToLayout=False,
                               keyboardTracking=False
                               )
        sb2.control.setSuffix("%")
        grid.addWidget(sb2.control, 2, 1)

        OWGUI.rubber(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Commit")
        cb = OWGUI.checkBox(box, self, "auto_commit", "Commit on any change")
        b = OWGUI.button(box, self, "Commit",
                         callback=self.update_components)
        OWGUI.setStopper(self, b, cb, "changed_flag", self.update_components)

        self.plot = Graph()
        canvas = self.plot.canvas()
        canvas.setFrameStyle(QFrame.StyledPanel)
        self.mainArea.layout().addWidget(self.plot)
        self.plot.setAxisTitle(QwtPlot.yLeft, "Proportion of Variance")
        self.plot.setAxisTitle(QwtPlot.xBottom, "Principal Components")
        self.plot.setAxisScale(QwtPlot.yLeft, 0.0, 1.0)
        self.plot.enableGridXB(True)
        self.plot.enableGridYL(True)
        self.plot.setGridColor(Qt.lightGray)

        self.variance_curve = plot_curve(
            "Variance",
            pen=QPen(Qt.red, 2),
            symbol=QwtSymbol.NoSymbol,
            xaxis=QwtPlot.xBottom,
            yaxis=QwtPlot.yLeft
        )
        self.cumulative_variance_curve = plot_curve(
            "Cumulative Variance",
            pen=QPen(Qt.darkYellow, 2),
            symbol=QwtSymbol.NoSymbol,
            xaxis=QwtPlot.xBottom,
            yaxis=QwtPlot.yLeft
        )

        self.variance_curve.attach(self.plot)
        self.cumulative_variance_curve.attach(self.plot)

        self.selection_tool = CutoffControler(parent=self.plot.canvas())
        self.selection_tool.cutoffMoved.connect(self.on_cutoff_moved)

        self.graphButton.clicked.connect(self.saveToFile)
        self.components = None
        self.variances = None
        self.variances_sum = None
        self.projector_full = None
        self.currently_selected = 0

        self.resize(800, 400)

    def clear(self):
        """
        Clear (reset) the widget state.
        """
        self.data = None
        self.selection_tool.setGraph(None)
        self.clear_cached()
        self.variance_curve.setVisible(False)
        self.cumulative_variance_curve.setVisible(False)

    def clear_cached(self):
        """Clear cached components
        """
        self.components = None
        self.variances = None
        self.variances_cumsum = None
        self.projector_full = None
        self.currently_selected = 0

    def set_data(self, data=None):
        """Set the widget input data.
        """
        self.clear()
        if data is not None:
            self.data = data
            self.on_change()
        else:
            self.send("Transformed Data", None)
            self.send("Eigen Vectors", None)

    def on_change(self):
        """Data has changed and we need to recompute the projection.
        """
        if self.data is None:
            return
        self.clear_cached()
        self.apply()

    def on_update(self):
        """Component selection was changed by the user.
        """
        if self.data is None:
            return
        self.update_cutoff_curve()
        if self.currently_selected != self.number_of_selected_components():
            self.update_components_if()

    def construct_pca_all_comp(self):
        pca = plinear.PCA(standardize=self.standardize,
                          max_components=0,
                          variance_covered=1,
                          use_generalized_eigenvectors=self.use_generalized_eigenvectors
                          )
        return pca

    def construct_pca(self):
        max_components = self.max_components
        variance_covered = self.variance_covered
        pca = plinear.PCA(standardize=self.standardize,
                          max_components=max_components,
                          variance_covered=variance_covered / 100.0,
                          use_generalized_eigenvectors=self.use_generalized_eigenvectors
                          )
        return pca

    def apply(self):
        """
        Apply PCA on input data, caching the full projection and
        updating the selected components.

        """
        pca = self.construct_pca_all_comp()
        self.projector_full = pca(self.data)

        self.variances = self.projector_full.variances
        self.variances /= np.sum(self.variances)
        self.variances_cumsum = np.cumsum(self.variances)

        self.max_components_spin.setRange(0, len(self.variances))
        self.max_components = min(self.max_components,
                                  len(self.variances) - 1)
        self.update_scree_plot()
        self.update_cutoff_curve()
        self.update_components_if()

    def update_components_if(self):
        if self.auto_commit:
            self.update_components()
        else:
            self.changed_flag = True

    def update_components(self):
        """Update the output components.
        """
        if self.data is None:
            return

        scale = self.projector_full.scale
        center = self.projector_full.center
        components = self.projector_full.projection
        input_domain = self.projector_full.input_domain
        variances = self.projector_full.variances

        # Get selected components (based on max_components and
        # variance_coverd)
        pca = self.construct_pca()
        variances, components, variance_sum = pca._select_components(variances, components)

        projector = plinear.PcaProjector(input_domain=input_domain,
                                         standardize=self.standardize,
                                         scale=scale,
                                         center=center,
                                         projection=components,
                                         variances=variances,
                                         variance_sum=variance_sum)
        projected_data = projector(self.data)

        append_metas(projected_data, self.data)

        eigenvectors = self.eigenvectors_as_table(components)

        self.currently_selected = self.number_of_selected_components()

        self.send("Transformed Data", projected_data)
        self.send("Eigen Vectors", eigenvectors)

        self.changed_flag = False

    def eigenvectors_as_table(self, U):
        features = [Orange.feature.Continuous("C%i" % i) \
                    for i in range(1, U.shape[1] + 1)]
        domain = Orange.data.Domain(features, False)
        return Orange.data.Table(domain, [list(v) for v in U])

    def update_scree_plot(self):
        x_space = np.arange(0, len(self.variances))
        self.plot.enableAxis(QwtPlot.xBottom, True)
        self.plot.enableAxis(QwtPlot.yLeft, True)
        if len(x_space) <= 5:
            self.plot.setXlabels(["PC" + str(i + 1) for i in x_space])
        else:
            # Restore continuous plot scale
            # TODO: disable minor ticks
            self.plot.setXlabels(None)

        self.variance_curve.setData(x_space, self.variances)
        self.cumulative_variance_curve.setData(x_space, self.variances_cumsum)
        self.variance_curve.setVisible(True)
        self.cumulative_variance_curve.setVisible(True)

        self.selection_tool.setRange(0, len(self.variances) - 1)
        self.selection_tool.setGraph(self.plot)
        self.plot.replot()

    def on_cutoff_moved(self, value):
        """Cutoff curve was moved by the user.
        """
        components = int(np.floor(value)) + 1
        # Did the number of components actually change
        self.max_components = components
        self.variance_covered = self.variances_cumsum[components - 1] * 100
        if self.currently_selected != self.number_of_selected_components():
            self.update_components_if()

    def update_cutoff_curve(self):
        """Update cutoff curve from 'Components Selection' control box.
        """
        if self.max_components == 0:
            # Special "All" value
            max_components = len(self.variances_cumsum)
        else:
            max_components = self.max_components

        variance = self.variances_cumsum[max_components - 1] * 100.0
        if variance < self.variance_covered:
            cutoff = max_components - 1
        else:
            cutoff = np.searchsorted(self.variances_cumsum,
                                     self.variance_covered / 100.0)

        self.selection_tool.setCutoff(float(cutoff + 0.5))

    def number_of_selected_components(self):
        """How many components are selected.
        """
        if self.data is None:
            return 0

        variance_components = np.searchsorted(self.variances_cumsum,
                                              self.variance_covered / 100.0)
        if self.max_components == 0:
            # Special "All" value
            max_components = len(self.variances_cumsum)
        else:
            max_components = self.max_components
        return min(variance_components + 1, max_components)

    def sendReport(self):
        self.reportSettings("PCA Settings",
                            [("Max. components", self.max_components),
                             ("Variance covered", "%i%%" % self.variance_covered),
                             ])
        if self.data is not None and self.projector_full:
            output_domain = self.projector_full.output_domain
            st_dev = np.sqrt(self.projector_full.variances)
            summary = [[""] + [a.name for a in output_domain.attributes],
                       ["Std. deviation"] + ["%.3f" % sd for sd in st_dev],
                       ["Proportion Var"] + ["%.3f" % v for v in self.variances * 100.0],
                       ["Cumulative Var"] + ["%.3f" % v for v in self.variances_cumsum * 100.0]
                       ]

            th = "<th>%s</th>".__mod__
            header = "".join(map(th, summary[0]))
            td = "<td>%s</td>".__mod__
            summary = ["".join(map(td, row)) for row in summary[1:]]
            tr = "<tr>%s</tr>".__mod__
            summary = "\n".join(map(tr, [header] + summary))
            summary = "<table>\n%s\n</table>" % summary

            self.reportSection("Summary")
            self.reportRaw(summary)

            self.reportSection("Scree Plot")
            self.reportImage(self.plot.saveToFileDirect)

    def saveToFile(self):
        self.plot.saveToFile()


def append_metas(dest, source):
    """
    Append all meta attributes from the `source` table to `dest` table.
    The tables must be of the same length.

    :param dest:
        An data table into which the meta values will be copied.
    :type dest: :class:`Orange.data.Table`

    :param source:
        A data table with the meta attributes/values to be copied into `dest`.
    :type source: :class:`Orange.data.Table`

    """
    if len(dest) != len(source):
        raise ValueError("'dest' and 'source' must have the same length.")

    dest.domain.add_metas(source.domain.get_metas())
    for dest_inst, source_inst in zip(dest, source):
        for meta_id, val in source_inst.get_metas().items():
            dest_inst[meta_id] = val


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPCA()
    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    w.set_data(Orange.data.Table("brown-selected"))
    app.exec_()
