"""
<name>PCA</name>
<description>Perform Principal Component Analysis</description>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>
<icons>icons/PCA.png</icons>
<tags>pca,principal,component,projection</tags>
<priority>3050</priority>

"""
import Orange
import Orange.utils.addons

from OWWidget import *
import OWGUI

import Orange
import Orange.projection.linear as plinear

import numpy as np
import sys

from plot.owplot import OWPlot
from plot.owcurve import OWCurve
from plot import owaxis


class ScreePlot(OWPlot):
    def __init__(self, parent=None, name="Scree Plot"):
        OWPlot.__init__(self, parent, name=name)
        self.cutoff_curve = CutoffCurve([0.0, 0.0], [0.0, 1.0],
                x_axis_key=owaxis.xBottom, y_axis_key=owaxis.yLeft)
        self.cutoff_curve.setVisible(False)
        self.cutoff_curve.set_style(OWCurve.Lines)
        self.add_custom_curve(self.cutoff_curve)

    def is_cutoff_enabled(self):
        return self.cutoff_curve and self.cutoff_curve.isVisible()

    def set_cutoff_curve_enabled(self, state):
        self.cutoff_curve.setVisible(state)

    def set_cutoff_value(self, value):
        xmin, xmax = self.x_scale()
        x = min(max(value, xmin), xmax)
        self.cutoff_curve.set_data([x, x], [0.0, 1.0])

    def mousePressEvent(self, event):
        if self.isLegendEvent(event, QGraphicsView.mousePressEvent):
            return

        if self.is_cutoff_enabled() and event.buttons() & Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x, _ = self.map_from_graph(pos)
            xmin, xmax = self.x_scale()
            if x >= xmin - 0.1 and x <= xmax + 0.1:
                x = min(max(x, xmin), xmax)
                self.cutoff_curve.set_data([x, x], [0.0, 1.0])
                self.emit_cutoff_moved(x)
        return QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.isLegendEvent(event, QGraphicsView.mouseMoveEvent):
            return

        if self.is_cutoff_enabled() and event.buttons() & Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x, _ = self.map_from_graph(pos)
            xmin, xmax = self.x_scale()
            if x >= xmin - 0.5 and x <= xmax + 0.5:
                x = min(max(x, xmin), xmax)
                self.cutoff_curve.set_data([x, x], [0.0, 1.0])
                self.emit_cutoff_moved(x)
        elif self.is_cutoff_enabled() and \
                self.is_pos_over_cutoff_line(event.pos()):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        return QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvene(self, event):
        return QGraphicsView.mouseReleaseEvent(self, event)

    def x_scale(self):
        ax = self.axes[owaxis.xBottom]
        if ax.labels:
            return 0, len(ax.labels) - 1
        elif ax.scale:
            return ax.scale[0], ax.scale[1]
        else:
            raise ValueError

    def emit_cutoff_moved(self, x):
        self.emit(SIGNAL("cutoff_moved(double)"), x)

    def set_axis_labels(self, *args):
        OWPlot.set_axis_labels(self, *args)
        self.map_transform = self.transform_for_axes()

    def is_pos_over_cutoff_line(self, pos):
        x1 = self.inv_transform(owaxis.xBottom, pos.x() - 1.5)
        x2 = self.inv_transform(owaxis.xBottom, pos.x() + 1.5)
        y = self.inv_transform(owaxis.yLeft, pos.y())
        if y < 0.0 or y > 1.0:
            return False
        curve_data = self.cutoff_curve.data()
        if not curve_data:
            return False
        cutoff = curve_data[0][0]
        return x1 < cutoff and cutoff < x2

class CutoffCurve(OWCurve):
    def __init__(self, *args, **kwargs):
        OWCurve.__init__(self, *args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeHorCursor)


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

        self.scree_plot = ScreePlot(self)
#        self.scree_plot.set_main_title("Scree Plot")
#        self.scree_plot.set_show_main_title(True)
        self.scree_plot.set_axis_title(owaxis.xBottom, "Principal Components")
        self.scree_plot.set_show_axis_title(owaxis.xBottom, 1)
        self.scree_plot.set_axis_title(owaxis.yLeft, "Proportion of Variance")
        self.scree_plot.set_show_axis_title(owaxis.yLeft, 1)

        self.variance_curve = self.scree_plot.add_curve(
                        "Variance",
                        Qt.red, Qt.red, 2,
                        xData=[],
                        yData=[],
                        style=OWCurve.Lines,
                        enableLegend=True,
                        lineWidth=2,
                        autoScale=1,
                        x_axis_key=owaxis.xBottom,
                        y_axis_key=owaxis.yLeft,
                        )

        self.cumulative_variance_curve = self.scree_plot.add_curve(
                        "Cumulative Variance",
                        Qt.darkYellow, Qt.darkYellow, 2,
                        xData=[],
                        yData=[],
                        style=OWCurve.Lines,
                        enableLegend=True,
                        lineWidth=2,
                        autoScale=1,
                        x_axis_key=owaxis.xBottom,
                        y_axis_key=owaxis.yLeft,
                        )

        self.mainArea.layout().addWidget(self.scree_plot)
        self.connect(self.scree_plot,
                     SIGNAL("cutoff_moved(double)"),
                     self.on_cutoff_moved
                     )

        self.connect(self.graphButton,
                     SIGNAL("clicked()"),
                     self.scree_plot.save_to_file)

        self.components = None
        self.variances = None
        self.variances_sum = None
        self.projector_full = None
        self.currently_selected = 0

        self.resize(800, 400)

    def clear(self):
        """Clear widget state
        """
        self.data = None
        self.scree_plot.set_cutoff_curve_enabled(False)
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
        """Apply PCA on input data, caching the full projection,
        then updating the selected components.
        
        """
        pca = self.construct_pca_all_comp()
        self.projector_full = projector = pca(self.data)

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
        variance_sum = self.projector_full.variance_sum

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
        self.scree_plot.set_axis_enabled(owaxis.xBottom, True)
        self.scree_plot.set_axis_enabled(owaxis.yLeft, True)
        self.scree_plot.set_axis_labels(owaxis.xBottom,
                                        ["PC" + str(i + 1) for i in x_space])

        self.variance_curve.set_data(x_space, self.variances)
        self.cumulative_variance_curve.set_data(x_space, self.variances_cumsum)
        self.variance_curve.setVisible(True)
        self.cumulative_variance_curve.setVisible(True)

        self.scree_plot.set_cutoff_curve_enabled(True)
        self.scree_plot.replot()

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
            cutoff = float(max_components - 1)
        else:
            cutoff = np.searchsorted(self.variances_cumsum,
                                     self.variance_covered / 100.0)
        self.scree_plot.set_cutoff_value(cutoff + 0.5)

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
            self.reportImage(self.scree_plot.save_to_file_direct)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPCA()
    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    app.exec_()
