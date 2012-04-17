"""
<name>PCA</name>

"""
import Orange
import Orange.utils.addons

from OWWidget import *
import OWGUI

import Orange
import Orange.projection.linear as plinear

import numpy as np

import sys
from Orange import orangeqt

sys.modules["orangeqt"] = orangeqt

from plot.owplot import OWPlot
from plot.owcurve import OWCurve
from plot import owaxis

class ScreePlot(OWPlot):
    def __init__(self, parent=None, name="Scree Plot"):
        OWPlot.__init__(self, parent, name=name)

class OWPCA(OWWidget):
    settingsList = ["standardize", "max_components", "variance_covered",
                    "use_generalized_eigenvectors"]
    def __init__(self, parent=None, signalManager=None, title="PCA"):
        OWWidget.__init__(self, parent, signalManager, title)

        self.inputs = [("Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Projected Data", Orange.data.Table, Default),
                        ("PCA Projector", Orange.projection.linear.PcaProjector),
                        ("Principal Vectors", Orange.data.Table)
                        ]

        self.standardize = True
        self.use_variance_covered = 0
        self.max_components = 0
        self.variance_covered = 100.0
        self.use_generalized_eigenvectors = False

        self.loadSettings()

        self.data = None

        #####
        # GUI
        #####
        grid = QGridLayout()
        box = OWGUI.widgetBox(self.controlArea, "Settings",
                              orientation=grid)
        cb = OWGUI.checkBox(box, self, "standardize", "Standardize",
                            tooltip="Standardize all input features.",
                            callback=self.on_change, 
                            addToLayout=False
                            )
        grid.addWidget(cb, 0, 0)

#        OWGUI.radioButtonsInBox(box, self, "use_variance_covered", [],
#                                callback=self.on_update)
#        rb1 = OWGUI.appendRadioButton(box, self, "use_variance_covered",
#                                      "Max components",
#                                      tooltip="Select max components",
#                                      callback=self.on_update,
#                                      addToLayout=False
#                                      )
#        grid.addWidget(rb1, 1, 0)
        label1 = QLabel("Max components", box)
        grid.addWidget(label1, 1, 0)

        sb1 = OWGUI.spin(box, self, "max_components", 0, 1000,
                         tooltip="Maximum number of components",
                         callback=self.on_change,
                         addToLayout=False,
                         keyboardTracking=False
                         )
        sb1.control.setSpecialValueText("All")
        grid.addWidget(sb1.control, 1, 1)

#        rb2 = OWGUI.appendRadioButton(box, self, "use_variance_covered",
#                                      "Variance covered", 
#                                      tooltip="Percent of variance covered.",
#                                      callback=self.on_update,
#                                      addToLayout=False
#                                      )
#        grid.addWidget(rb2, 2, 0)
        label2 = QLabel("Variance covered", box)
        grid.addWidget(label2, 2, 0)

        sb2 = OWGUI.doubleSpin(box, self, "variance_covered", 1.0, 100.0, 5.0,
                               tooltip="Percent of variance covered.",
                               callback=self.on_change,
                               decimals=1,
                               addToLayout=False,
                               keyboardTracking=False
                               )
        sb2.control.setSuffix("%")
        grid.addWidget(sb2.control, 2, 1)

        cb = OWGUI.checkBox(box, self, "use_generalized_eigenvectors",
                            "Use generalized eigenvectors",
                            callback=self.on_change,
                            addToLayout=False,
                            )
        grid.addWidget(cb, 3, 0, 1, 2)

        OWGUI.rubber(self.controlArea)

        self.scree_plot = ScreePlot(self)
#        self.scree_plot.set_main_title("Scree Plot")
#        self.scree_plot.set_show_main_title(True)
        self.scree_plot.set_axis_title(owaxis.xBottom, "Principal Components")
        self.scree_plot.set_show_axis_title(owaxis.xBottom, 1)
        self.scree_plot.set_axis_title(owaxis.yLeft, "Proportion of Variance")
        self.scree_plot.set_show_axis_title(owaxis.yLeft, 1)
        
        self.mainArea.layout().addWidget(self.scree_plot)

        self.components = None
        self.variances = None
        self.variances_sum = None
        self.projector_full = None

        self.resize(800, 600)

    def clear(self):
        """Clear widget state
        """
        self.data = None
        self.clear_cached()
        
    def clear_cached(self):
        """Clear cached components
        """
        self.components = None
        self.variances = None
        self.variances_sum = None
        self.projector_full = None

    def set_data(self, data=None):
        """Set the widget input data.
        """
        self.clear()
        if data is not None:
            self.data = data
            self.on_change()

    def on_change(self):
        if self.data is None:
            return
        self.clear_cached()
        self.apply()

    def on_update(self):
        if self.data is None:
            return
        self.update_components()

    def construct_pca_all_comp(self):
        pca = plinear.PCA(standardize=self.standardize,
                          max_components=0,
                          variance_covered=1,
                          use_generalized_eigenvectors=self.use_generalized_eigenvectors
                          )
        return pca

    def construct_pca(self):
        max_components = self.max_components #if not self.use_variance_covered else 0
        variance_covered = self.variance_covered #if self.use_variance_covered else 0
        pca = plinear.PCA(standardize=self.standardize,
                          max_components=max_components,
                          variance_covered=variance_covered / 100.0,
                          use_generalized_eigenvectors=self.use_generalized_eigenvectors
                          )
        return pca

    def apply(self):
        """Apply PCA in input data, caching the full projection,
        then updating the selected components.
        
        """
        pca = self.construct_pca_all_comp()
        self.projector_full = projector = pca(self.data)
        self.update_scree_plot()
        self.update_components()

    def update_components(self):
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

        self.send("Projected Data", projected_data)
        self.send("PCA Projector", projector)
        self.send("Principal Vectors", eigenvectors)

    def eigenvectors_as_table(self, U):
        features = [Orange.feature.Continuous("C%i" % i) \
                    for i in range(1, U.shape[1] + 1)]
        domain = Orange.data.Domain(features, False)
        return Orange.data.Table(domain, [list(v) for v in U])

    def update_scree_plot(self):
        variances = self.projector_full.variances
        s = np.sum(variances)
        cv = variances / s
        cs = np.cumsum(cv)
        x_space = np.arange(0, len(variances))
        self.scree_plot.set_axis_enabled(owaxis.xBottom, True)
        self.scree_plot.set_axis_enabled(owaxis.yLeft, True)
        self.scree_plot.set_axis_labels(owaxis.xBottom, 
                                        ["PC" + str(i + 1) for i in x_space])

        self.c = self.scree_plot.add_curve("Variance",
                        Qt.red, Qt.red, 2, 
                        xData=x_space,
                        yData=cv,
                        style=OWCurve.Lines,
                        enableLegend=True,
                        lineWidth=2,
                        autoScale=1,
                        x_axis_key=owaxis.xBottom,
                        y_axis_key=owaxis.yLeft,
                        )
        
        self.c = self.scree_plot.add_curve("Cumulative Variance",
                        Qt.darkYellow, Qt.darkYellow, 2, 
                        xData=x_space,
                        yData=cs,
                        style=OWCurve.Lines,
                        enableLegend=True,
                        lineWidth=2,
                        autoScale=1,
                        x_axis_key=owaxis.xBottom,
                        y_axis_key=owaxis.yLeft,
                        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPCA()
    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    app.exec_()
