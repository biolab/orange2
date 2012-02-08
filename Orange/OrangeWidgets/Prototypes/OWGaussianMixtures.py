"""\
<name>Gaussian Mixture</name>
<description>Gaussian Mixture Modeling</description>

"""

from OWWidget import *
import OWGUI

import Orange
from Orange.clustering import mixture

GM_PARAMS = [{"name": "n",
              "type": int,
              "default": 3,
              "range": range(1,11),
              "doc": "Number of gaussians in the mixtrue."
              },
             {"name": "init_function",
              "type": ":func:",
#              "type": ":class:Orange.clustering.mixture:Initializer",
              "default": ":func:Orange.clustering.mixture:init_kmeans"
#              "default": ":class:Orange.clustering.mixture:init_kmeans"
              }
             ]

class OWGaussianMixtures(OWWidget):
    settingsList = ["init_method", "n"]
    
    def __init__(self, parent=None, signalManager=None, title="Gaussin Mixture"):
        OWWidget.__init__(self, parent, signalManager, title)
        
        self.inputs = [("Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Data with Indicator Matrix", Orange.data.Table)]
        
        self.init_method = 0
        self.n = 3
        self.auto_commit = True
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        OWGUI.spin(self.controlArea, self, "n", min=1, max=10, step=1,
                   box="Settings",
                   label="Number of gaussians", 
                   tooltip="The number of gaussians in the mixture ",
                   callback=self.on_params_changed)
        
        OWGUI.comboBox(self.controlArea, self, "init_method",
                       box="Initialization",
                       items=["K-means", "Random"],
                       tooltip="Method used to initialize the mixture", callback=self.on_params_changed)
        
        OWGUI.button(self.controlArea, self, "Apply", callback=self.commit)
    
    def set_data(self, data=None):
        self.input_data = data
        self.gmm = None
        if self.input_data:
            self.run_opt()
            if self.auto_commit:
                self.commit()
            
    def run_opt(self):
        from Orange.clustering import mixture
        init_function = mixture.init_kmeans if self.init_method == 1 else mixture.init_random
        
        gmm = mixture.GaussianMixture(self.input_data,
                                      n=self.n,
                                      init_function=init_function)
        
        data = self.input_data.translate(gmm.domain)
        
        input_matrix, _, _ = data.to_numpy_MA()
        self.gmm = gmm
        
        self.indicator_matrix = mixture.prob_est(input_matrix, gmm.weights,
                        gmm.means, gmm.covariances)
        
        vars = []
        for i, w in enumerate(self.gmm.weights):
            var = Orange.feature.Continuous("Cluster {0}".format(i))
            var.attributes["weight"] = str(w)
            vars.append(var)
        input_domain = self.input_data.domain
        domain = Orange.data.Domain(input_domain.attributes + vars, input_domain.class_var)
        domain.add_metas(input_domain.get_metas())
        data = self.input_data.translate(domain)
        for ex, indicator_row in zip(data, self.indicator_matrix):
            for var, value in zip(vars, indicator_row):
                ex[var] = float(value)
        
        self.output_data = data
        
        if self.auto_commit:
            self.commit()
        
    def on_params_changed(self):
        pass
#        if self.auto_commit:
#            self.run_opt()
    
    def commit(self):
        if self.gmm and self.input_data:
            self.send("Data with Indicator Matrix", self.output_data)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWGaussianMixtures()
    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    app.exec_()
