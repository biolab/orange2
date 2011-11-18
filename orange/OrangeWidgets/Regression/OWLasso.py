"""
<name>Lasso Regression</name>
<description>Least absolute shrinkage and selection operator</description>
<priority>120</priority>
<icon>icons/Lasso.png</icons>
<keywords>lasso, least, absolute, shrinkage, linear</keywords>
"""

from OWWidget import *

import Orange
from Orange.regression import lasso
from Orange.optimization import PreprocessedLearner

class Interval(object):
    def __init__(self, lower, upper, step=None):
        self.lower = lower
        self.upper = upper
        self.step = step or 1e-3
        
    def __iter__(self):
        yield self.lower
        yield self.upper
         
PARAMETERS = [{"name": "name",
               "type": str,
               "display_name": "Learner/Predictor name",
               "doc": "Name of the learner/Predictor",
               "default": "Lasso regression",
               },
              {"name": "t",
               "type": float,
               "display_name": "Lasso bound",
               "doc": "Tuning parameter, upper bound for the L1-norm of the regression coefficients",
               "range": Interval(0.0, None, step=0.1),
               "default": 1.0,
               },
              {"name": "tol",
               "type": float,
               "display_name": "Tolerance",
               "doc": "Tolerance parameter, regression coefficients (absoulute value) under tolerance are set to 0",
               "range": Interval(0.0, 1.0, step=0.01),
               "default": 0.001,
               }
              ]

def init_param_model(model, params):
    for p in params:
        setattr(model, p["name"], p["default"])
        
def get_model_params(model, params):
    d = {}
    for p in params:
        d[p["name"]] = getattr(model, p["name"])
    return d
    
class OWLasso(OWWidget):
    settingsList = ["name", "t", "tol"]
    PARAMETERS = PARAMETERS
    
    def __init__(self, parent=None, signalManager=None, title="Lasso regression"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Training data", Orange.data.Table, self.set_data),
                       ("Preprocessor", PreprocessedLearner, self.set_preprocessor)]
        
        self.outputs = [("Learner", lasso.LassoRegressionLearner),
                        ("Predictor", lasso.LassoRegression)]
        
        self.init_param_model(self.PARAMETERS)
        
        self.params_changed = False
        
        self.loadSettings()
        
        name = PARAMETERS[0]
        box = OWGUI.widgetBox(self.controlArea, name["display_name"], addSpace=True)
        OWGUI.lineEdit(box, self, name["name"], tooltip=name["doc"])
        
        box = OWGUI.widgetBox(self.controlArea, "Parameters", addSpace=True)
        
        for p in PARAMETERS[1:]:
            min, max = p["range"]
            OWGUI.doubleSpin(box, self, p["name"], min, max or 1e3, p["range"].step,
                             label=p["display_name"],
                             tooltip=p["doc"],
                             callback=self.on_param_change,)
            
        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     autoDefault=True)
        
        self.data = None
        self.preproc = None
        
        self.resize(300, 100)
        
        self.apply()
        
    init_param_model = init_param_model
    lasso_params = get_model_params
    
    def on_param_change(self):
        self.params_changed = True
        
    def set_data(self, data=None):
        if data is not None and self.isDataWithClass(data,
                Orange.core.VarTypes.Continuous, True):
            self.data = data
        else:
            self.data = None
        
    def set_preprocessor(self, preproc=None):
        self.preproc = preproc
        
    def handleNewSignals(self):
        self.apply()
        
    def apply(self):
        params = self.lasso_params(self.PARAMETERS)
        params["nBoot"] = 0
        params["nPerm"] = 0
        
        learner = lasso.LassoRegressionLearner(**params)
        predictor = None
        
        if self.preproc is not None:
            learner = self.preproc.wrapLearner(learner)
            
        if self.data is not None:
            predictor = learner(self.data)
            predictor.name = learner.name
            
        self.send("Learner", learner)
        self.send("Predictor", predictor)
        self.params_changed = False
        
        