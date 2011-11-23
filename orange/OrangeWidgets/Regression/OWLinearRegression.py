"""
<name>Linear Regression</name>
<description>Linear Regression</name>
<icon>icons/LinearRegression.png</icon>
<priority>10</priority>
<category>Regression</category>
<keywords>linear, model, ridge, regression, lasso, least, absolute, shrinkage</keywords>

"""

import os, sys
from OWWidget import *

import Orange
from Orange.regression import linear, lasso
from orngWrap import PreprocessedLearner
from Orange.data import variable

class OWLinearRegression(OWWidget):
    settingsList = ["name", "use_ridge", "ridge_lambda", 
                    "use_lasso", "t", "tol"]
    
    def __init__(self, parent=None, signalManager=None, title="Linear Regression"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Training data", Orange.data.Table, self.set_data),
                       ("Preprocessor", PreprocessedLearner, self.set_preprocessor)]
        
        self.outputs = [("Learner", Orange.core.Learner), 
                        ("Predictor", Orange.core.Classifier),
                        ("Model Statistics", Orange.data.Table)]
        
        ##########
        # Settings
        ##########
         
        self.name = "Linear Regression"
        self.use_ridge = False
        self.ridge_lambda = 1.0
        self.use_lasso = False
        self.t = 1.0
        self.tol = 1e-3
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        OWGUI.lineEdit(self.controlArea, self, "name", box="Learner/predictor name",   
                       tooltip="Name of the learner/predictor")
        
        bbox = OWGUI.radioButtonsInBox(self.controlArea, self, "use_lasso", [], 
                                       box=None,
                                       callback=self.on_method_changed
                                       )
        rb = OWGUI.appendRadioButton(bbox, self, "use_lasso",
                                     label="Ordinary/Ridge Linear Regression",
                                     tooltip="",
                                     insertInto=bbox)

        self.lm_box = box = OWGUI.indentedBox(self.controlArea, 
                                sep=OWGUI.checkButtonOffsetHint(rb))
        self.lm_box.setEnabled(not self.use_lasso)
        
        OWGUI.checkWithSpin(box, self, "Ridge lambda", 1, 10,
                            "use_ridge", "ridge_lambda", step=1, 
                            tooltip="Ridge lambda for ridge regression")
        
        rb = OWGUI.appendRadioButton(bbox, self, "use_lasso",
                                     label="LASSO Regression",
                                     tooltip="",
                                     insertInto=bbox)
        
        self.lasso_box = box = OWGUI.indentedBox(self.controlArea, 
                                sep=OWGUI.checkButtonOffsetHint(rb))
        self.lasso_box.setEnabled(self.use_lasso)        
        
        OWGUI.doubleSpin(box, self, "t", 0.0, 1e4, 0.1,
                         label="Lasso bound",
                         tooltip="Tuning parameter, upper bound for the L1-norm of the regression coefficients",
                         )
        
        OWGUI.doubleSpin(box, self, "tol", 0.0, 1.0, 0.01,
                         label="Tolerance", 
                         tooltip="Tolerance parameter, regression coefficients (absoulute value) under tolerance are set to 0",
                         )
        
        OWGUI.rubber(self.controlArea)
        
        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     tooltip="Send the learner on",
                     autoDefault=True)
        
        self.data = None
        self.preprocessor = None
        self.resize(300, 100)
        self.apply()
        
    def set_data(self, data=None):
        self.data = data
            
    def set_preprocessor(self, pproc=None):
        self.preprocessor = pproc
        
    def handleNewSignals(self):
        self.apply()
            
    def on_method_changed(self):
        self.lm_box.setEnabled(not self.use_lasso)
        self.lasso_box.setEnabled(self.use_lasso)
    
    def apply(self):
        if self.use_lasso:
            self.apply_lasso()
        else:
            self.apply_ridge()
            
    def apply_ridge(self):
        if self.use_ridge:
            learner = linear.LinearRegressionLearner(name=self.name,
                                                ridgeLambda=self.ridge_lambda)
        else:
            learner = linear.LinearRegressionLearner(name=self.name)
        predictor = None
        if self.preprocessor:
            learner = self.preprocessor.wrapLearner(learner)
        
        self.error(0)
        if self.data is not None:
            try:
                predictor = learner(self.data)
                predictor.name = self.name
            except Exception, ex:
                self.error(0, "An error during learning: %r" % ex)
            
        self.send("Learner", learner)
        self.send("Predictor", predictor)
        self.send("Model Statistics", self.statistics_olr(predictor))

    def apply_lasso(self):
        learner = lasso.LassoRegressionLearner(t=self.t,
                            tol=self.tol, n_boot=0, n_perm=0,
                            name=self.name)
        predictor = None
        
        if self.preprocessor is not None:
            learner = self.preprocessor.wrapLearner(learner)

        self.error(0)
        try:
            if self.data is not None:
                ll = lasso.LassoRegressionLearner(t=self.t,
                            tol=self.tol, n_boot=10, n_perm=10)
                predictor = ll(self.data)
                predictor.name = self.name
        except Exception, ex:
            self.error(0, "An error during learning: %r" % ex)
            
        self.send("Learner", learner)
        self.send("Predictor", predictor)
        self.send("Model Statistics", self.statistics_lasso(predictor))
        
    def statistics_olr(self, m):
        if m is None:
            return None
        
        columns = [variable.String("Variable"),
                   variable.Continuous("Coeff Est"),
                   variable.Continuous("Std Error"),
                   variable.Continuous("t-value"),
                   variable.Continuous("p"),]
        
        domain = Orange.data.Domain(columns, None)
        vars = ["Intercept"] #if m.intercept else []
        vars.extend([a.name for a in m.domain.attributes])
        stats = []
        geti = lambda list, i: list[i] if list is not None else "?"
        
        for i, var in enumerate(vars):
            coef = m.coefficients[i]
            std_err = geti(m.std_error, i)
            t_val = geti(m.t_scores, i)
            p = geti(m.p_vals, i)
            stats.append([var, coef, std_err, t_val, p])
                
        print Orange.data.Table(domain, stats)[:]
        return Orange.data.Table(domain, stats)
    
    def statistics_lasso(self, m):
        if m is None:
            return None
        
        columns = [variable.String("Variable"),
                   variable.Continuous("Coeff Est"),
                   variable.Continuous("Std Error"),
                   variable.Continuous("p"),]
        
        domain = Orange.data.Domain(columns, None)
        vars = []
        vars.extend([a.name for a in m.domain.attributes])
        stats = [["Intercept", m.coef0, "?", "?"]]
        geti = lambda list, i: list[i] if list is not None else "?"
        
        for i, var in enumerate(vars):
            coef = m.coefficients[i]
            std_err = geti(m.std_errors_fixed_t, i)
            p = geti(m.p_vals, i)
            stats.append([var, coef, std_err, p])
        
        return Orange.data.Table(domain, stats)
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWLinearRegression()
    w.set_data(Orange.data.Table("housing"))
    w.show()
    app.exec_()
#    w.saveSettings()
