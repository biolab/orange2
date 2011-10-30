"""
<name>Reliability</name>
"""

import Orange
from Orange.evaluation import reliability
from Orange.evaluation import testing

from functools import partial
 
from OWWidget import *
import OWGUI

class OWReliability(OWWidget):
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None, title="Reliability"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Learner", Orange.core.Learner, self.set_learner),
                       ("Train Data", Orange.data.Table, self.set_train_data),
                       ("Test Data", Orange.data.Table, self.set_test_data)]
        
        self.ouputs = [("Reliability Scores", Orange.data.Table)]
        
        self.variance_checked = False
        self.bias_checked = False
        self.bagged_variance = False
        self.local_cv = False
        self.local_model_pred_error = False
        self.bagging_variance_cn = False
        self.mahalanobis_distance = True
        
        self.var_e = "0.01, 0.1, 0.5, 1.0, 2.0"
        self.bias_e =  "0.01, 0.1, 0.5, 1.0, 2.0"
        self.bagged_m = 50
        self.local_cv_k = 1
        self.local_pe_k = 5
        self.bagged_cn_m = 5
        self.bagged_cn_k = 1
        self.mahalanobis_k = 5
        
        self.include_error = True
        self.include_class = True
        self.include_input_features = True
        self.auto_commit = False
        
        
        self.methods = [("variance_checked", self.run_SAVar),
                        ("bias_checked", self.run_SABias),
                        ("bagged_variance", self.run_BAGV),
                        ("local_cv", self.run_LCV),
                        ("local_model_pred_error", self.run_CNK),
                        ("bagging_variance_cn", self.run_BVCK),
                        ("mahalanobis_distance", self.run_Mahalanobis)]
        
        #####
        # GUI
        #####
        self.loadSettings()
        
        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace=True)
        self.info_box = OWGUI.widgetLabel(box, "\n\n")
        
        rbox = OWGUI.widgetBox(self.controlArea, "Methods", addSpace=True)
        def method_box(parent, name, value):
            box = OWGUI.widgetBox(rbox, name, flat=False)
            box.setCheckable(True)
            box.setChecked(bool(getattr(self, value)))
            self.connect(box, SIGNAL("toggled(bool)"),
                         lambda on: (setattr(self, value, on),
                                     self.method_selection_changed(value)))
            return box
            
        variance_box = method_box(rbox, "Sensitivity analysis (variance)",
                                  "variance_checked")
        OWGUI.lineEdit(variance_box, self, "var_e", "Sensitivities:", 
                       tooltip="List of possible e values (comma separated) for SAvar reliability estimates.", 
                       callback=partial(self.method_param_changed, 0))
        
        bias_box = method_box(rbox, "Sensitivity analysis (bias)",
                                    "bias_checked")
        OWGUI.lineEdit(bias_box, self, "bias_e", "Sensitivities:", 
                       tooltip="List of possible e values (comma separated) for SAbias reliability estimates.", 
                       callback=partial(self.method_param_changed, 1))
        
        bagged_box = method_box(rbox, "Variance of bagged models",
                                "bagged_variance")
        
        OWGUI.spin(bagged_box, self, "bagged_m", 2, 100, step=1,
                   label="Models:",
                   tooltip="Number of bagged models to be used with BAGV estimate.",
                   callback=partial(self.method_param_changed, 2),
                   keyboardTracking=False)
        
        local_cv_box = method_box(rbox, "Local cross validation",
                                  "local_cv")
        
        OWGUI.spin(local_cv_box, self, "local_cv_k", 0, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in LCV estimate.",
                   callback=partial(self.method_param_changed, 3),
                   keyboardTracking=False)
        
        local_pe = method_box(rbox, "Local modeling of prediction error",
                              "local_model_pred_error")
        
        OWGUI.spin(local_pe, self, "local_pe_k", 0, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in CNK estimate.",
                   callback=partial(self.method_param_changed, 4),
                   keyboardTracking=False)
        
        bagging_cnn = method_box(rbox, "Bagging variance c-neighbors",
                                 "bagging_variance_cn")
        
        OWGUI.spin(bagging_cnn, self, "bagged_cn_m", 2, 100, step=1,
                   label="Models:",
                   tooltip="Number of bagged models to be used with BVCK estimate.",
                   callback=partial(self.method_param_changed, 5),
                   keyboardTracking=False)
        
        OWGUI.spin(bagging_cnn, self, "bagged_cn_k", 0, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in BVCK estimate.",
                   callback=partial(self.method_param_changed, 5),
                   keyboardTracking=False)
        
        mahalanobis_box = method_box(rbox, "Mahalanobis distance",
                                     "mahalanobis_distance")
        OWGUI.spin(mahalanobis_box, self, "mahalanobis_k", 0, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in BVCK estimate.",
                   callback=partial(self.method_param_changed, 6),
                   keyboardTracking=False)
        
        box = OWGUI.widgetBox(self.controlArea, "Output")
        
        OWGUI.checkBox(box, self, "include_error", "Include prediction error",
                       tooltip="Include predicion error in the output",
                       callback=self.commit_if)
        
        OWGUI.checkBox(box, self, "include_class", "Include orignial class",
                       tooltip="Include orignal class.",
                       callback=self.commit_if)
        
        OWGUI.checkBox(box, self, "include_input_features", "Include input features",
                       tooltip="Include faetures from the input data set.",
                       callback=self.commit_if)
        
        self.learner = None
        self.train_data = None
        self.test_data = None
        self.output_changed = False
         
        self.invalidate_results()
        
    def set_train_data(self, data=None):
        self.train_data = data
        self.invalidate_results()
        
    def set_test_data(self, data=None):
        self.test_data = data
        self.invalidate_results()
        
    def set_learner(self, learner=None):
        self.learner = learner
        self.invalidate_results()
        
    def handleNewSignals(self):
        name = test = train = ""
        if self.learner:
            name = getattr(self.learner, "name") or type(self.learner).__name__
        if self.train_data is not None:
            test = "Train Data: %i features, %i instances" % \
                (len(self.train_data.domain), len(self.train_data))
            
        if self.test_data is not None:
            test = "Train Data: %i features, %i instances" % \
                (len(self.test_data.domain), len(self.test_data))
        else:
            test = "Test data: using training data"
        
        self.info_box.setText("\n".join([name, test, train]))
        
        if self.learner and self._test_data() is not None:
            self.run()
        
    def invalidate_results(self, which=None):
        if which is None:
            self._cached_SA_estimates = None
            self.results = [None for f in self.methods]
            print "Invalidating all"
        else:
            for i in which:
                self.results[i] = None
            if 0 in which or 1 in which:
                self._cached_SA_estimates = None
            print "Invalidating", which
    
    def run(self):
        for i, (selected, method) in enumerate(self.methods):
            if self.results[i] is None and getattr(self, selected):
                print 'Computing', i, selected, method
                self.results[i] = method()
                print self.results[i]
        self.commit()
            
    def _test_data(self):
        if self.test_data is not None:
            return self.test_data
        else:
            return self.train_data
    
    def get_estimates(self, estimator):
        test = self._test_data()
        res = []
        for inst in test:
            value, prob = estimator(inst, result_type=Orange.core.GetBoth)
            res.append((value, prob))
        return res
                
    def run_estimation(self, method):
        rel = reliability.Learner(self.learner, estimators=[method])
        estimator = rel(self.train_data)
        return self.get_estimates(estimator)
    
    def get_SA_estimates(self):
        if not self._cached_SA_estimates:
            est = reliability.SensitivityAnalysis()
            rel = reliability.Learner(self.learner, estimators=[est])
            estimator = rel(self.train_data)
            # TODO: SAVar and SABias report both estimates in one pass,
            self._cached_SA_estimates = self.get_estimates(estimator)
        return self._cached_SA_estimates 
    
    def run_SAVar(self):
#        est = reliability.SensitivityAnalysis()
#        rel = reliability.Learner(self.learner, estimators=[est])
#        estimator = rel(self.train_data)
#        # TODO: SAVar and SABias report both estimates in one pass,
#        return self.get_estimates(estimator)
        return self.get_SA_estimates()
        
    def run_SABias(self):
#        est = reliability.SensitivityAnalysis()
#        rel = reliability.Learner(self.learner, estimators=[est])
#        estimator = rel(self.train_data) 
#        # TODO: SAVar and SABias report both estimates in one pass,   
#        return self.get_estimates(estimator)
        return self.get_SA_estimates()
    
    def run_BAGV(self):
        est = reliability.BaggingVariance(m=self.bagged_m)
        return self.run_estimation(est)
    
    def run_LCV(self):
        est = reliability.LocalCrossValidation(k=self.local_cv_k)
        return self.run_estimation(est)
    
    def run_CNK(self):
        est = reliability.CNeighbours(k=self.local_pe_k)
        return self.run_estimation(est)
    
    def run_BVCK(self):
        bagv = reliability.BaggingVariance(m=self.bagged_cn_m)
        cnk = reliability.CNeighbours(k=self.bagged_cn_k)
        est = reliability.BaggingVarianceCNeighbours(bagv, cnk)
        return self.run_estimation(est)
    
    def run_Mahalanobis(self):
        est = reliability.Mahalanobis(k=self.mahalanobis_k)
        return self.run_estimation(est)
    
    def method_selection_changed(self, method=None):
        if method is not None:
            i = [i for i, (name, _) in enumerate(self.methods)][0]
            self.invalidate_results([i])
        self.run()
    
    def method_param_changed(self, method=None):
        if method is not None:
            self.invalidate_results([method])
        self.run()
        
    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.output_changed = True
            
    def commit(self):
        from Orange.data import variable
        import numpy
        
        all_predictions = []
        all_estimates = []
        score_vars = []
        table = None
        if self._test_data() is not None:
            scores = []
            
            for res, (selected, method) in zip(self.results, self.methods):
                if res is not None and getattr(self, selected):
                    if selected == "bias_checked":
                        ei = 1
                    else:
                        ei = 0
                    values, estimates = [], []
                    for value, probs in res:
                        values.append(value)
                        estimates.append(probs.reliability_estimate[ei])
                    name = estimates[0].method_name
                    var = variable.Continuous(name)
                    score_vars.append(var)
                    all_predictions.append(values)
                    all_estimates.append(estimates)
            data = [[] for _ in self._test_data()]
            for preds, estimations in zip(all_predictions, all_estimates):
                for d, p, e in zip(data, preds, estimations):
                    d.append(e.estimate)
            
            domain = Orange.data.Domain(score_vars, False)
            print data
            table = Orange.data.Table(domain, data)
            print table[:]
            
        self.send("Reliability Scores", table)
        self.output_changed = True
        
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWReliability()
    data = Orange.data.Table("housing")
    indices = Orange.core.MakeRandomIndices2(p0=20)(data)
    print indices
    data = data.select(indices, 0)
    
    learner = Orange.regression.tree.TreeLearner()
    w.set_learner(learner)
    w.set_train_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()
    
        
