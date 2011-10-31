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
        
        self.outputs = [("Reliability Scores", Orange.data.Table)]
        
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
        self.local_cv_k = 2
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
            
        e_validator = QRegExpValidator(QRegExp(r"\s*(-?[0-9]+(\.[0-9]*)\s*,\s*)+"), self)
        variance_box = method_box(rbox, "Sensitivity analysis (variance)",
                                  "variance_checked")
        OWGUI.lineEdit(variance_box, self, "var_e", "Sensitivities:", 
                       tooltip="List of possible e values (comma separated) for SAvar reliability estimates.", 
                       callback=partial(self.method_param_changed, 0),
                       validator=e_validator)
        
        bias_box = method_box(rbox, "Sensitivity analysis (bias)",
                                    "bias_checked")
        OWGUI.lineEdit(bias_box, self, "bias_e", "Sensitivities:", 
                       tooltip="List of possible e values (comma separated) for SAbias reliability estimates.", 
                       callback=partial(self.method_param_changed, 1),
                       validator=e_validator)
        
        bagged_box = method_box(rbox, "Variance of bagged models",
                                "bagged_variance")
        
        OWGUI.spin(bagged_box, self, "bagged_m", 2, 100, step=1,
                   label="Models:",
                   tooltip="Number of bagged models to be used with BAGV estimate.",
                   callback=partial(self.method_param_changed, 2),
                   keyboardTracking=False)
        
        local_cv_box = method_box(rbox, "Local cross validation",
                                  "local_cv")
        
        OWGUI.spin(local_cv_box, self, "local_cv_k", 2, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in LCV estimate.",
                   callback=partial(self.method_param_changed, 3),
                   keyboardTracking=False)
        
        local_pe = method_box(rbox, "Local modeling of prediction error",
                              "local_model_pred_error")
        
        OWGUI.spin(local_pe, self, "local_pe_k", 1, 20, step=1,
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
        
        OWGUI.spin(bagging_cnn, self, "bagged_cn_k", 1, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in BVCK estimate.",
                   callback=partial(self.method_param_changed, 5),
                   keyboardTracking=False)
        
        mahalanobis_box = method_box(rbox, "Mahalanobis distance",
                                     "mahalanobis_distance")
        OWGUI.spin(mahalanobis_box, self, "mahalanobis_k", 1, 20, step=1,
                   label="Nearest neighbors:",
                   tooltip="Number of nearest neighbors used in BVCK estimate.",
                   callback=partial(self.method_param_changed, 6),
                   keyboardTracking=False)
        
        box = OWGUI.widgetBox(self.controlArea, "Output")
        
        OWGUI.checkBox(box, self, "include_error", "Include prediction error",
                       tooltip="Include prediction error in the output",
                       callback=self.commit_if)
        
        OWGUI.checkBox(box, self, "include_class", "Include original class and prediction",
                       tooltip="Include original class and prediction in the output.",
                       callback=self.commit_if)
        
        OWGUI.checkBox(box, self, "include_input_features", "Include input features",
                       tooltip="Include features from the input data set.",
                       callback=self.commit_if)
        
        cb = OWGUI.checkBox(box, self, "auto_commit", "Commit on any change",
                            callback=self.commit_if)
        b = OWGUI.button(box, self, "Commit",
                         callback=self.commit,
                         autoDefault=True)
        
        OWGUI.setStopper(self, b, cb, "output_changed", callback=self.commit)
        
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
            train = "Train Data: %i features, %i instances" % \
                (len(self.train_data.domain), len(self.train_data))
            
        if self.test_data is not None:
            test = "Train Data: %i features, %i instances" % \
                (len(self.test_data.domain), len(self.test_data))
        else:
            test = "Test data: using training data"
        
        self.info_box.setText("\n".join([name, train, test]))
        
        if self.learner and self._test_data() is not None:
            self.run()
        
    def invalidate_results(self, which=None):
        if which is None:
            self._cached_SA_estimates = None
            self.results = [None for f in self.methods]
#            print "Invalidating all"
        else:
            for i in which:
                self.results[i] = None
            if 0 in which or 1 in which:
                self._cached_SA_estimates = None
#            print "Invalidating", which
    
    def run(self):
        for i, (selected, method) in enumerate(self.methods):
            if self.results[i] is None and getattr(self, selected):
#                print 'Computing', i, selected, method
                self.results[i] = method()
#                print self.results[i]
        self.commit_if()
            
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
            self._cached_SA_estimates = self.get_estimates(estimator)
        return self._cached_SA_estimates 
    
    def run_SAVar(self):
        est = reliability.SensitivityAnalysis(e=eval(self.var_e))
        return self.run_estimation(est)
#        return self.get_SA_estimates()
        
    def run_SABias(self):
        est = reliability.SensitivityAnalysis(e=eval(self.bias_e))
        return self.run_estimation(est)
#        return self.get_SA_estimates()
    
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
        
        all_predictions = []
        all_estimates = []
        score_vars = []
        features = []
        table = None
        if self._test_data() is not None:
            scores = []
            
            if self.include_class and not self.include_input_features:
                original_class = self._test_data().domain.class_var
                features.append(original_class)
                
            if self.include_class:
                prediction_var = variable.Continuous("Prediction")
                features.append(prediction_var)
                
            if self.include_error:
                error_var = variable.Continuous("Error")
                abs_error_var = variable.Continuous("Abs Error")
                features.append(error_var)
                features.append(abs_error_var)
                
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
                    features.append(var)
                    score_vars.append(var)
                    all_predictions.append(values)
                    all_estimates.append(estimates)
                    
            if self.include_input_features:
                dom = self._test_data().domain
                domain = Orange.data.Domain(dom.attributes, dom.class_var)
                domain.add_metas(dom.get_metas())
                data = Orange.data.Table(domain, self._test_data())
            else:
                domain = Orange.data.Domain([])
                data = Orange.data.Table(domain, [[] for _ in self._test_data()])
                
            for f in features:
                data.domain.add_meta(Orange.core.newmetaid(), f)
            
            if self.include_class:
                for d, inst, pred in zip(data, self._test_data(), all_predictions[0]):
                    if not self.include_input_features:
                        d[features[0]] = float(inst.get_class())
                    d[prediction_var] = float(pred)
            
            if self.include_error:
                for d, inst, pred in zip(data, self._test_data(), all_predictions[0]):
                    error = float(pred) - float(inst.get_class())
                    d[error_var] = error
                    d[abs_error_var] = abs(error)
                    
            for estimations, var in zip(all_estimates, score_vars):
                for d, e in zip(data, estimations):
                    d[var] = e.estimate
            
#            domain = Orange.data.Domain(features, False)
#            print data
#            table = Orange.data.Table(domain, data)
#            print data[:]
            table = data
            
        self.send("Reliability Scores", table)
        self.output_changed = True
        
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWReliability()
    data = Orange.data.Table("housing")
    indices = Orange.core.MakeRandomIndices2(p0=20)(data)
    data = data.select(indices, 0)
    
    learner = Orange.regression.tree.TreeLearner()
    w.set_learner(learner)
    w.set_train_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()
    
        
