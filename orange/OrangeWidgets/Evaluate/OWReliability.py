"""
<name>Reliability</name>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>310</priority>
<icon>icons/Reliability.png</icon>
"""

import Orange
from Orange.evaluation import reliability
from Orange.evaluation import testing
#from Orange.misc import progress_bar_milestones
from functools import partial
 
from OWWidget import *
import OWGUI

class OWReliability(OWWidget):
    settingsList = ["variance_checked", "bias_checked", "bagged_variance",
        "local_cv", "local_model_pred_error", "bagging_variance_cn", 
        "mahalanobis_distance", "var_e", "bias_e", "bagged_m", "local_cv_k",
        "local_pe_k", "bagged_cn_m", "bagged_cn_k", "mahalanobis_k",
        "include_error", "include_class", "include_input_features",
        "auto_commit"]
    
    def __init__(self, parent=None, signalManager=None, title="Reliability"):
        OWWidget.__init__(self, parent, signalManager, title, wantMainArea=False)
        
        self.inputs = [("Learner", Orange.core.Learner, self.set_learner),
                       ("Training Data", Orange.data.Table, self.set_train_data),
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
        self.bagged_m = 10
        self.local_cv_k = 2
        self.local_pe_k = 5
        self.bagged_cn_m = 5
        self.bagged_cn_k = 1
        self.mahalanobis_k = 3
        
        self.include_error = True
        self.include_class = True
        self.include_input_features = False
        self.auto_commit = False
        
        # (selected attr name, getter function, count of returned estimators, index of estimator)
        self.estimators = \
            [("variance_checked", self.get_SAVar, 3, 0),
             ("bias_checked", self.get_SABias, 3, 1),
             ("bagged_variance", self.get_BAGV, 1, 0),
             ("local_cv", self.get_LCV, 1, 0),
             ("local_model_pred_error", self.get_CNK, 2, 0),
             ("bagging_variance_cn", self.get_BVCK, 4, 0),
             ("mahalanobis_distance", self.get_Mahalanobis, 1, 0)]
        
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
        
        self.commit_button = b = OWGUI.button(box, self, "Commit",
                                              callback=self.commit,
                                              autoDefault=True)
        
        OWGUI.setStopper(self, b, cb, "output_changed", callback=self.commit)
        
        self.commit_button.setEnabled(any([getattr(self, selected) \
                                for selected, _, _, _ in  self.estimators]))
        
        self.learner = None
        self.train_data = None
        self.test_data = None
        self.output_changed = False
        self.train_data_has_no_class = False
        self.train_data_has_discrete_class = False
        self.invalidate_results()
        
    def set_train_data(self, data=None):
        self.error()
        self.train_data_has_no_class = False
        self.train_data_has_discrete_class = False
        
        if data is not None:
            if not self.isDataWithClass(data, Orange.core.VarTypes.Continuous):
                if not data.domain.class_var:
                    self.train_data_has_no_class = True
                elif not isinstance(data.domain.class_var,
                                    Orange.data.variable.Continuous):
                    self.train_data_has_discrete_class = True
                    
                data = None
        
        self.train_data = data
        self.invalidate_results() 
        
    def set_test_data(self, data=None):
        self.test_data = data
        self.invalidate_results()
        
    def set_learner(self, learner=None):
        self.learner = learner
        self.invalidate_results()
        
    def handleNewSignals(self):
        name = "No learner on input"
        train = "No train data on input"
        test = "No test data on input"
        
        if self.learner:
            name = "Learner: " + (getattr(self.learner, "name") or type(self.learner).__name__)
            
        if self.train_data is not None:
            train = "Train Data: %i features, %i instances" % \
                (len(self.train_data.domain), len(self.train_data))
        elif self.train_data_has_no_class:
            train = "Train Data has no class variable"
        elif self.train_data_has_discrete_class:
            train = "Train Data doesn't have a continuous class"
            
        if self.test_data is not None:
            test = "Test Data: %i features, %i instances" % \
                (len(self.test_data.domain), len(self.test_data))
        elif self.train_data:
            test = "Test data: using training data"
        
        self.info_box.setText("\n".join([name, train, test]))
        
        if self.learner and self._test_data() is not None:
            self.commit_if()
        
    def invalidate_results(self, which=None):
        if which is None:
            self.results = [None for f in self.estimators]
#            print "Invalidating all"
        else:
            for i in which:
                self.results[i] = None
#            print "Invalidating", which
        
    def run(self):
        plan = []
        estimate_index = 0
        for i, (selected, method, count, offset) in enumerate(self.estimators):
            if self.results[i] is None and getattr(self, selected):
                plan.append((i, method, estimate_index + offset))
                estimate_index += count
                
        estimators = [method() for _, method, _ in plan]
        
        if not estimators:
            return
            
        pb = OWGUI.ProgressBar(self, len(self._test_data()))
        estimates = self.run_estimation(estimators, pb.advance)
        pb.finish()
        
        self.predictions = [v for v, _ in estimates]
        estimates = [prob.reliability_estimate for _, prob in estimates]
        
        for i, (index, method, estimate_index) in enumerate(plan):
            self.results[index] = [e[estimate_index] for e in estimates]
        
    def _test_data(self):
        if self.test_data is not None:
            return self.test_data
        else:
            return self.train_data
    
    def get_estimates(self, estimator, advance=None):
        test = self._test_data()
        res = []
        for i, inst in enumerate(test):
            value, prob = estimator(inst, result_type=Orange.core.GetBoth)
            res.append((value, prob))
            if advance:
                advance()
        return res
                
    def run_estimation(self, estimators, advance=None):
        rel = reliability.Learner(self.learner, estimators=estimators)
        estimator = rel(self.train_data)
        return self.get_estimates(estimator, advance) 
    
    def get_SAVar(self):
        return reliability.SensitivityAnalysis(e=eval(self.var_e))
    
    def get_SABias(self):
        return reliability.SensitivityAnalysis(e=eval(self.bias_e))
    
    def get_BAGV(self):
        return reliability.BaggingVariance(m=self.bagged_m)
    
    def get_LCV(self):
        return reliability.LocalCrossValidation(k=self.local_cv_k)
    
    def get_CNK(self):
        return reliability.CNeighbours(k=self.local_pe_k)
    
    def get_BVCK(self):
        bagv = reliability.BaggingVariance(m=self.bagged_cn_m)
        cnk = reliability.CNeighbours(k=self.bagged_cn_k)
        return reliability.BaggingVarianceCNeighbours(bagv, cnk)
    
    def get_Mahalanobis(self):
        return reliability.Mahalanobis(k=self.mahalanobis_k)
    
    def method_selection_changed(self, method=None):
        self.commit_button.setEnabled(any([getattr(self, selected) \
                                for selected, _, _, _ in  self.estimators]))
        self.commit_if()
    
    def method_param_changed(self, method=None):
        if method is not None:
            self.invalidate_results([method])
        self.commit_if()
        
    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.output_changed = True
            
    def commit(self):
        from Orange.data import variable
        name_mapper = {"Mahalanobis absolute": "Mahalanobis"}
        all_predictions = []
        all_estimates = []
        score_vars = []
        features = []
        table = None
        if self.learner and self.train_data is not None \
                and self._test_data() is not None:
            self.run()
            
            scores = []
            if self.include_class and not self.include_input_features:
                original_class = self._test_data().domain.class_var
                features.append(original_class)
                
            if self.include_class:
                prediction_var = variable.Continuous("Prediction")
                features.append(prediction_var)
                
            if self.include_error:
                error_var = variable.Continuous("Error")
                abs_error_var = variable.Continuous("Abs. Error")
                features.append(error_var)
                features.append(abs_error_var)
                
            for estimates, (selected, method, _, _) in zip(self.results, self.estimators):
                if estimates is not None and getattr(self, selected):
                    name = estimates[0].method_name
                    name = name_mapper.get(name, name)
                    var = variable.Continuous(name)
                    features.append(var)
                    score_vars.append(var)
                    all_estimates.append(estimates)
                    
            if self.include_input_features:
                dom = self._test_data().domain
                attributes = list(dom.attributes) + features
                domain = Orange.data.Domain(attributes, dom.class_var)
                domain.add_metas(dom.get_metas())
                
                data = Orange.data.Table(domain, self._test_data())
            else:
                domain = Orange.data.Domain(features, None)
                data = Orange.data.Table(domain, [[None] * len(features) for _ in self._test_data()])
            
            if self.include_class:
                for d, inst, pred in zip(data, self._test_data(), self.predictions):
                    if not self.include_input_features:
                        d[features[0]] = float(inst.get_class())
                    d[prediction_var] = float(pred)
            
            if self.include_error:
                for d, inst, pred in zip(data, self._test_data(), self.predictions):
                    error = float(pred) - float(inst.get_class())
                    d[error_var] = error
                    d[abs_error_var] = abs(error)
                    
            for estimations, var in zip(all_estimates, score_vars):
                for d, e in zip(data, estimations):
                    d[var] = e.estimate
            
            table = data
            
        self.send("Reliability Scores", table)
        self.output_changed = False
        
        
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
    
        
