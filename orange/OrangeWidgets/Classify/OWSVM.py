# coding=utf-8
"""
<name>SVM</name>
<description>Support Vector Machines learner/classifier.</description>
<icon>icons/BasicSVM.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
"""
import orange, orngSVM, OWGUI, sys
from OWWidget import *
from exceptions import SystemExit
from orngWrap import PreprocessedLearner

class OWSVM(OWWidget):
    settingsList=["C","nu","p", "eps", "probability","gamma","degree", 
                  "coef0", "kernel_type", "name", "useNu", "normalization"]
    def __init__(self, parent=None, signalManager=None, name="SVM"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)
        
        self.inputs = [("Example Table", ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", orange.Learner, Default),
                        ("Classifier", orange.Classifier, Default),
                        ("Support Vectors", ExampleTable)]

        self.kernel_type = 2
        self.gamma = 0.0
        self.coef0 = 0.0
        self.degree = 3
        self.C = 1.0
        self.p = 0.1
        self.eps = 1e-3
        self.nu = 0.5
        self.shrinking = 1
        self.probability=1
        self.useNu=0
        self.nomogram=0
        self.normalization=1
        self.data = None
        self.selFlag=False
        self.preprocessor = None
        self.name="SVM"

        OWGUI.lineEdit(self.controlArea, self, 'name', 
                       box='Learner/Classifier Name', 
                       tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        b = OWGUI.radioButtonsInBox(self.controlArea, self, "useNu", [], 
                                    box="SVM Type", 
                                    orientation = QGridLayout(), 
                                    addSpace=True)
        
        b.layout().addWidget(OWGUI.appendRadioButton(b, self, "useNu", "C-SVM",
                                                     addToLayout=False),
                             0, 0, Qt.AlignLeft)
        
        b.layout().addWidget(QLabel("Cost (C)", b), 0, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "C", 0.5, 512.0, 0.5, 
                                addToLayout=False, 
                                callback=lambda *x: self.setType(0), 
                                alignment=Qt.AlignRight,
                                tooltip= "Cost for a mis-classified training instance."),
                             0, 2)
        
        b.layout().addWidget(OWGUI.appendRadioButton(b, self, "useNu", u"ν-SVM", 
                                                     addToLayout=False),
                             1, 0, Qt.AlignLeft)
        
        b.layout().addWidget(QLabel(u"Complexity bound (\u03bd)", b), 1, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "nu", 0.1, 1.0, 0.1, 
                                tooltip="Lower bound on the ratio of support vectors",
                                addToLayout=False, 
                                callback=lambda *x: self.setType(1),
                                alignment=Qt.AlignRight),
                             1, 2)
        
        self.kernelBox=b = OWGUI.widgetBox(self.controlArea, "Kernel")
        self.kernelradio = OWGUI.radioButtonsInBox(b, self, "kernel_type",
                                        btnLabels=[u"Linear,   x∙y",
                                                   u"Polynomial,   (g x∙y + c)^d",
                                                   u"RBF,   exp(-g|x-y|²)",
                                                   u"Sigmoid,   tanh(g x∙y + c)"],
                                        callback=self.changeKernel)

        OWGUI.separator(b)
        self.gcd = OWGUI.widgetBox(b, orientation="horizontal")
        self.leg = OWGUI.doubleSpin(self.gcd, self, "gamma",0.0,10.0,0.0001,
                                label="  g: ",
                                orientation="horizontal",
                                callback=self.changeKernel,
                                alignment=Qt.AlignRight)
        
        self.led = OWGUI.doubleSpin(self.gcd, self, "coef0", 0.0,10.0,0.0001,
                                label="  c: ",
                                orientation="horizontal",
                                callback=self.changeKernel,
                                alignment=Qt.AlignRight)
        
        self.lec = OWGUI.doubleSpin(self.gcd, self, "degree", 0.0,10.0,0.5,
                                label="  d: ",
                                orientation="horizontal",
                                callback=self.changeKernel,
                                alignment=Qt.AlignRight)

        OWGUI.separator(self.controlArea)
        
        self.optionsBox=b=OWGUI.widgetBox(self.controlArea, "Options",
                                          addSpace=True)
        
        OWGUI.doubleSpin(b,self, "eps", 0.0005, 1.0, 0.0005,
                         label=u"Numerical tolerance",
                         labelWidth = 180,
                         orientation="horizontal",
                         tooltip="Numerical tolerance of termination criterion.",
                         alignment=Qt.AlignRight)

        self.probBox = OWGUI.checkBox(b,self, "probability",
                                      label="Estimate class probabilities",
                                      tooltip="Create classifiers that support class probability estimation."
                                      )
        
        OWGUI.checkBox(b, self, "normalization",
                       label="Normalize data", 
                       tooltip="Use data normalization")

        self.paramButton=OWGUI.button(self.controlArea, self, "Automatic parameter search",
                                      callback=self.parameterSearch,
                                      tooltip="Automatically searches for parameters that optimize classifier accuracy", 
                                      debuggingEnabled=0)
        
        self.paramButton.setDisabled(True)

        OWGUI.button(self.controlArea, self,"&Apply", 
                     callback=self.applySettings, 
                     default=True)
        
        OWGUI.rubber(self.controlArea)
        
        self.loadSettings()
        self.changeKernel()
        self.searching=False
        self.applySettings()

    def sendReport(self):
        if self.kernel_type == 0:
            kernel = "Linear, x.y"
        elif self.kernel_type == 1:
            kernel = "Polynomial, (%.4f*x.y+%.4f)<sup>%.4f</sup>" % (self.gamma, self.coef0, self.degree)
        elif self.kernel_type == 2:
            kernel = "RBF, e<sup>-%.4f*(x-y).(x-y)</sup>" % self.gamma
        else:
            kernel = "Sigmoid, tanh(%.4f*x.y+%.4f)" % (self.gamma, self.coef0)
        self.reportSettings("Learning parameters",
                            [("Kernel", kernel),
                             ("Cost (C)", self.C),
                             ("Numeric precision", self.eps),
                             self.useNu and ("Complexity bound (nu)", self.nu),
                             ("Estimate class probabilities", OWGUI.YesNo[self.probability]),
                             ("Normalize data", OWGUI.YesNo[self.normalization])])
        self.reportData(self.data)
        
    def setType(self, type):
        self.useNu = type
        
    def changeKernel(self):
        if self.kernel_type==0:
            for a,b in zip([self.leg, self.led, self.lec], [1,1,1]):
                a.setDisabled(b)
        elif self.kernel_type==1:
            for a,b in zip([self.leg, self.led, self.lec], [0,0,0]):
                a.setDisabled(b)
        elif self.kernel_type==2:
            for a,b in zip([self.leg, self.led, self.lec], [0,1,1]):
                a.setDisabled(b)
        elif self.kernel_type==3:
            for a,b in zip([self.leg, self.led, self.lec], [0,0,1]):
                a.setDisabled(b)

    def setData(self, data=None):
        self.data = self.isDataWithClass(data, checkMissing=True) and data or None
        self.paramButton.setDisabled(not self.data)
        
    def setPreprocessor(self, pp):
        self.preprocessor = pp
        
    def handleNewSignals(self):
        self.applySettings()

    def applySettings(self):
        self.learner=orngSVM.SVMLearner()
        for attr in ("name", "kernel_type", "degree", "shrinking", "probability", "normalization"):
            setattr(self.learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(self.learner, attr, float(getattr(self, attr)))

        self.learner.svm_type=orngSVM.SVMLearner.C_SVC

        if self.useNu:
            self.learner.svm_type=orngSVM.SVMLearner.Nu_SVC

        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.classifier=None
        self.supportVectors=None
        
        if self.data:
            if self.data.domain.classVar.varType==orange.VarTypes.Continuous:
                self.learner.svm_type+=3
            self.classifier=self.learner(self.data)
            self.supportVectors=self.classifier.supportVectors
            self.classifier.name=self.name
            
        self.send("Learner", self.learner)
        self.send("Classifier", self.classifier)
        self.send("Support Vectors", self.supportVectors)

    def parameterSearch(self):
        if not self.data:
            return
        if self.searching:
            self.searching=False
        else:
            self.kernelBox.setDisabled(1)
            self.optionsBox.setDisabled(1)
            self.progressBarInit()
            self.paramButton.setText("Stop")
            self.searching=True
            self.search_()

    def progres(self, f, best=None):
        qApp.processEvents()
        self.progressBarSet(f)
        if not self.searching:
            raise UnhandledException()

    def finishSearch(self):
        self.progressBarFinished()
        self.kernelBox.setDisabled(0)
        self.optionsBox.setDisabled(0)
        self.paramButton.setText("Automatic parameter search")
        self.searching=False

    def search_(self):
        learner=orngSVM.SVMLearner()
        for attr in ("name", "kernel_type", "degree", "shrinking", "probability", "normalization"):
            setattr(learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(learner, attr, float(getattr(self, attr)))

        learner.svm_type=0

        if self.useNu:
            learner.svm_type=1
        params=[]
        if self.useNu:
            params.append("nu")
        else:
            params.append("C")
        if self.kernel_type in [1,2]:
            params.append("gamma")
        if self.kernel_type==1:
            params.append("degree")
        try:
            learner.tuneParameters(self.data, params, 4, verbose=0, progressCallback=self.progres)
        except UnhandledException:
            pass
        for param in params:
            setattr(self, param, getattr(learner, param))
            
        self.finishSearch()

from exceptions import Exception
class UnhandledException(Exception):
    pass

import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWSVM()
    w.show()
    #d=orange.ExampleTable("../../doc/datasets/iris.tab")
    #w.setData(d)
    app.exec_()
    w.saveSettings()
