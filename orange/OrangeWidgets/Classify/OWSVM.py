"""
<name>SVM</name>
<description>Support Vector Machines learner/classifier.</description>
<icon>icons/BasicSVM.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>100</priority>
"""

import orange, orngSVM, OWGUI, thread
from OWWidget import *
from threading import Thread
from exceptions import SystemExit

class OWSVM(OWWidget):
    settingsList=["C","nu","p","probability","shrinking","gamma","degree", "coef0", "kernel_type", "name", "useNu", "nomogram"]
    def __init__(self, parent=None, signalManager=None, name="SVM"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs=[("Example Table", ExampleTable, self.cdata)]
        self.outputs=[("Learner", orange.Learner),("Classifier", orange.Classifier),("Support Vectors", ExampleTable)]

        self.kernel_type = 2
        self.gamma = 0.0
        self.coef0 = 0.0
        self.degree = 3
        self.C = 1.0
        self.p = 0.5
        self.eps = 1e-3
        self.nu = 0.5
        self.shrinking = 1
        self.probability=1
        self.useNu=0
        self.nomogram=0
        self.data = None
        self.selFlag=False
        self.name="SVM Learner/Classifier"
        
        OWGUI.lineEdit(self.controlArea, self, 'name', box='Learner/Classifier Name', tooltip='Name to be used by other widgets to identify your learner/classifier.')
        OWGUI.separator(self.controlArea)

        self.kernelBox=b=QVButtonGroup("Kernel", self.controlArea)
        self.kernelradio = OWGUI.radioButtonsInBox(b, self, "kernel_type", btnLabels=["Linear,   x.y", "Polynomial,   (g*x.y+c)^d",
                    "RBF,   exp(-g*(x-y).(x-y))", "Sigmoid,   tanh(g*x.y+c)"], callback=self.changeKernel)

        self.gcd = OWGUI.widgetBox(b, orientation="horizontal")
        self.leg = OWGUI.doubleSpin(self.gcd, self, "gamma",0.0,10.0,0.0001, label="g: ", orientation="horizontal", callback=self.changeKernel)
        self.led = OWGUI.doubleSpin(self.gcd, self, "coef0", 0.0,10.0,0.0001, label="  c: ", orientation="horizontal", callback=self.changeKernel)
        self.lec = OWGUI.doubleSpin(self.gcd, self, "degree", 0.0,10.0,0.5, label="  d: ", orientation="horizontal", callback=self.changeKernel)

        OWGUI.separator(self.controlArea)
        
        self.optionsBox=b=OWGUI.widgetBox(self.controlArea, "Options", addSpace = True)
        OWGUI.doubleSpin(b,self, "C", 0.0, 100.0, 0.5, label="Model complexity (C)", labelWidth = 120, orientation="horizontal")
        OWGUI.doubleSpin(b,self, "p", 0.0, 10.0, 0.1, label="Tolerance (p)", labelWidth = 120, orientation="horizontal")
        OWGUI.doubleSpin(b,self, "eps", 0.0, 0.5, 0.001, label="Numeric precision (eps)", labelWidth = 120, orientation="horizontal")

        OWGUI.checkBox(b,self, "probability", label="Support probabilities")        
        OWGUI.checkBox(b,self, "shrinking", label="Shrinking")
        OWGUI.checkBox(b,self, "useNu", label="Limit the number of support vectors", callback=lambda:self.nuBox.setDisabled(not self.useNu))
        self.nuBox=OWGUI.doubleSpin(OWGUI.indentedBox(b), self, "nu", 0.0,1.0,0.1, label="Complexity bound (nu)", labelWidth = 120, orientation="horizontal", tooltip="Upper bound on the ratio of support vectors")
        self.nomogramBox=OWGUI.checkBox(b, self, "nomogram", "For nomogram if posible", tooltip="Builds a model that can be visualized in a nomogram (works only\nfor discrete class values with two values)")

        OWGUI.separator(self.controlArea)
                
        self.paramButton=OWGUI.button(self.controlArea, self, "Automatic parameter search", callback=self.parameterSearch,
                                      tooltip="Automaticaly searches for parameters that optimize classifier acuracy") 

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self,"&Apply settings", callback=self.applySettings)
        self.nuBox.setDisabled(not self.useNu)
        self.adjustSize()
        self.loadSettings()
        self.changeKernel()
        self.thread=MyThread(self)
        self.lock=thread.allocate_lock()
        self.terminateThread=False
        self.applySettings()
        
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

    def cdata(self, data=None):
        if data:
            self.data=data
        else:
            self.data=None
        self.applySettings()

    def applySettings(self):
        self.learner=orngSVM.SVMLearner()
        for attr in ("name", "kernel_type", "degree", "shrinking", "probability"):
            setattr(self.learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(self.learner, attr, float(getattr(self, attr)))

        self.learner.svm_type=0
        
        if self.useNu:
            self.learner.svm_type=1

        self.classifier=None
        self.supportVectors=None
        if self.nomogram and self.data and self.data.domain.classVar.varType==orange.VarTypes.Discrete and \
                len(self.data.domain.classVar.values)==2 and self.kernel_type==0:
            import orngLR_Jakulin
            self.learner=orngSVM.BasicSVMLearner()
            for attr in ("name", "shrinking"):
                setattr(self.learner, attr, getattr(self, attr))
            for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
                setattr(self.learner, attr, float(getattr(self, attr)))
            self.learner.kernel=self.kernel_type
            self.learner.for_nomogram=1
            self.classifier=orngLR_Jakulin.MarginMetaLearner(self.learner, folds=1)(self.data)
            self.classifier.name=self.name
            self.classifier.domain=self.data.domain
            self.classifier.data=self.data
            self.supportVectors=self.data.getitemsref(self.classifier.classifier.model["SVi"])
        elif self.data:
            if self.data.domain.classVar.varType==orange.VarTypes.Continuous:
                self.learner.svm_type+=3
            self.classifier=self.learner(self.data)
            self.supportVectors=self.classifier.supportVectors
            self.classifier.name=self.name
        if self.learner:
            self.send("Learner", self.learner)
        if self.classifier:
            self.send("Classifier", self.classifier)
        if self.supportVectors:
            self.send("Support Vectors", self.supportVectors)

    def parameterSearch(self):
        if self.thread.isAlive():
            self.lock.acquire()
            self.terminateThread=True
            self.lock.release()
            #self.thread.join()
            self.kernelBox.setDisabled(0)
            self.optionsBox.setDisabled(0)
            self.paramButton.setText("Automatic parameter search")
        else:
            self.kernelBox.setDisabled(1)
            self.optionsBox.setDisabled(1)
            self.progressBarInit()
            self.thread=MyThread(self)
            self.thread.start()
            self.paramButton.setText("Stop")
            
    def progres(self, f, best):
        self.best=best
        self.progressBarSet(int(f*100))
        if self.terminateThread:
            self.lock.acquire()
            self.terminateThread=False
            self.lock.release()
            import sys
            sys.exit()
            raise "thread exit"
            
    def finishSearch(self):
        del self.best["error"]
        for key in self.best.keys():
            self.__setattr__(key, self.best[key])
        self.progressBarFinished()
        self.kernelBox.setDisabled(0)
        self.optionsBox.setDisabled(0)
        self.paramButton.setText("Automatic parameter search")
        
        
class MyThread(Thread):
    def __init__(self, widget):
        apply(Thread.__init__,(self,))
        self.widget=widget

    def run(self):
        params={}
        if self.widget.useNu:
            params["nu"]=[0.25, 0.5, 0.75]
        else:
            params["C"]=map(lambda g:2**g, range(-5,10,2))
        if self.widget.kernel_type in [1,2]:
            params["gamma"]=map(lambda g:2**g, range(-3,10,2))
        if self.widget.kernel_type==1:
            params["degree"]=[1,2,3]
        best={}
        try:
            best=orngSVM.parameter_selection(orngSVM.SVMLearner(),self.widget.data, 4, params, best, callback=self.widget.progres)
        except SystemExit:
            pass
        self.widget.finishSearch()
import sys
if __name__=="__main__":
    app=QApplication(sys.argv)
    w=OWSVM()
    app.setMainWidget(w)
    w.show()
    d=orange.ExampleTable("../../doc/datasets/tic_tac_toe.tab")
    w.cdata(d)
    app.exec_loop()
    w.saveSettings()
