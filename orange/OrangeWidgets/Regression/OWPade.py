"""
<name>Pade</name>
<description>Computes local partial derivatives</description>
<icon>icons/Pade.png</icon>
<priority>3500</priority>
"""

import orange, orngPade
from OWWidget import *
import OWGUI

class OWPade(OWWidget):

    settingsList = ["output", "method", "derivativeAsMeta", "originalAsMeta", "savedDerivativeAsMeta", "differencesAsMeta", "enableThreshold", "threshold"]
    contextHandlers = {"": PerfectDomainContextHandler("", ["outputAttr", ContextField("attributes", selected="dimensions")])}

    methodNames = ["First Triangle", "Star Univariate Regression", "Tube Regression"]
    methods = [orngPade.firstTriangle, orngPade.starUnivariateRegression, orngPade.tubedRegression]
    outputTypes = ["Qualitative constraint", "Quantitative differences"]
    
    def __init__(self, parent = None, signalManager = None, name = "Pade"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)  #initialize base class
        self.inputs = [("Examples", ExampleTable, self.onDataInput)]
        self.outputs = [("Examples", ExampleTable)]

        self.attributes = []
        self.dimensions = []
        self.output = 0
        self.outputAttr = 0
        self.derivativeAsMeta = 0
        self.savedDerivativeAsMeta = 0
        self.correlationsAsMeta = 1
        self.differencesAsMeta = 1
        self.originalAsMeta = 1
        self.enableThreshold = 0
        self.threshold = 0.0
        self.method = 2
        self.useMQCNotation = False
        #self.persistence = 40

        self.nNeighbours = 30

        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Attributes", addSpace = True)
        lb = self.lb = OWGUI.listBox(box, self, "dimensions", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.dimensionsChanged)
        hbox = OWGUI.widgetBox(box, orientation=0)
        OWGUI.button(hbox, self, "All", callback=self.onAllAttributes)
        OWGUI.button(hbox, self, "None", callback=self.onNoAttributes)
        lb.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        lb.setMinimumSize(200, 200)

        box = OWGUI.widgetBox(self.controlArea, "Method", addSpace = 24)
        OWGUI.comboBox(box, self, "method", callback = self.methodChanged, items = self.methodNames)
#        self.nNeighboursSpin = OWGUI.spin(box, self, "nNeighbours", 10, 200, 10, label = "Number of neighbours" + "  ", callback = self.methodChanged)
        #self.persistenceSpin = OWGUI.spin(box, self, "persistence", 0, 100, 5, label = "Persistence (0-100)" + "  ", callback = self.methodChanged, controlWidth=50)

        OWGUI.separator(box)
        hbox = OWGUI.widgetBox(box, orientation=0)
        threshCB = OWGUI.checkBox(hbox, self, "enableThreshold", "Ignore differences below ")
#        OWGUI.rubber(hbox, orientation = 0)
        ledit = OWGUI.lineEdit(hbox, self, "threshold", valueType=float, validator=QDoubleValidator(0, 1e30, 0, self), controlWidth=50)
        threshCB.disables.append(ledit)
        threshCB.makeConsistent()
        OWGUI.checkBox(box, self, "useMQCNotation", label = "Use MQC notation")

        box = OWGUI.radioButtonsInBox(self.controlArea, self, "output", self.outputTypes, box="Output class", addSpace = True, callback=self.dimensionsChanged)
        self.outputLB = OWGUI.comboBox(OWGUI.indentedBox(box), self, "outputAttr", callback=self.outputDiffChanged)

        box = OWGUI.widgetBox(self.controlArea, "Output meta attributes", addSpace = True)
        self.metaCB = OWGUI.checkBox(box, self, "derivativeAsMeta", label="Qualitative constraint")
        OWGUI.checkBox(box, self, "differencesAsMeta", label="Derivatives of selected attributes")
        OWGUI.checkBox(box, self, "correlationsAsMeta", label="Absolute values of derivatives")
        OWGUI.checkBox(box, self, "originalAsMeta", label="Original class attribute")

        self.applyButton = OWGUI.button(self.controlArea, self, "&Apply", callback=self.apply)

        #self.persistenceSpin.setEnabled(self.methods[self.method] == orngPade.canceling)
        #self.setFixedWidth(self.sizeHint().width())

    def sendReport(self):
        self.reportSettings("Learning parameters", 
                            [("Attributes derived by", ", ".join(self.attributes[i][0] for i in self.dimensions) or "none"),
                             ("Method", self.methodNames[self.method]),
                             ("Threshold", self.threshold if self.enableThreshold else "None"),
                           ])
        self.reportSettings("Output", 
                            [("Label", self.outputTypes[self.output]),
                             not self.output and ("Notation", ["Pade", "QUIN"][self.useMQCNotation]),
                             ("Meta attributes", ", ".join(s for s, c in [("qualitative constraint", self.derivativeAsMeta),
                                                                         ("derivative", self.differencesAsMeta),
                                                                         ("absolute derivative", self.correlationsAsMeta),
                                                                         ("original class", self.originalAsMeta)] if c) or "none")])
        self.reportData(self.data)

    def onAllAttributes(self):
        self.dimensions = range(len(self.attributes))
        self.dimensionsChanged()

    def onNoAttributes(self):
        self.dimensions = []
        self.dimensionsChanged()

    def outputDiffChanged(self):
        if not self.output:
            self.output = 1
        self.dimensionsChanged()

    def dimensionsChanged(self):
        if self.output and self.dimensions:
            if not self.metaCB.isEnabled():
                self.derivativeAsMeta = self.savedDerivativeAsMeta
                self.metaCB.setEnabled(True)
        else:
            if self.metaCB.isEnabled():
                self.savedDerivativeAsMeta = self.derivativeAsMeta
                self.derivativeAsMeta = 0
                self.metaCB.setEnabled(False)

        self.applyButton.setEnabled(bool(self.dimensions) or (self.output and bool(self.contAttributes)))

    def methodChanged(self):
        self.deltas = None
        #self.persistenceSpin.setEnabled(self.methods[self.method] == orngPade.canceling)
        #self.nNeighboursSpin.setEnabled(bool(self.method==3))


    def onDataInput(self, data):
        self.closeContext()
        if data and self.isDataWithClass(data, orange.VarTypes.Continuous, checkMissing=True):
            orngPade.makeBasicCache(data, self)

            icons = OWGUI.getAttributeIcons()
            self.outputLB.clear()
            for attr in self.contAttributes:
                self.outputLB.addItem(icons[attr.varType], attr.name)

            self.dimensions = range(len(self.attributes))
        else:
            orngPade.makeEmptyCache(self)
            self.dimensions = []

        self.openContext("", data)
        self.dimensionsChanged()


    def apply(self):
        data = self.data
        if not data or not self.contAttributes:
            self.send("Examples", None)
            return

        if not self.deltas:
            self.deltas = [[None] * len(self.contAttributes) for x in xrange(len(self.data))]
        if not getattr(self, "errors", None):
            self.errors = [[None] * len(self.contAttributes) for x in xrange(len(self.data))]

        dimensionsToCompute = [d for d in self.dimensions if not self.deltas[0][d]]
        if self.output and self.outputAttr not in self.dimensions and not self.deltas[0][self.outputAttr]:
            dimensionsToCompute.append(self.outputAttr)
        if dimensionsToCompute:
            self.progressBarInit()
            self.methods[self.method](self, dimensionsToCompute, self.progressBarSet)
            self.progressBarFinished()

        paded, derivativeID, metaIDs, classID, corrIDs = orngPade.createQTable(self, data, self.dimensions,
                                                             not self.output and -1 or self.outputAttr,
                                                             self.enableThreshold and abs(self.threshold),
                                                             self.useMQCNotation, self.derivativeAsMeta, self.differencesAsMeta, False, self.originalAsMeta)
        self.send("Examples", paded)



if __name__=="__main__":
    import sys

    a=QApplication(sys.argv)
    ow=OWPade()
    ow.show()
    #ow.onDataInput(orange.ExampleTable(r"c:\D\ai\Orange\test\squin\xyz-t"))
    #ow.onDataInput(orange.ExampleTable(r"C:\delo\PADE\JJ_testi\sinxsiny\sinxsiny_noise05.tab"))
    a.exec_()

    ow.saveSettings()
