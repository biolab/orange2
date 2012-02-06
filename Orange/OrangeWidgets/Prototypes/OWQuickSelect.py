"""<name>Quick Select</name>
<description>Select examples based on values of a single attribute</description>
<icon>icons/QuickSelect.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from OWDlgs import OWChooseImageSizeDlg
import OWQCanvasFuncts, OWColorPalette

class OWQuickSelect(OWWidget):
# Context for this widgets are problematic: when they are retrieved, values
# of attributes cannot be set since the corresponding listbox is not initialized
# yet. The widget does saves the selected values using a callback function,
# while when loading them, it safely stores them into a separate attribute
# from which they get restored after the corresponding listbox is filled.
# Even this unfortunately works only when widget is constructed anew or loaded
# from the file, but not when it simply gets new data. To handle this, the
# setData method does not reconstruct the listboxes when the domain is the same.
# This semi-works, but fails when reloading data from file (it changes the domain)
# and, worse, forgets the settings if domains are switched back
# and forth or if it recieves a None data signal. I don't know a cure for that.
   
    contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", DomainContextHandler.RequiredList + DomainContextHandler.IncludeMetaAttributes, selected="selectedAttribute"),
                                                    ],
                                                matchValue = DomainContextHandler.MatchValuesAttributes)}
       
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Quick Select", wantMainArea=0)
        self.inputs = [("Data", ExampleTable, self.setData, Default)]
        self.outputs = [("Selected Data", ExampleTable, Default), ("Remaining Data", ExampleTable)]
        self.icons = self.createAttributeIconDict()

        self.attributes = []
        self.selectedAttribute = []
        self.valuesList = []
        self.selectedValues = []
        self.inExamples = self.outSelected = self.outRemaining = 0
        self.data = None
        self.loadSettings()

        OWGUI.listBox(self.controlArea, self, "selectedAttribute", box="Attribute", labels="attributes", callback=self.attributeChanged, selectionMode=QListWidget.SingleSelection)
        sv = OWGUI.listBox(self.controlArea, self, "selectedValues", box="Values", labels="valuesList", callback=self.updateOutput, selectionMode=QListWidget.ExtendedSelection)
        ib = OWGUI.widgetBox(self.controlArea, "Info", orientation=0)
        OWGUI.label(ib, self, "Input: %(inExamples)i instances")
        OWGUI.rubber(ib)
        OWGUI.label(ib, self, "Selected: %(outSelected)i instances")
        self.resize(300, 620)

    def setData(self, data):
        if data and self.data and data.domain == self.data.domain:
            self.data = data
            self.inExamples = len(self.data)
            self.updateOutput()
            return
        
        self.closeContext()
        self.data = data
        if not data:
            self.attributes = []
            self.valuesList = self.selectedValues = []
            self.inExamples = 0
        else:
            self.inExamples = len(self.data)
            self.rawAttributes = [attr for attr in list(data.domain) + data.domain.getmetas().values()
                                  if attr.varType in [orange.Variable.Discrete, orange.Variable.String]]
            self.attributes = [(attr.name, attr.varType) for attr in self.rawAttributes]
            self.selectedAttribute = []
        self.openContext("", data)
        self.attributeChanged()

    def settingsFromWidgetCallback(self, handler, context):
        context.selectedValues = self.selectedValues

    def settingsToWidgetCallback(self, handler, context):
        if hasattr(context, "selectedValues"):
             self.storedSelectedValues = context.selectedValues

    def attributeChanged(self):
        if self.data and self.selectedAttribute:
            attr = self.rawAttributes[self.selectedAttribute[0]]
            if attr.varType == orange.Variable.Discrete:
                self.valuesList = [a.decode("utf-8") for a in attr.values]
            else:
                self.valuesList = list(set(str(ex[attr]).decode("utf-8") for ex in self.data))
            if hasattr(self, "storedSelectedValues"):
                try:
                    # may fail if values do not match (can they not match?!)
                    self.selectedValues = self.storedSelectedValues
                except:
                    pass 
                delattr(self, "storedSelectedValues") 
        else:
            self.valuesList = []
        self.updateOutput()
        
    def updateOutput(self):
        if not self.data or not self.selectedAttribute or not self.selectedValues:
            self.send("Selected Data", None)
            self.send("Remaining Data", None)
            self.outSelected = self.outRemaining = 0
            return
        
        attr = self.rawAttributes[self.selectedAttribute[0]]
        pp = orange.Preprocessor_take() 
        pp.values[attr] = [self.valuesList[j].encode("utf-8") for j in self.selectedValues]
        selected = pp(self.data)
        self.send("Selected Data", selected)
        
        pp = orange.Preprocessor_drop() 
        pp.values[attr] = [self.valuesList[j].encode("utf-8") for j in self.selectedValues]
        remaining = pp(self.data)
        self.send("Remaining Data", remaining)

        self.outSelected = len(selected)
        self.outRemaining = len(remaining)
        
    def sendReport(self):
        self.reportSettings("Criteria", [("Attribute", self.attributes[self.selectedAttribute[0]][0] 
                                                              if self.selectedAttribute and 0 <= self.selectedAttribute[0] <= len(self.attributes) else _("(none)")),
                                            ("Values", ", ".join(self.valuesList[i] for i in self.selectedValues) if self.selectedValues else _("(none)"))])
        self.reportSettings("Input/Output", [("Input instances", self.inExamples),
                                                ("Selected instances", self.outSelected),
                                                ("Remaining instances", self.outRemaining)])
