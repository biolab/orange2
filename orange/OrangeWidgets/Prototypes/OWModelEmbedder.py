"""
<name>Model Embedder</name>
<description>Embeds a model widget</description>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact>
<icon>icons/DistanceFile.png</icon>
<priority>1100</priority>
"""

import sip

from OWWidget import *

import OWGUI
import orange

import OWScatterPlot
import OWRadviz
import OWLinProj
import OWPolyviz
import OWClassificationTreeGraph
import OWNomogram
import OWMDS

class OWModelEmbedder(OWWidget):
    settingsList = []

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Model Embedder")
        
        self.inputs = [("Examples", orange.ExampleTable, self.setData, Default),
                       ("Model", orange.Example, self.setModel)]
        
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable)]
        
        self.ow = None
        self.data = None
        self.model = None
        
        self.resize(800, 600)
        self.widgets = {}
        
    def setData(self, data):
        self.data = data
        self.showWidget()
        
    def setModel(self, model):
        self.model = model
        self.showWidget()
        
    def setWidget(self, widgetType):
        if str(widgetType) in self.widgets:
            self.ow = self.widgets[str(widgetType)]
        else:
            self.ow = widgetType(self) 
            self.widgets[str(widgetType)] = self.ow
        return self.ow
                
    def showWidget(self):
        self.information()
        
        if self.ow is not None:
            self.ow.topWidgetPart.hide()
            self.ow.setLayout(self.layout())
        elif self.layout() is not None: 
            sip.delete(self.layout())
            
        self.ow = None
        if self.data is None: 
            self.information("No learning data given.")
            return
        if self.model is None: return
        if "model" not in self.model.domain: return
        if "label" in self.model.domain:
            attr = self.model["label"].value.split(', ')
        
        modelType = self.model["model"].value.upper()
        
        projWidget = None
        if modelType == "SCATTERPLOT" or modelType == "SCATTTERPLOT": 
            projWidget = self.setWidget(OWScatterPlot.OWScatterPlot)

        if modelType == "RADVIZ":
            projWidget = self.setWidget(OWRadviz.OWRadviz) 
            
        if modelType == "POLYVIZ": 
            projWidget = self.setWidget(OWPolyviz.OWPolyviz) 
            
        if projWidget is not None:
            self.ow.setData(self.data)
            self.ow.setShownAttributes(attr)
            self.ow.handleNewSignals() 
        
        #####################################
        ### TODO: add new modelTypes here ###
        #####################################
        
        if modelType == "SPCA" or modelType == "LINPROJ": 
            self.setWidget(OWLinProj.OWLinProj) 
            self.ow.setData(self.data)
            self.ow.setShownAttributes(attr)
            self.ow.handleNewSignals() 
            xAnchors, yAnchors = self.model["anchors"].value
            self.ow.updateGraph(None, setAnchors=1, XAnchors=xAnchors, YAnchors=yAnchors)
            
        if modelType == "TREE":
            self.setWidget(OWClassificationTreeGraph.OWClassificationTreeGraph)
            classifier = self.model["classifier"].value
            self.ow.ctree(classifier)
            
        if modelType == "BAYES":
            self.setWidget(OWNomogram.OWNomogram) 
            classifier = self.model["classifier"].value
            self.ow.classifier(classifier)
            
        if modelType == "KNN":
            exclude = [att for att in self.data.domain if att.name not in attr + [self.data.domain.classVar.name]]
            data2 = orange.Preprocessor_ignore(self.data, attributes = exclude)
            dist = orange.ExamplesDistanceConstructor_Euclidean(data2)
            smx = orange.SymMatrix(len(data2))
            smx.setattr('items', data2)
            pb = OWGUI.ProgressBar(self, 100)
            milestones  = orngMisc.progressBarMilestones(len(data2)*(len(data2)-1)/2, 100)
            count = 0
            for i in range(len(data2)):
                for j in range(i+1):
                    smx[i, j] = dist(data2[i], data2[j])
                    if count in milestones:
                        pb.advance()
                    count += 1
            pb.finish()
            self.setWidget(OWMDS.OWMDS)
            self.ow.cmatrix(smx)
            
        if self.ow is not None:
            self.ow.send = self.send
            if self.layout() is not None: sip.delete(self.layout())
            self.setLayout(self.ow.layout())
            self.ow.topWidgetPart.show()
        
        self.update()
        
        
if __name__ == "__main__":
    import sys
    from PyQt4 import QtGui
    
    app = QtGui.QApplication(sys.argv)
    view = OWModelEmbedder()
    view.show()
    view.setWindowTitle("Model Embedder")
    sys.exit(app.exec_())