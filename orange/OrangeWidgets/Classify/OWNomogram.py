"""
<name>Nomogram</name>
<description>Visualizes Naive Bayesian or logistic regression or any linear classifier using interactive nomogram.</description>
<category>Classification</category>
<icon>icons/Nomogram.png</icon>
<priority>9998</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with Naive Bayes or logistic regression classifier
#

import Numeric
import orange
import orngLR
import OWGUI
import orngLinVis
import orngSVM
from OWWidget import *
#from OW_KN_NomogramOptions import *
from OWNomogramGraph import * 
#import OW_NomogramGraph_Martin
#reload(OW_NomogramGraph_Martin)


def getStartingPoint(d, min):
    if min<0:
        curr_num = arange(-min+d)
        curr_num = curr_num[len(curr_num)-1]
        curr_num = -curr_num
    elif min - d <= 0:
        curr_num = 0
    else:
        curr_num = arange(min-d)
        curr_num = curr_num[len(curr_num)-1]
    return curr_num

def getRounding(d):
    rndFac = math.floor(math.log10(d));
    if rndFac<-2:
        rndFac = int(-rndFac)
    else:
        rndFac = 2
    return rndFac
    

class OWNomogram(OWWidget):
    settingsList = ["alignType", "contType", "bubble", "histogram", "histogram_size", "confidence_percent"]

    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "&Nomogram",
        """OWNomogram is an Orange Widget
for displaying a nomogram of a Naive Bayesian or logistic regression classifier.""",
        FALSE,
        TRUE)
        self.setWFlags(Qt.WResizeNoErase | Qt.WRepaintNoErase) #this works like magic.. no flicker during repaint!
        self.parent = parent        
#        self.setWFlags(self.getWFlags()+Qt.WStyle_Maximize)

        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.alignType = 0
        self.contType = 0
        self.yAxis = 0
        self.probability = 0
        self.table = 0
        self.verticalSpacing = 40
        self.verticalSpacingContinuous = 100
        self.fontSize = 9
        self.lineWidth = 1
        self.bubble = 1
        self.histogram = 1
        self.histogram_size = 10
        self.data = None
        self.cl = None
        self.confidence_check = 0
        self.confidence_percent = 95
        
        self.loadSettings()

        self.pointsName = ["Points","Log Odds"]
        self.totalPointsName = ["Total Points","Log Odds Sum"]

        #inputs
        self.inputs=[("Classifier", orange.Classifier, self.classifier, 1), ("Examples", ExampleTable, self.cdata, 1)]

        # GUI definition
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        
        # GENERAL TAB
        GeneralTab = QVGroupBox(self)

        self.alignRadio = OWGUI.radioButtonsInBox(GeneralTab, self,  'alignType', ['Left', '0-point'], box='Align',
                                                  tooltips=['Attributes in nomogram are left aligned', 'Attributes are not aligned, top scale represents true (normalized) regression coefficient value'],
                                                  callback=self.showNomogram)
        self.yAxisRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'yAxis', ['100', 'log odds'], 'yAxis',  
                                tooltips=['values are normalized on a 0-100 point scale','values on top axis show log-linear contribution of attribute to full model'],
                                callback=self.showNomogram)
        self.ContRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'contType',   ['1D', '2D'], 'Continuous',
                                tooltips=['Continuous attribute are presented on a single scale', 'Two dimensional space is used to present continuous attributes in nomogram.'],
                                callback=self.showNomogram)

        #self.yAxisRadio.setDisabled(True)
        self.probabilityCheck = OWGUI.checkBox(GeneralTab, self, 'probability','Show prediction',  tooltip='', callback = self.setProbability)
        #self.probabilityCheck.setDisabled(True)
        self.tableCheck = OWGUI.checkBox(GeneralTab, self, 'table','Show table',  tooltip='Show table of selected attribute values?')
        self.bubbleCheck = OWGUI.checkBox(GeneralTab, self, 'bubble', 'Show details bubble',  tooltip='Show details of selected attribute value in a roll-over blob.')
        self.tableCheck.setDisabled(True)
        
        self.tabs.insertTab(GeneralTab, "General")
        
        # TREE TAB
        NomogramStyleTab = QVGroupBox(self)

        self.verticalSpacingLabel = OWGUI.spin(NomogramStyleTab, self, 'verticalSpacing', 15, 100, box = 'Vertical spacing:',  tooltip='Define space (pixels) between adjacent attributes.')
        self.verticalSpacingLabel.setDisabled(True)
        self.fontSizeLabel = OWGUI.spin(NomogramStyleTab, self, 'fontSize', 4, 14, box = 'Font size:', tooltip='Font size of nomogram labels.')
        self.fontSizeLabel.setDisabled(True)
        self.lineWidthLabel = OWGUI.spin(NomogramStyleTab, self, 'lineWidth', 1, 10, box = 'Line width:',  tooltip='Define width of lines shown in nomogram.')
        self.lineWidthLabel.setDisabled(True)
        self.histogramCheck, self.histogramLabel = OWGUI.checkWithSpin(NomogramStyleTab, self, 'Histogram, max. size:', min=1, max=30, checked='histogram', value='histogram_size', step = 1, tooltip='-(TODO)-', checkCallback=self.showNomogram, spinCallback = self.showNomogram)
        self.histogramCheck.setChecked(False)
        self.histogramCheck.setDisabled(True)
        self.histogramLabel.setDisabled(True)

        # save button
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)

        # objects/gui widgets in settings tab for showing and adjusting confidence intervals properties       
        self.CICheck, self.CILabel = OWGUI.checkWithSpin(NomogramStyleTab, self, 'Confidence Interval (%):', min=1, max=99, step = 1, checked='confidence_check', value='confidence_percent', tooltip='-(TODO)-', checkCallback=self.showNomogram, spinCallback = self.showNomogram)
        self.CICheck.setChecked(False)
        self.CICheck.setDisabled(True)
        self.CILabel.setDisabled(True)
        
        self.tabs.insertTab(NomogramStyleTab, "Settings")
        
        #add a graph widget
        self.bnomogram = None
        self.box=QBoxLayout(self.mainArea, QVBoxLayout.TopToBottom, 0)
        self.graph=OWNomogramGraph(self.bnomogram, self.mainArea)
        self.graph.setMinimumWidth(200)
        self.header = OWNomogramHeader(None, self.mainArea)
        self.header.setMinimumHeight(self.verticalSpacing)
        self.header.setMaximumHeight(self.verticalSpacing)
        self.footer = OWNomogramHeader(None, self.mainArea)
        self.footer.setMinimumHeight(self.verticalSpacing*2+10)
        self.footer.setMaximumHeight(self.verticalSpacing*2+10)

        self.box.addWidget(self.header)
        self.box.addWidget(self.graph)
        self.box.addWidget(self.footer)
        self.repaint()
        self.update()

        # mouse pressed flag
        self.mousepr = False


    # Input channel: the Bayesian classifier   
    def nbClassifier(self, cl):
        # thisd subroutine computes standard error of estimated beta /for the time being use it only with discrete data
        def err(e, priorError, key, data):
            inf = 0.0
            sume = e[0]+e[1]
            for d in data:
                if d[at]==key:
                    inf += (e[0]*e[1]/sume/sume)
            inf = max(inf, 0.00000001)
            var = 1/inf - priorError*priorError
            return (math.sqrt(var))

        classVal = cl.domain.classVar
        att = cl.domain.attributes

        # calculate prior probability
        dist1 = max(0.00000001, cl.distribution[classVal[1]])
        dist0 = max(0.00000001, cl.distribution[classVal[0]])
        prior = dist0/dist1
        if self.data:
            sumd = dist1+dist0
            priorError = math.sqrt(1/((dist1*dist0/sumd/sumd)*len(self.data)))
        else:
            priorError = 0
        self.bnomogram = BasicNomogram(self, AttValue("Constant", Numeric.log(prior), error = priorError))

        if self.data:
            stat = orange.DomainBasicAttrStat(self.data)

        for at in range(len(att)):
            a = AttrLine(att[at].name, self.bnomogram)
            if att[at].varType == orange.VarTypes.Discrete:
                for cd in cl.conditionalDistributions[at].keys():
                    # calculuate thickness
                    conditional0 = max(cl.conditionalDistributions[at][cd][classVal[0]], 0.00000001)
                    conditional1 = max(cl.conditionalDistributions[at][cd][classVal[1]], 0.00000001)
                    beta = Numeric.log(conditional0/conditional1/prior)
                    if self.data:
                        #thickness = int(round(4.*float(len(self.data.filter({att[at].name:str(cd)})))/float(len(self.data))))
                        thickness = float(len(self.data.filter({att[at].name:str(cd)})))/float(len(self.data))
                        se = err(cl.conditionalDistributions[at][cd], priorError, cd, self.data) # standar error of beta 
                    else:
                        thickness = 0
                        se = 0
                        
                    a.addAttValue(AttValue(str(cd), beta, lineWidth=thickness, error = se))
            else:
                numOfPartitions = 50 

                if self.data:
                    maxAtValue = stat[at].max
                    minAtValue = stat[at].min
                else:
                    maxAtValue = cl.conditionalDistributions[at].keys()[len(cl.conditionalDistributions[at].keys())-1]
                    minAtValue = cl.conditionalDistributions[at].keys()[0]

                d = maxAtValue-minAtValue
                d = getDiff(d/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling                
                curr_num = getStartingPoint(d, minAtValue) 
                rndFac = getRounding(d)                

                n = sum = 0

                for i in range(2*numOfPartitions):
                    if curr_num+i*d>=minAtValue and curr_num+i*d<=maxAtValue:
                        # get thickness
                        if self.data:
                            curr_data = self.data.filter({att[at].name:(curr_num+i*d-d/2, curr_num+(i+1)*d+d/2)})
                            #thickness = 1+round(20.*float(len(curr_data))/len(self.data))
                            thickness = float(len(curr_data))/len(self.data)
                        else:
                            thickness = 0
                        
                        d_filter = filter(lambda x: x>curr_num+i*d-d/2 and x<curr_num+i*d+d/2, cl.conditionalDistributions[at].keys())
                        if len(d_filter)>0:
                            d_filter = d_filter[len(d_filter)/2]
                            conditional0 = max(cl.conditionalDistributions[at][d_filter][classVal[0]], 0.00000001)
                            conditional1 = max(cl.conditionalDistributions[at][d_filter][classVal[1]], 0.00000001)
                            a.addAttValue(AttValue(str(round(curr_num+i*d,rndFac)), Numeric.log(conditional0/conditional1/prior),lineWidth=thickness))
                a.continuous = True
            self.bnomogram.addAttribute(a)        

        self.graph.setCanvas(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the logistic regression classifier    
    def lrClassifier(self, cl):
        self.bnomogram = BasicNomogram(self, AttValue('Constant', -cl.beta[0], cl.beta_se[0]))
        at = cl.domain.attributes
        at_num = 1
        curr_att = ""
        a = None

        for i in range(len(at)):
            if at[i].getValueFrom:
                name = at[i].getValueFrom.variable.name
                var = at[i].getValueFrom.variable
            else:
                name = at[i].name
                var = at[i]
            if name != curr_att:
                curr_att = name
                a = AttrLine(name, self.bnomogram)
                at_num = at_num+1

                # get all values in variable and get thos that are fitted
                # if both lists are the same size its ok, else missing attributes have value 0
                if var.varType == orange.VarTypes.Discrete:
                    for v in range(len(var.values)):
                        val = 0
                        name = var.values[v]
                        for j in range(len(var.values)):
                            if i+j<len(at) and at[i+j].getValueFrom and at[i+j].getValueFrom.variable==var and at[i+j].getValueFrom.lookupTable[v].value==1.0:
                                val = -cl.beta[i+j+1]
                                break
                        a.addAttValue(AttValue(name,val))
                if var.varType == orange.VarTypes.Continuous:
                    if self.data:
                        bas = orange.DomainBasicAttrStat(self.data)
                        maxAtValue = bas[var].max
                        minAtValue = bas[var].min
                    else:
                        maxAtValue = 1.
                        minAtValue = -1.
                    numOfPartitions = 50. 
                    d = getDiff((maxAtValue-minAtValue)/numOfPartitions)

                    # get curr_num = starting point for continuous att. sampling
                    curr_num = getStartingPoint(d, minAtValue) 
                    rndFac = getRounding(d)

                    while curr_num<maxAtValue+d:
                        if abs(-curr_num*cl.beta[i+1])<0.000001:
                            a.addAttValue(AttValue("0.0", 0))
                        else:
                            a.addAttValue(AttValue(str(curr_num), -curr_num*cl.beta[i+1]))
                        curr_num += d
                    a.continuous = True
                    
                self.bnomogram.addAttribute(a)    

        #bnomogram.printOUT()                                                
        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setCanvas(self.bnomogram)
        self.bnomogram.show()

    def svmClassifier(self, cl):
        visualizer = orngLinVis.Visualizer(self.data, cl, buckets=1, dimensions=1)
        self.bnomogram = BasicNomogram(self, AttValue('Constant', visualizer.beta, 0))

        # get maximum and minimum values in visualizer.m        
        maxMap = reduce(Numeric.maximum, visualizer.m)
        minMap = reduce(Numeric.minimum, visualizer.m)

        coeff = 0 #
        at_num = 1
        for c in visualizer.coeff_names:
            if type(c[1])==str:
                for i in range(len(c)):
                    if i == 0:
                        a = AttrLine(c[i], self.bnomogram)
                        at_num = at_num + 1
                    else:
                        a.addAttValue(AttValue(c[i], -visualizer.coeffs[coeff]))
                        coeff = coeff + 1
            else:
                a = AttrLine(c[0], self.bnomogram)

                # get min and max from Data and transform coeff accordingly
                maxNew=maxMap[coeff]
                minNew=maxMap[coeff]
                if self.data:
                    bas = orange.DomainBasicAttrStat(self.data)
                    maxNew = bas[c[0]].max
                    minNew = bas[c[0]].min

                # transform SVM betas to nomogram appropirate betas
                beta = ((maxMap[coeff]-minMap[coeff])/(maxNew-minNew))*visualizer.coeffs[coeff]
                n = -minNew+minMap[coeff]
                
                numOfPartitions = 50
                d = getDiff((maxNew-minNew)/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minNew) 
                rndFac = getRounding(d)
                
                while curr_num<maxNew+d:
                    a.addAttValue(AttValue(str(curr_num), -(curr_num-minNew)*beta-minMap[coeff]*visualizer.coeffs[coeff]))
                    curr_num += d

                at_num = at_num + 1
                coeff = coeff + 1
                a.continuous = True
                
            self.bnomogram.addAttribute(a)
        self.cl.domain = orange.Domain(self.data.domain.classVar)
#        self.cl.domain.classVar = orange
#        self.cl.domain.classVar.values = self.data.domain.classVar.values       
        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setCanvas(self.bnomogram)
        self.bnomogram.show()
       
    # Input channel: the target outcome (optional)    
    def target(self, data):
        self.graph.setTarget(data)

    def classifier(self, cl):
        self.cl = cl
        self.updateNomogram()
        
    # Input channel: data
    def cdata(self, data):
        # call appropriate classifier
        self.data = data
        if not data:
            self.histogramCheck.setChecked(False)
            self.histogramCheck.setDisabled(True)
            self.histogramLabel.setDisabled(True)
            self.CICheck.setChecked(False)
            self.CICheck.setDisabled(True)
            self.CILabel.setDisabled(True)
        else:
            self.histogramCheck.setEnabled(True)
            self.histogramLabel.setEnabled(True)
            self.CICheck.setEnabled(True)
            self.CILabel.setEnabled(True)
        self.updateNomogram()
        

    def updateNomogram(self):
        def setNone():
            self.footer.setCanvas(None)
            self.header.setCanvas(None)
            self.graph.setCanvas(None)

        if type(self.cl) == orngSVM.BasicSVMClassifier and self.data:
                self.svmClassifier(self.cl)
        elif type(self.cl) == orange.BayesClassifier:
            if len(self.cl.domain.classVar.values)>2:
                QMessageBox("OWNomogram:", " Please use only Bayes classifiers that are induced on data with dichotomous class", QMessageBox.Warning,
                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            else:
                self.nbClassifier(self.cl)
        elif type(self.cl) == orange.LogRegClassifier:
            # get if there are any continuous attributes in data -> then we need data to compute margins
            cont = False
            for at in self.cl.domain.attributes:
                if not at.getValueFrom:
                    cont = True
            if self.data or not cont:
                self.lrClassifier(self.cl)            
        else:
            setNone()
        
    def setProbability(self):
        if self.probability and self.bnomogram:
            self.bnomogram.showAllMarkers()
        elif self.bnomogram:
            self.bnomogram.hideAllMarkers()

    def saveToFileCanvas(self):
        sizeW = self.graph.canvas().pright
        sizeH = self.graph.canvas().gbottom + self.header.canvas().size().height() + self.footer.canvas().size().height()
        size = QSize(sizeW, sizeH)

        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()

        # create buffers and painters
        headerBuffer = QPixmap(self.header.canvas().size())
        graphBuffer = QPixmap(QSize(self.graph.canvas().pright, self.graph.canvas().gbottom))
        footerBuffer = QPixmap(self.footer.canvas().size())
        
        headerPainter = QPainter(headerBuffer)
        graphPainter = QPainter(graphBuffer)
        footerPainter = QPainter(footerBuffer)

        # fill painters
        headerPainter.fillRect(headerBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        graphPainter.fillRect(graphBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        footerPainter.fillRect(footerBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        
        self.header.drawContents(headerPainter, 0, 0, sizeW, self.header.canvas().size().height())
        self.graph.drawContents(graphPainter, 0, 0, sizeW, self.graph.canvas().gbottom)
        self.footer.drawContents(footerPainter, 0, 0, sizeW, self.footer.canvas().size().height())

        

        buffer = QPixmap(size) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background

        bitBlt(buffer, 0, 0, headerBuffer, 0, 0,  sizeW, self.header.canvas().size().height(), Qt.CopyROP)
        bitBlt(buffer, 0, self.header.canvas().size().height(), graphBuffer, 0, 0,  sizeW, self.graph.canvas().gbottom, Qt.CopyROP)
        bitBlt(buffer, 0, self.header.canvas().size().height()+self.graph.canvas().gbottom, footerBuffer, 0, 0,  sizeW, self.footer.canvas().size().height(), Qt.CopyROP)

        painter.end()
        headerPainter.end()
        graphPainter.end()
        footerPainter.end()
        
        buffer.save(fileName, ext)

        
    # Callbacks
    def showNomogram(self):
        if self.bnomogram:
            self.bnomogram.show()


# test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNomogram()
    a.setMainWidget(ow)
    data = orange.ExampleTable("titanic")
    bayes = orange.BayesLearner(data)
    #logistic = orngLR.LogRegLearner(data)
    ow.classifier(bayes)
    ow.cdata(data)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()

    # save settings
    ow.saveSettings()

