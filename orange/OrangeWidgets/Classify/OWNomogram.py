"""
<name>Nomogram</name>
<description>Visualizes Naive Bayesian or logistic regression or any linear classifier using interactive nomogram.</description>
<icon>icons/Nomogram.png</icon>
<priority>2500</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with Naive Bayes or logistic regression classifier
#

import math
import orange
import OWGUI, orngLR_Jakulin
from OWWidget import *
from OWNomogramGraph import *



def getStartingPoint(d, min):
    if min<0:
        curr_num = arange(-min+d, step=d)
        curr_num = curr_num[len(curr_num)-1]
        curr_num = -curr_num
    elif min - d <= 0:
        curr_num = 0
    else:
        curr_num = arange(min-d, step=d)
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
    settingsList = ["alignType", "contType", "bubble", "histogram", "histogram_size", "confidence_percent", "sort_type"]

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
        self.showBaseLine = 1
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
        self.notTargetClassIndex = 1
        self.TargetClassIndex = 0
        self.sort_type = 0
        
        self.loadSettings()

        self.pointsName = ["Points","Log OR"]
        self.totalPointsName = ["Total Points","Log OR Sum"]
        self.bnomogram = None


        #inputs
        self.inputs=[("Classifier", orange.Classifier, self.classifier, 1), ("Examples", ExampleTable, self.cdata, 1), ("Target Class Value", int, self.ctarget, 1)]

        # GUI definition
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        
        # GENERAL TAB
        GeneralTab = QVGroupBox(self)

        self.alignRadio = OWGUI.radioButtonsInBox(GeneralTab, self,  'alignType', ['Left', '0-point'], box='Align',
                                                  tooltips=['Attributes in nomogram are left aligned', 'Attributes are not aligned, top scale represents true (normalized) regression coefficient value'],
                                                  callback=self.showNomogram)
        self.yAxisRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'yAxis', ['100', 'log OR'], 'yAxis',  
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

        self.sortBox = OWGUI.comboBox(GeneralTab, self, "sort_type", box="Sorting", label="Criteria: ", items=["No sorting", "Absolute importance", "Positive influence", "Negative influence"], callback = self.sortNomogram)
    
        self.tabs.insertTab(GeneralTab, "General")
        
        # TREE TAB
        NomogramStyleTab = QVGroupBox(self)

        self.verticalSpacingLabel = OWGUI.spin(NomogramStyleTab, self, 'verticalSpacing', 15, 100, box = 'Vertical spacing:',  tooltip='Define space (pixels) between adjacent attributes.', callback = self.showNomogram)
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
        self.showBaseLineCB = OWGUI.checkBox(NomogramStyleTab, self, 'showBaseLine', 'Show Base Line (at 0-point)', callback = self.setBaseLine)
        
        self.tabs.insertTab(NomogramStyleTab, "Settings")
        
        #add a graph widget
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
            var = max(1/inf - priorError*priorError, 0)
            return (math.sqrt(var))

        classVal = cl.domain.classVar
        att = cl.domain.attributes

        # calculate prior probability
        dist1 = max(0.00000001, cl.distribution[classVal[self.notTargetClassIndex]])
        dist0 = max(0.00000001, cl.distribution[classVal[self.TargetClassIndex]])
        prior = dist0/dist1
        if self.data:
            sumd = dist1+dist0
            priorError = math.sqrt(1/((dist1*dist0/sumd/sumd)*len(self.data)))
        else:
            priorError = 0
        self.bnomogram = BasicNomogram(self, AttValue("Constant", math.log(prior), error = priorError))

        if self.data:
            stat = orange.DomainBasicAttrStat(self.data)

        for at in range(len(att)):
            a = AttrLine(att[at].name, self.bnomogram)
            if att[at].varType == orange.VarTypes.Discrete:
                for cd in cl.conditionalDistributions[at].keys():
                    # calculuate thickness
                    conditional0 = max(cl.conditionalDistributions[at][cd][classVal[self.TargetClassIndex]], 0.00000001)
                    conditional1 = max(cl.conditionalDistributions[at][cd][classVal[self.notTargetClassIndex]], 0.00000001)
                    beta = math.log(conditional0/conditional1/prior)
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
                            conditional0 = max(cl.conditionalDistributions[at][d_filter][classVal[self.TargetClassIndex]], 0.00000001)
                            conditional1 = max(cl.conditionalDistributions[at][d_filter][classVal[self.notTargetClassIndex]], 0.00000001)
                            a.addAttValue(AttValue(str(round(curr_num+i*d,rndFac)), math.log(conditional0/conditional1/prior),lineWidth=thickness))
                        
                a.continuous = True
            self.bnomogram.addAttribute(a)        

        self.graph.setCanvas(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the logistic regression classifier    
    def lrClassifier(self, cl):
        if self.notTargetClassIndex == 1 or self.notTargetClassIndex == cl.domain.classVar[1]:
            mult = -1
        else:
            mult = 1
        
        self.bnomogram = BasicNomogram(self, AttValue('Constant', mult*cl.beta[0], error = 0))
        a = None

        # After applying feature subset selection on discrete attributes
        # aproximate unknown error for each attribute is math.sqrt(math.pow(cl.beta_se[0],2)/len(at))
        aprox_prior_error = math.sqrt(math.pow(cl.beta_se[0],2)/len(cl.domain.attributes))

        for at in cl.domain.attributes:
            at.setattr("visited",0)
            
        for at in cl.domain.attributes:
            if at.getValueFrom and at.visited==0:
                name = at.getValueFrom.variable.name
                var = at.getValueFrom.variable
                a = AttrLine(name, self.bnomogram)
                listOfExcludedValues = []
                for val in var.values:
                    foundValue = False
                    for same in cl.domain.attributes:
                        if same.visited==0 and same.getValueFrom and same.getValueFrom.variable == var and hasattr(same, "originValue") and same.originValue==val:
                            same.setattr("visited",1)
                            a.addAttValue(AttValue(same.originValue, mult*cl.beta[same], error = cl.beta_se[same]))
                            foundValue = True
                    if not foundValue:
                        listOfExcludedValues.append(val)
                if len(listOfExcludedValues) == 1:
                    a.addAttValue(AttValue(listOfExcludedValues[0], 0, error = aprox_prior_error))
                elif len(listOfExcludedValues) == 2:
                    a.addAttValue(AttValue("("+listOfExcludedValues[0]+","+listOfExcludedValues[1]+")", 0, error = aprox_prior_error))
                elif len(listOfExcludedValues) > 2:
                    a.addAttValue(AttValue("Other", 0, error = aprox_prior_error))
                self.bnomogram.addAttribute(a)    
                    
                
            elif at.visited==0:
                name = at.name
                var = at
                a = AttrLine(name, self.bnomogram)
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
                    if abs(mult*curr_num*cl.beta[at])<0.000001:
                        a.addAttValue(AttValue("0.0", 0))
                    else:
                        a.addAttValue(AttValue(str(curr_num), mult*curr_num*cl.beta[at]))
                    curr_num += d
                a.continuous = True
                at.setattr("visited", 1)
                self.bnomogram.addAttribute(a)



        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setCanvas(self.bnomogram)
        self.bnomogram.show()

    def svmClassifier(self, cl):
        import Numeric
        import orngLinVis
        
        if self.notTargetClassIndex == 1 or self.notTargetClassIndex == cl.domain.classVar[1]:
            mult = -1
        else:
            mult = 1

        visualizer = orngLinVis.Visualizer(self.data, cl, buckets=1, dimensions=1)
        self.bnomogram = BasicNomogram(self, AttValue('Constant', mult*visualizer.beta, 0))

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
                        a.addAttValue(AttValue(c[i], mult*visualizer.coeffs[coeff]))
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

                # transform SVM betas to betas siutable for nomogram
                beta = ((maxMap[coeff]-minMap[coeff])/(maxNew-minNew))*visualizer.coeffs[coeff]
                n = -minNew+minMap[coeff]
                
                numOfPartitions = 50
                d = getDiff((maxNew-minNew)/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minNew) 
                rndFac = getRounding(d)
                
                while curr_num<maxNew+d:
                    a.addAttValue(AttValue(str(curr_num), mult*(curr_num-minNew)*beta-minMap[coeff]*visualizer.coeffs[coeff]))
                    curr_num += d

                at_num = at_num + 1
                coeff = coeff + 1
                a.continuous = True
                
            self.bnomogram.addAttribute(a)
        self.cl.domain = orange.Domain(self.data.domain.classVar)
#        self.cl.domain.classVar = orange
#        self.cl.domain.classVar.values = self.data.domain.classVar.values       
        self.graph.setCanvas(self.bnomogram)
#        self.bnomogram.printOUT()
        self.bnomogram.show()
       
    def classifier(self, cl):
        self.cl = cl
        self.updateNomogram()
        
    # Input channel: data
    def cdata(self, data):
        # call appropriate classifier
        if data and data.domain and not data.domain.classVar:
            QMessageBox("OWNomogram:", " This domain has no class attribute!", QMessageBox.Warning,
                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            return
        
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

    def ctarget(self, target):
        self.TargetClassIndex = target
        if self.TargetClassIndex == 1:
            self.notTargetClassIndex = 0
        else:
            self.notTargetClassIndex = 1
        if self.cl and self.cl.domain and self.TargetClassIndex == self.cl.domain.classVar[1]:
            self.notTargetClassIndex = self.cl.domain.classVar[0]
        elif self.cl and self.cl.domain:
            self.notTargetClassIndex = self.cl.domain.classVar[1]
        self.updateNomogram()
            

    def updateNomogram(self):
        import orngSVM

        def setNone():
            self.footer.setCanvas(None)
            self.header.setCanvas(None)
            self.graph.setCanvas(None)

        if self.data and self.cl: # and not type(self.cl) == orngLR_Jakulin.MarginMetaClassifier:
            #check domains
            for at in self.cl.domain:
                if at.getValueFrom and ('variable' in dir(at.getValueFrom)):
                    if not at.getValueFrom.variable in self.data.domain:
                        return
                else:
                    if not at in self.data.domain:
                        return

        if type(self.cl) == orange.BayesClassifier:
            if len(self.cl.domain.classVar.values)>2:
                QMessageBox("OWNomogram:", " Please use only Bayes classifiers that are induced on data with dichotomous class!", QMessageBox.Warning,
                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            else:
                self.nbClassifier(self.cl)
        elif type(self.cl) == orngLR_Jakulin.MarginMetaClassifier and self.data:
                self.svmClassifier(self.cl)

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
        if self.sort_type>0:
            self.sortNomogram()

    def sortNomogram(self):
        def sign(x):
            if x<0:
                return -1;
            return 1;
        def compate_beta_difference(x,y):
            return -sign(x.maxValue-x.minValue-y.maxValue+y.minValue)
        def compare_beta_positive(x, y):
            return -sign(x.maxValue-y.maxValue)
        def compare_beta_negative(x, y):
            return sign(x.minValue-y.minValue)

        if self.sort_type == 1:
            self.bnomogram.attributes.sort(compate_beta_difference)               
        elif self.sort_type == 2:
            self.bnomogram.attributes.sort(compare_beta_positive)               
        elif self.sort_type == 3:
            self.bnomogram.attributes.sort(compare_beta_negative)               

        # update nomogram
        self.showNomogram()
        
        
    def setProbability(self):
        if self.probability and self.bnomogram:
            self.bnomogram.showAllMarkers()
        elif self.bnomogram:
            self.bnomogram.hideAllMarkers()

    def setBaseLine(self):
        if self.bnomogram:
            self.bnomogram.showBaseLine(self.showBaseLine)

    def saveToFileCanvas(self):
        EMPTY_SPACE = 25 # Empty space between nomogram and summarization scale
        
        sizeW = self.graph.canvas().pright
        sizeH = self.graph.canvas().gbottom + self.header.canvas().size().height() + self.footer.canvas().size().height()+EMPTY_SPACE
        size = QSize(sizeW, sizeH)

        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()

        # create buffers and painters
        headerBuffer = QPixmap(self.header.canvas().size())
        graphBuffer = QPixmap(QSize(self.graph.canvas().pright, self.graph.canvas().gbottom+EMPTY_SPACE))
        footerBuffer = QPixmap(self.footer.canvas().size())
        
        headerPainter = QPainter(headerBuffer)
        graphPainter = QPainter(graphBuffer)
        footerPainter = QPainter(footerBuffer)

        # fill painters
        headerPainter.fillRect(headerBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        graphPainter.fillRect(graphBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        footerPainter.fillRect(footerBuffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        
        self.header.drawContents(headerPainter, 0, 0, sizeW, self.header.canvas().size().height())
        self.graph.drawContents(graphPainter, 0, 0, sizeW, self.graph.canvas().gbottom+EMPTY_SPACE)
        self.footer.drawContents(footerPainter, 0, 0, sizeW, self.footer.canvas().size().height())

        

        buffer = QPixmap(size) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background

        bitBlt(buffer, 0, 0, headerBuffer, 0, 0,  sizeW, self.header.canvas().size().height(), Qt.CopyROP)
        bitBlt(buffer, 0, self.header.canvas().size().height(), graphBuffer, 0, 0,  sizeW, self.graph.canvas().gbottom+EMPTY_SPACE, Qt.CopyROP)
        bitBlt(buffer, 0, self.header.canvas().size().height()+self.graph.canvas().gbottom+EMPTY_SPACE, footerBuffer, 0, 0,  sizeW, self.footer.canvas().size().height(), Qt.CopyROP)

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
    import orngLR, orngSVM
    
    a=QApplication(sys.argv)
    ow=OWNomogram()
    a.setMainWidget(ow)
    data = orange.ExampleTable("d:\\delo\\data\\ionosphere")
    bayes = orange.BayesLearner(data)
    #l = orngSVM.BasicSVMLearner()
    #l.kernel = 0 # linear SVM
    #svm = orngLR_Jakulin.MarginMetaLearner(l,folds = 1)(data)
   # logistic = orngLR.LogRegLearner(data, removeSingular = 1)
    ow.classifier(bayes)
    ow.cdata(data)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()

    # save settings
    ow.saveSettings()

