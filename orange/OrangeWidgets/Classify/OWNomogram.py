"""
<name>Nomogram</name>
<description>Nomogram viewer for Naive Bayesian, logistic regression or linear SVM classifiers.</description>
<icon>icons/Nomogram.png</icon>
<contact>Martin Mozina (martin.mozina(@at@)fri.uni-lj.si)</contact>
<priority>2500</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with Naive Bayes or logistic regression classifier
#
import math
import orange
import OWGUI
from OWWidget import *
from OWNomogramGraph import *
from orngDataCaching import *
import orngLR


aproxZero = 0.0001

def getStartingPoint(d, min):
    if d == 0:
        return min
    elif min<0:
        curr_num = numpy.arange(-min+d, step=d)
        curr_num = curr_num[len(curr_num)-1]
        curr_num = -curr_num
    elif min - d <= 0:
        curr_num = 0
    else:
        curr_num = numpy.arange(min-d, step=d)
        curr_num = curr_num[len(curr_num)-1]
    return curr_num

def getRounding(d):
    if d == 0:
        return 2
    rndFac = math.floor(math.log10(d));
    if rndFac<-2:
        rndFac = int(-rndFac)
    else:
        rndFac = 2
    return rndFac

def avg(l):
    return sum(l)/len(l)


class OWNomogram(OWWidget):
    settingsList = ["alignType", "verticalSpacing", "contType", "verticalSpacingContinuous", "yAxis", "probability", "confidence_check", "confidence_percent", "histogram", "histogram_size", "sort_type"]
    contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"], matchValues=1)}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Nomogram", 1)

        #self.setWFlags(Qt.WResizeNoErase | Qt.WRepaintNoErase) #this works like magic.. no flicker during repaint!
        self.parent = parent
#        self.setWFlags(self.getWFlags()+Qt.WStyle_Maximize)

        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.alignType = 0
        self.contType = 0
        self.yAxis = 0
        self.probability = 0
        self.verticalSpacing = 60
        self.verticalSpacingContinuous = 100
        self.diff_between_ordinal = 30
        self.fontSize = 9
        self.lineWidth = 1
        self.histogram = 0
        self.histogram_size = 10
        self.data = None
        self.cl = None
        self.confidence_check = 0
        self.confidence_percent = 95
        self.sort_type = 0

        self.loadSettings()

        self.pointsName = ["Total", "Total"]
        self.totalPointsName = ["Probability", "Probability"]
        self.bnomogram = None


        self.inputs=[("Classifier", orange.Classifier, self.classifier)]


        self.TargetClassIndex = 0
        self.targetCombo = OWGUI.comboBox(self.controlArea, self, "TargetClassIndex", " Target Class ", addSpace=True, tooltip='Select target (prediction) class in the model.', callback = self.setTarget)

        self.alignRadio = OWGUI.radioButtonsInBox(self.controlArea, self,  'alignType', ['Align left', 'Align by zero influence'], box='Attribute placement',
                                                  tooltips=['Attributes in nomogram are left aligned', 'Attributes are not aligned, top scale represents true (normalized) regression coefficient value'],
                                                  addSpace=True,
                                                  callback=self.showNomogram)
        self.verticalSpacingLabel = OWGUI.spin(self.alignRadio, self, 'verticalSpacing', 15, 200, label = 'Vertical spacing:',  orientation = 0, tooltip='Define space (pixels) between adjacent attributes.', callback = self.showNomogram)

        self.ContRadio = OWGUI.radioButtonsInBox(self.controlArea, self, 'contType',   ['1D projection', '2D curve'], 'Continuous attributes',
                                tooltips=['Continuous attribute are presented on a single scale', 'Two dimensional space is used to present continuous attributes in nomogram.'],
                                addSpace=True,
                                callback=[lambda:self.verticalSpacingContLabel.setDisabled(not self.contType), self.showNomogram])

        self.verticalSpacingContLabel = OWGUI.spin(OWGUI.indentedBox(self.ContRadio), self, 'verticalSpacingContinuous', 15, 200, label = "Height", orientation=0, tooltip='Define space (pixels) between adjacent 2d presentation of attributes.', callback = self.showNomogram)
        self.verticalSpacingContLabel.setDisabled(not self.contType)

        self.yAxisRadio = OWGUI.radioButtonsInBox(self.controlArea, self, 'yAxis', ['Point scale', 'Log odds ratios'], 'Scale',
                                tooltips=['values are normalized on a 0-100 point scale','values on top axis show log-linear contribution of attribute to full model'],
                                addSpace=True,
                                callback=self.showNomogram)

        layoutBox = OWGUI.widgetBox(self.controlArea, "Display", orientation=1, addSpace=True)

        self.probabilityCheck = OWGUI.checkBox(layoutBox, self, 'probability', 'Show prediction',  tooltip='', callback = self.setProbability)

        self.CICheck, self.CILabel = OWGUI.checkWithSpin(layoutBox, self, 'Confidence intervals (%):', min=1, max=99, step = 1, checked='confidence_check', value='confidence_percent', checkCallback=self.showNomogram, spinCallback = self.showNomogram)

        self.histogramCheck, self.histogramLabel = OWGUI.checkWithSpin(layoutBox, self, 'Show histogram, size', min=1, max=30, checked='histogram', value='histogram_size', step = 1, tooltip='-(TODO)-', checkCallback=self.showNomogram, spinCallback = self.showNomogram)

        OWGUI.separator(layoutBox)
        self.sortOptions = ["No sorting", "Absolute importance", "Positive influence", "Negative influence"]
        self.sortBox = OWGUI.comboBox(layoutBox, self, "sort_type", label="Sort by ", items=self.sortOptions, callback = self.sortNomogram, orientation="horizontal")


        OWGUI.rubber(self.controlArea)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.menuItemPrinter)



        #add a graph widget
        self.header = OWNomogramHeader(None, self.mainArea)
        self.header.setFixedHeight(self.verticalSpacing)
        self.header.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graph = OWNomogramGraph(self.bnomogram, self.mainArea)
        self.graph.setMinimumWidth(200)
        self.graph.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.footer = OWNomogramHeader(None, self.mainArea)
        self.footer.setFixedHeight(self.verticalSpacing*2+10)
        self.footer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.footer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.mainArea.layout().addWidget(self.header)
        self.mainArea.layout().addWidget(self.graph)
        self.mainArea.layout().addWidget(self.footer)
        self.resize(700,500)
        #self.repaint()
        #self.update()

        # mouse pressed flag
        self.mousepr = False

    def sendReport(self):
        self.reportSettings("Information",
                            [("Target class", self.cl.domain.classVar.values[self.TargetClassIndex]),
                             self.confidence_check and ("Confidence intervals", "%i%%" % self.confidence_percent),
                             ("Sorting", self.sortOptions[self.sort_type] if self.sort_type else "None")])
        
        canvases = header, graph, footer = self.header.scene(), self.graph.scene(), self.footer.scene()
        painter = QPainter()
        buffer = QPixmap(max(c.width() for c in canvases), sum(c.height() for c in canvases))
        painter.begin(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        header.render(painter, QRectF(0, 0, header.width(), header.height()), QRectF(0, 0, header.width(), header.height()))
        graph.render(painter, QRectF(0, header.height(), graph.width(), graph.height()), QRectF(0, 0, graph.width(), graph.height()))
        footer.render(painter, QRectF(0, header.height()+graph.height(), footer.width(), footer.height()), QRectF(0, 0, footer.width(), footer.height()))
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))

        
    # Input channel: the Bayesian classifier
    def nbClassifier(self, cl):
        # this subroutine computes standard error of estimated beta. Note that it is used only for discrete data,
        # continuous data have a different computation.
        def errOld(e, priorError, key, data):
            inf = 0.0
            sume = e[0]+e[1]
            for d in data:
                if d[at]==key:
                    inf += (e[0]*e[1]/sume/sume)
            inf = max(inf, aproxZero)
            var = max(1/inf - priorError*priorError, 0)
            return (math.sqrt(var))

        def err(condDist, att, value, targetClass, priorError, data):
            sumE = sum(condDist)
            valueE = condDist[targetClass]
            distAtt = orange.Distribution(att, data)
            inf = distAtt[value]*(valueE/sumE)*(1-valueE/sumE)
            inf = max(inf, aproxZero)
            var = max(1/inf - priorError*priorError, 0)
            return (math.sqrt(var))

        classVal = cl.domain.classVar
        att = cl.domain.attributes

        # calculate prior probability
        dist1 = max(aproxZero, 1-cl.distribution[classVal[self.TargetClassIndex]])
        dist0 = max(aproxZero, cl.distribution[classVal[self.TargetClassIndex]])
        prior = dist0/dist1
        if self.data:
            sumd = dist1+dist0
            priorError = math.sqrt(1/((dist1*dist0/sumd/sumd)*len(self.data)))
        else:
            priorError = 0

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue("Constant", math.log(prior), error = priorError))
        else:
            self.bnomogram = BasicNomogram(self, AttValue("Constant", math.log(prior), error = priorError))

        if self.data:
            stat = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))

        for at in range(len(att)):
            a = None
            if att[at].varType == orange.VarTypes.Discrete:
                if att[at].ordered:
                    a = AttrLineOrdered(att[at].name, self.bnomogram)
                else:
                    a = AttrLine(att[at].name, self.bnomogram)
                for cd in cl.conditionalDistributions[at].keys():
                    # calculuate thickness
                    conditional0 = max(cl.conditionalDistributions[at][cd][classVal[self.TargetClassIndex]], aproxZero)
                    conditional1 = max(1-cl.conditionalDistributions[at][cd][classVal[self.TargetClassIndex]], aproxZero)
                    beta = math.log(conditional0/conditional1/prior)
                    if self.data:
                        #thickness = int(round(4.*float(len(self.data.filter({att[at].name:str(cd)})))/float(len(self.data))))
                        thickness = float(len(self.data.filter({att[at].name:str(cd)})))/float(len(self.data))
                        se = err(cl.conditionalDistributions[at][cd], att[at], cd, classVal[self.TargetClassIndex], priorError, self.data) # standar error of beta
                    else:
                        thickness = 0
                        se = 0

                    a.addAttValue(AttValue(str(cd), beta, lineWidth=thickness, error = se))

            else:
                a = AttrLineCont(att[at].name, self.bnomogram)
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

                values = []
                for i in range(2*numOfPartitions):
                    if curr_num+i*d>=minAtValue and curr_num+i*d<=maxAtValue:
                        # get thickness
                        if self.data:
                            thickness = float(len(self.data.filter({att[at].name:(curr_num+i*d-d/2, curr_num+i*d+d/2)})))/len(self.data)
                        else:
                            thickness = 0.0
                        d_filter = filter(lambda x: x>curr_num+i*d-d/2 and x<curr_num+i*d+d/2, cl.conditionalDistributions[at].keys())
                        if len(d_filter)>0:
                            cd = cl.conditionalDistributions[at]
                            conditional0 = avg([cd[f][classVal[self.TargetClassIndex]] for f in d_filter])
                            conditional0 = min(1-aproxZero,max(aproxZero,conditional0))
                            conditional1 = 1-conditional0
                            try:
                                # compute error of loess in logistic space
                                var = avg([cd[f].variances[self.TargetClassIndex] for f in d_filter])
                                standard_error= math.sqrt(var)
                                rightError0 = (conditional0+standard_error)/max(conditional1-standard_error, aproxZero)
                                leftError0  =  max(conditional0-standard_error, aproxZero)/(conditional1+standard_error)
                                se = (math.log(rightError0) - math.log(leftError0))/2
                                se = math.sqrt(math.pow(se,2)+math.pow(priorError,2))

                                # add value to set of values
                                a.addAttValue(AttValue(str(round(curr_num+i*d,rndFac)),
                                                       math.log(conditional0/conditional1/prior),
                                                       lineWidth=thickness,
                                                       error = se))
                            except:
                                pass
                a.continuous = True
                # invert values:
            # if there are more than 1 value in the attribute, add it to the nomogram
            if a and len(a.attValues)>1:
                self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(False)
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the logistic regression classifier
    def lrClassifier(self, cl):
        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = -1
        else:
            mult = 1

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue('Constant', mult*cl.beta[0], error = 0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue('Constant', mult*cl.beta[0], error = 0))

        # After applying feature subset selection on discrete attributes
        # aproximate unknown error for each attribute is math.sqrt(math.pow(cl.beta_se[0],2)/len(at))
        try:
            aprox_prior_error = math.sqrt(math.pow(cl.beta_se[0],2)/len(cl.domain.attributes))
        except:
            aprox_prior_error = 0

        domain = cl.continuizedDomain or cl.domain
        if domain:
            for at in domain.attributes:
                at.setattr("visited",0)

            for at in domain.attributes:
                if at.getValueFrom and at.visited==0:
                    name = at.getValueFrom.variable.name
                    var = at.getValueFrom.variable
                    if var.ordered:
                        a = AttrLineOrdered(name, self.bnomogram)
                    else:
                        a = AttrLine(name, self.bnomogram)
                    listOfExcludedValues = []
                    for val in var.values:
                        foundValue = False
                        for same in domain.attributes:
                            if same.visited==0 and same.getValueFrom and same.getValueFrom.variable == var and same.getValueFrom.variable.values[same.getValueFrom.transformer.value]==val:
                                same.setattr("visited",1)
                                a.addAttValue(AttValue(val, mult*cl.beta[same], error = cl.beta_se[same]))
                                foundValue = True
                        if not foundValue:
                            listOfExcludedValues.append(val)
                    if len(listOfExcludedValues) == 1:
                        a.addAttValue(AttValue(listOfExcludedValues[0], 0, error = aprox_prior_error))
                    elif len(listOfExcludedValues) == 2:
                        a.addAttValue(AttValue("("+listOfExcludedValues[0]+","+listOfExcludedValues[1]+")", 0, error = aprox_prior_error))
                    elif len(listOfExcludedValues) > 2:
                        a.addAttValue(AttValue("Other", 0, error = aprox_prior_error))
                    # if there are more than 1 value in the attribute, add it to the nomogram
                    if len(a.attValues)>1:
                        self.bnomogram.addAttribute(a)


                elif at.visited==0:
                    name = at.name
                    var = at
                    a = AttrLineCont(name, self.bnomogram)
                    if self.data:
                        bas = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))
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
                        if abs(mult*curr_num*cl.beta[at])<aproxZero:
                            a.addAttValue(AttValue("0.0", 0))
                        else:
                            a.addAttValue(AttValue(str(curr_num), mult*curr_num*cl.beta[at]))
                        curr_num += d
                    a.continuous = True
                    at.setattr("visited", 1)
                    # if there are more than 1 value in the attribute, add it to the nomogram
                    if len(a.attValues)>1:
                        self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    def svmClassifier(self, cl):
        import orngLR_Jakulin

        import orngLinVis

        self.error(0)
        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = -1
        else:
            mult = 1

        try:
            visualizer = orngLinVis.Visualizer(self.data, cl, buckets=1, dimensions=1)
            beta_from_cl = self.cl.estimator.classifier.classifier.beta[0] - self.cl.estimator.translator.trans[0].disp*self.cl.estimator.translator.trans[0].mult*self.cl.estimator.classifier.classifier.beta[1]
            beta_from_cl = mult*beta_from_cl
        except:
            self.error(0, "orngLinVis.Visualizer error"+ str(sys.exc_info()[0])+":"+str(sys.exc_info()[1]))
#            QMessageBox("orngLinVis.Visualizer error", str(sys.exc_info()[0])+":"+str(sys.exc_info()[1]), QMessageBox.Warning,
#                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            return

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue('Constant', -mult*math.log((1.0/min(max(visualizer.probfunc(0.0),aproxZero),0.9999))-1), 0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue('Constant', -mult*math.log((1.0/min(max(visualizer.probfunc(0.0),aproxZero),0.9999))-1), 0))

        # get maximum and minimum values in visualizer.m
        maxMap = reduce(numpy.maximum, visualizer.m)
        minMap = reduce(numpy.minimum, visualizer.m)

        coeff = 0 #
        at_num = 1
        correction = self.cl.coeff*self.cl.estimator.translator.trans[0].mult*self.cl.estimator.classifier.classifier.beta[1]
        for c in visualizer.coeff_names:
            if type(c[1])==str:
                for i in range(len(c)):
                    if i == 0:
                        if self.data.domain[c[0]].ordered:
                            a = AttrLineOrdered(c[i], self.bnomogram)
                        else:
                            a = AttrLine(c[i], self.bnomogram)
                        at_num = at_num + 1
                    else:
                        if self.data:
                            thickness = float(len(self.data.filter({self.data.domain[c[0]].name:str(c[i])})))/float(len(self.data))
                        a.addAttValue(AttValue(c[i], correction*mult*visualizer.coeffs[coeff], lineWidth=thickness))
                        coeff = coeff + 1
            else:
                a = AttrLineCont(c[0], self.bnomogram)

                # get min and max from Data and transform coeff accordingly
                maxNew=maxMap[coeff]
                minNew=maxMap[coeff]
                if self.data:
                    bas = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))
                    maxNew = bas[c[0]].max
                    minNew = bas[c[0]].min

                # transform SVM betas to betas siutable for nomogram
                if maxNew == minNew:
                    beta = ((maxMap[coeff]-minMap[coeff])/aproxZero)*visualizer.coeffs[coeff]
                else:
                    beta = ((maxMap[coeff]-minMap[coeff])/(maxNew-minNew))*visualizer.coeffs[coeff]
                n = -minNew+minMap[coeff]

                numOfPartitions = 50
                d = getDiff((maxNew-minNew)/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minNew)
                rndFac = getRounding(d)

                while curr_num<maxNew+d:
                    a.addAttValue(AttValue(str(curr_num), correction*(mult*(curr_num-minNew)*beta-minMap[coeff]*visualizer.coeffs[coeff])))
                    curr_num += d

                at_num = at_num + 1
                coeff = coeff + 1
                a.continuous = True

            # if there are more than 1 value in the attribute, add it to the nomogram
            if len(a.attValues)>1:
                self.bnomogram.addAttribute(a)
        self.cl.domain = orange.Domain(self.data.domain.classVar)
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the rule classifier (from CN2-EVC only)
    def ruleClassifier(self, cl):
        def selectSign(oper):
            if oper == orange.ValueFilter_continuous.Less:
                return "<"
            elif oper == orange.ValueFilter_continuous.LessEqual:
                return "<="
            elif oper == orange.ValueFilter_continuous.Greater:
                return ">"
            elif oper == orange.ValueFilter_continuous.GreaterEqual:
                return ">="
            else: return "="

        def getConditions(rule):
            conds = rule.filter.conditions
            domain = rule.filter.domain
            ret = []
            if len(conds)==0:
                ret = ret + ["TRUE"]
            for i,c in enumerate(conds):
                if i > 0:
                    ret[-1] += " & "
                if type(c) == orange.ValueFilter_discrete:
                    ret += [domain[c.position].name + "=" + str(domain[c.position].values[int(c.values[0])])]
                elif type(c) == orange.ValueFilter_continuous:
                    ret += [domain[c.position].name + selectSign(c.oper) + "%.3f"%c.ref]
            return ret

        self.error(1)
        if not len(self.data.domain.classVar.values) == 2:
            self.error(1, "Rules require binary classes")
        classVal = cl.domain.classVar
        att = cl.domain.attributes

        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = 1.
        else:
            mult = -1.

        # calculate prior probability (from self.TargetClassIndex)
        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue("Constant", 0.0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue("Constant", 0.0))
        self.cl.setattr("rulesOrdering", [])
        for r_i,r in enumerate(cl.rules):
            a = AttrLine(getConditions(r), self.bnomogram)
            self.cl.rulesOrdering.append(getConditions(r))
            if r.classifier.defaultVal == 0:
                sign = mult
            else: sign = -mult
            a.addAttValue(AttValue("yes", sign*cl.ruleBetas[r_i], lineWidth=0, error = 0.0))
            a.addAttValue(AttValue("no", 0.0, lineWidth=0, error = 0.0))
            self.bnomogram.addAttribute(a)

        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()


    def initClassValues(self, classValue):
        self.targetCombo.clear()
        self.targetCombo.addItems([str(v) for v in classValue])

    def classifier(self, cl):
        self.closeContext()
        self.error(2) 

        oldcl = self.cl
        self.cl = None
        
        if cl:
            for acceptable in (orange.BayesClassifier, orange.LogRegClassifier, orange.RuleClassifier_logit):
                if isinstance(cl, acceptable):
                    self.cl = cl
                    break
            else:
                self.error(2, "Nomograms can be drawn for only for Bayesian classifier and logistic regression")
                 
        if not oldcl or not self.cl or not oldcl.domain == self.cl.domain:
            if self.cl:
                self.initClassValues(self.cl.domain.classVar)
            self.TargetClassIndex = 0
            
        self.data = getattr(self.cl, "data", None)

        if self.data and self.data.domain and not self.data.domain.classVar:
            self.error(2, "Classless domain")
            # Here it said "return", but let us report an error and clean up the widget
            self.cl = self.data = None

        self.openContext("", self.data)
        if not self.data:
            self.histogramCheck.setChecked(False)
            self.histogramCheck.setDisabled(True)
            self.histogramLabel.setDisabled(True)
            self.CICheck.setChecked(False)
            self.CICheck.setDisabled(True)
            self.CILabel.setDisabled(True)
        else:
            self.histogramCheck.setEnabled(True)
            self.histogramCheck.makeConsistent()
            self.CICheck.setEnabled(True)
            self.CICheck.makeConsistent()
        self.updateNomogram()

    def setTarget(self):
        self.updateNomogram()

    def updateNomogram(self):
##        import orngSVM

        def setNone():
            for view in [self.footer, self.header, self.graph]:
                scene = view.scene()
                if scene:
                    for item in scene.items():
                        scene.removeItem(item)

        if self.data and self.cl: # and not type(self.cl) == orngLR_Jakulin.MarginMetaClassifier:
            #check domains
            for at in self.cl.domain:
                if at.getValueFrom and hasattr(at.getValueFrom, "variable"):
                    if (not at.getValueFrom.variable in self.data.domain) and (not at in self.data.domain):
                        return
                else:
                    if not at in self.data.domain:
                        return

        if isinstance(self.cl, orange.BayesClassifier):
#            if len(self.cl.domain.classVar.values)>2:
#                QMessageBox("OWNomogram:", " Please use only Bayes classifiers that are induced on data with dichotomous class!", QMessageBox.Warning,
#                            QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
#            else:
                self.nbClassifier(self.cl)
##        elif isinstance(self.cl, orngLR_Jakulin.MarginMetaClassifier) and self.data:
##            self.svmClassifier(self.cl)
        elif isinstance(self.cl, orange.RuleClassifier_logit):
            self.ruleClassifier(self.cl)

        elif isinstance(self.cl, orange.LogRegClassifier):
            # get if there are any continuous attributes in data -> then we need data to compute margins
            cont = False
            if self.cl.continuizedDomain:
                for at in self.cl.continuizedDomain.attributes:
                    if not at.getValueFrom:
                        cont = True
            if self.data or not cont:
                self.lrClassifier(self.cl)
            else:
                setNone()
        else:
            setNone()
        if self.sort_type>0:
            self.sortNomogram()

    def sortNomogram(self):
        def sign(x):
            if x<0:
                return -1;
            return 1;
        def compare_to_ordering_in_rules(x,y):
            return self.cl.rulesOrdering.index(x.name) - self.cl.rulesOrdering.index(y.name)
        def compare_to_ordering_in_data(x,y):
            return self.data.domain.attributes.index(self.data.domain[x.name]) - self.data.domain.attributes.index(self.data.domain[y.name])
        def compare_to_ordering_in_domain(x,y):
            return self.cl.domain.attributes.index(self.cl.domain[x.name]) - self.cl.domain.attributes.index(self.cl.domain[y.name])
        def compate_beta_difference(x,y):
            return -sign(x.maxValue-x.minValue-y.maxValue+y.minValue)
        def compare_beta_positive(x, y):
            return -sign(x.maxValue-y.maxValue)
        def compare_beta_negative(x, y):
            return sign(x.minValue-y.minValue)

        if not self.bnomogram:
            return
        if self.sort_type == 0 and hasattr(self.cl, "rulesOrdering"):
            self.bnomogram.attributes.sort(compare_to_ordering_in_rules)
        elif self.sort_type == 0 and self.data:
            self.bnomogram.attributes.sort(compare_to_ordering_in_data)
        elif self.sort_type == 0 and self.cl and self.cl.domain:
            self.bnomogram.attributes.sort(compare_to_ordering_in_domain)
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
            self.bnomogram.showBaseLine(True)

    def menuItemPrinter(self):
        import copy
        canvases = header, graph, footer = self.header.scene(), self.graph.scene(), self.footer.scene()
        # all scenes together
        scene_confed = QGraphicsScene(0, 0, max(c.width() for c in canvases), sum(c.height() for c in canvases))
        # add items from header
        header_its = header.items()
        for it in header_its:
            scene_confed.addItem(it)
        # add items from graph
        graph_its = graph.items()
        for it in graph_its:
            scene_confed.addItem(it)
            it.moveBy(0., header.height())
        # add from footer
        footer_its = footer.items()
        for it in footer_its:
            scene_confed.addItem(it)
            it.moveBy(0.,header.height() + graph.height())
        try:
            import OWDlgs
        except:
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(scene_confed)
        sizeDlg.exec_()

        # set all items back to original canvases            
        for it in header_its:
            header.addItem(it)
        for it in graph_its:
            graph.addItem(it)
            it.moveBy(0., -header.height())
        for it in footer_its:
            footer.addItem(it)
            it.moveBy(0, - header.height() - graph.height())
        self.showNomogram()

    # Callbacks
    def showNomogram(self):
        if self.bnomogram and self.cl:
            #self.bnomogram.hide()
            self.bnomogram.show()
            self.bnomogram.update()


# test widget appearance
if __name__=="__main__":
    import orngLR, orngSVM

    a=QApplication(sys.argv)
    ow=OWNomogram()
    a.setMainWidget(ow)
    data = orange.ExampleTable("../../doc/datasets/heart_disease.tab")

    bayes = orange.BayesLearner(data)
    bayes.setattr("data",data)
    ow.classifier(bayes)

    # here you can test setting some stuff
    
    a.exec_()

    # save settings
    ow.saveSettings()

