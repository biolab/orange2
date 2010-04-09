"""
<name>Predictions</name>
<description>Displays predictions of models for a particular data set.</description>
<icon>icons/Predictions.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>300</priority>
"""

from OWWidget import *
import OWGUI
import statc

from OWDataTable import ExampleTableModel

class PredictionTableModel(QAbstractItemModel):
    def __init__(self, prediction_results, *args, **kwargs):
        QAbstractItemModel.__init__(self, *args, **kwargs)
        self.prediction_results = prediction_results
        
    def data(self, index, role):
        col, row = index.column(), index.row()
        
            
        

##############################################################################

class colorItem(QTableWidgetItem):
    brush = QBrush(Qt.lightGray)
    def __init__(self, text, type=0):
        QTableWidgetItem.__init__(self, str(text), type)
        self.setBackground(self.brush)

##    def paint(self, painter, colorgroup, rect, selected):
##        g = QPalette(colorgroup)
##        g.setColor(QPalette.Base, Qt.lightGray)
##        QTableWidgetItem.paint(self, painter, g, rect, selected)

##############################################################################

class OWPredictions(OWWidget):
    settingsList = ["showProb", "showClass", "ShowAttributeMethod", "sendOnChange", "precision"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Predictions")

        self.callbackDeposit = []
        self.inputs = [("Examples", ExampleTable, self.setData), ("Predictors", orange.Classifier, self.setPredictor, Multiple)]
        self.outputs = [("Predictions", ExampleTable)]
        self.predictors = {}

        # saveble settings
        self.showProb = 1;
        self.showClass = 1
        self.ShowAttributeMethod = 0
        self.sendOnChange = 1
        self.classes = []
        self.selectedClasses = []
        self.loadSettings()
        self.datalabel = "N/A"
        self.predictorlabel = "N/A"
        self.tasklabel = "N/A"
        self.precision = 2
        self.outvar = None # current output variable (set by the first predictor/data set send in)

        self.data = None

        # GUI - Options

        # Options - classification
        ibox = OWGUI.widgetBox(self.controlArea, "Info")
        OWGUI.label(ibox, self, "Data: %(datalabel)s")
        OWGUI.label(ibox, self, "Predictors: %(predictorlabel)s")
        OWGUI.label(ibox, self, "Task: %(tasklabel)s")
        OWGUI.separator(self.controlArea)
        
        self.copt = OWGUI.widgetBox(self.controlArea, "Options (classification)")
        self.copt.setDisabled(1)
        OWGUI.checkBox(self.copt, self, 'showProb', "Show predicted probabilities", callback=self.updateTableOutcomes)

#        self.lbClasses = OWGUI.listBox(self.copt, self, selectionMode = QListWidget.MultiSelection, callback = self.updateTableOutcomes)

        self.lbcls = OWGUI.listBox(self.copt, self, "selectedClasses", "classes",
                                   callback=[self.updateTableOutcomes, self.checksendpredictions],
                                   selectionMode=QListWidget.MultiSelection)
        self.lbcls.setFixedHeight(50)

        OWGUI.spin(self.copt, self, "precision", 1, 6, label="No. of decimals: ",
                   orientation=0, callback=self.updateTableOutcomes)

        OWGUI.checkBox(self.copt, self, 'showClass', "Show predicted class",
                       callback=[self.updateTableOutcomes, self.checksendpredictions])

        # Options - regression
        # self.ropt = QVButtonGroup("Options (regression)", self.controlArea)
        # OWGUI.checkBox(self.ropt, self, 'showClass', "Show predicted class",
        #                callback=[self.updateTableOutcomes, self.checksendpredictions])
        # self.ropt.hide()

        OWGUI.separator(self.controlArea)

        self.att = OWGUI.widgetBox(self.controlArea, "Data attributes")
        OWGUI.radioButtonsInBox(self.att, self, 'ShowAttributeMethod', ['Show all', 'Hide all'], callback=self.updateAttributes)
        self.att.setDisabled(1)
        OWGUI.rubber(self.controlArea)

        OWGUI.separator(self.controlArea)
        self.outbox = OWGUI.widgetBox(self.controlArea, "Output")
        
        self.commitBtn = OWGUI.button(self.outbox, self, "Send Predictions", callback=self.sendpredictions)
        OWGUI.checkBox(self.outbox, self, 'sendOnChange', 'Send automatically')

        self.outbox.setDisabled(1)

        # GUI - Table
        self.table = OWGUI.table(self.mainArea, selectionMode = QTableWidget.NoSelection)

        self.table.setItemDelegate(OWGUI.TableBarItem(self))
        self.table.verticalHeader().setDefaultSectionSize(22)
        
        self.header = self.table.horizontalHeader()
        self.vheader = self.table.verticalHeader()
        # manage sorting (not correct, does not handle real values)
        self.connect(self.header, SIGNAL("sectionPressed(int)"), self.sort)
        self.sortby = -1
        self.resize(800, 600)


    ##############################################################################
    # Contents painting

    def updateTableOutcomes(self):
        """updates the columns associated with the classifiers"""
        if not self.data or not self.predictors or not self.outvar:
            return

        classification = self.outvar.varType == orange.VarTypes.Discrete

        # sindx is the column where these start
        sindx = len(self.data.domain.variables)
        col = sindx
        showprob = self.showProb and len(self.selectedClasses)
        fmt = "%%1.%df" % self.precision
        if self.showClass or (classification and showprob):
            for (cid, c) in enumerate(self.predictors.values()):
                if classification:
                    for (i, d) in enumerate(self.data):
                        (cl, p) = c(d, orange.GetBoth)

                        self.classifications[i].append(cl)
                        s = ""
                        if showprob:
                            s = " : ".join([fmt % p[k] for k in self.selectedClasses])
                            if self.showClass: s += " -> "
                        if self.showClass: s += "%s" % str(cl)
                        self.table.setItem(self.rindx[i], col, QTableWidgetItem(s))
                        #print s, self.rindx[i], col
                else:
                    # regression
                    for (i, d) in enumerate(self.data):
                        cl = c(d)
                        self.classifications[i].append(cl)
                        self.table.setItem(self.rindx[i], col, QTableWidgetItem(str(cl)))
                col += 1
        else:
            for i in range(len(self.data)):
                for c in range(len(self.predictors)):
                    self.table.setItem(self.rindx[i], col+c, QTableWidgetItem(''))
            col += len(self.predictors)

        for i in range(sindx, col):
            if self.showClass or (classification and self.showProb):
                self.table.showColumn(i)
##                self.table.adjustColumn(i)
            else:
                self.table.hideColumn(i)

    def updateTrueClass(self):
        return #TODO: ???
    
        col = len(self.data.domain.attributes)
        if self.data.domain.classVar:
            self.table.showColumn(col)
##            self.table.adjustColumn(col)
        else:
            self.table.hideColumn(col)

    def updateAttributes(self):
        if self.ShowAttributeMethod == 0:
            for i in range(len(self.data.domain.variables)):
                self.table.showColumn(i)
##                self.table.adjustColumn(i)
        else:
            for i in range(len(self.data.domain.variables)):
                self.table.hideColumn(i)

    def setTable(self):
        """defines the attribute/predictions table and paints its contents"""
        if not self.outvar or self.data==None:
            return

        self.table.setColumnCount(len(self.data.domain.attributes) + (self.data.domain.classVar != None) + len(self.predictors))
        self.table.setRowCount(len(self.data))
        
        #print self.table.rowCount(), len(self.data.domain.attributes), (self.data.domain.classVar != None), len(self.predictors)

        # HEADER: set the header (attribute names)
##        for col in range(len(self.data.domain.attributes)):
##            self.header.setLabel(col, self.data.domain.attributes[col].name)
        labels = [attr.name for attr in self.data.domain.variables] + [c.name for c in self.predictors.values()]
        self.table.setHorizontalHeaderLabels(labels)
##        col = len(self.data.domain.attributes)
##        if self.data.domain.classVar != None:
##            self.header.setLabel(col, self.data.domain.classVar.name)
##        col += 1
##        for (i,c) in enumerate(self.predictors.values()):
##            self.header.setLabel(col+i, c.name)

        # ATTRIBUTE VALUES: set the contents of the table (values of attributes), data first
        for i in range(len(self.data)):
            for j in range(len(self.data.domain.attributes)):
##                self.table.setText(i, j, str(self.data[i][j]))
                self.table.setItem(i, j, QTableWidgetItem(str(self.data[i][j])))
        col = len(self.data.domain.attributes)

        # TRUE CLASS: set the contents of the table (values of attributes), data first
        self.classifications = [[]] * len(self.data)
        if self.data.domain.classVar:
            for (i, d) in enumerate(self.data):
                c = d.getclass()
                item = colorItem(str(c))
                self.table.setItem(i, col, item)
                self.classifications[i] = [c]
        col += 1

##        for i in range(col):
##            self.table.adjustColumn(i)

        # include predictions, handle show/hide columns
        self.updateTableOutcomes()
        self.updateAttributes()
        self.updateTrueClass()
        self.table.show()

    def sort(self, col):
        "sorts the table by column col"
        self.sortby = - self.sortby
        self.table.sortItems(col, self.sortby>=0)

        # the table may be sorted, figure out data indices
        for i in range(len(self.data)):
            self.rindx[int(str(self.table.item(i,0).text()))-1] = i
        for (i, indx) in enumerate(self.rindx):
            self.vheader.setLabel(i, self.table.item(i,0).text())

    def checkenable(self):
        # following should be more complicated and depends on what data are we showing
        cond = (self.outvar != None) and (self.data != None)
        self.outbox.setEnabled(cond)
        self.att.setEnabled(cond)
        self.copt.setEnabled(cond)
        e = (self.data and (self.data.domain.classVar <> None) + len(self.predictors)) >= 2
        # need at least two classes to compare predictions

    def clear(self):
        self.send("Predictions", None)
        self.checkenable()
        if len(self.predictors) == 0:
            self.outvar = None
            self.classes = []
            self.selectedClasses = []
            self.predictorlabel = "N/A"
        self.table.hide()

    ##############################################################################
    # Input signals

    def consistenttarget(self, target):
        """returns TRUE if target is consistent with current predictiors and data"""
        if self.predictors:
            return target == self.predictors.values()[0]
        return True

    def setData(self, data):
        if not data:
            self.data = data
            self.datalabel = "N/A"
            self.clear()
        else:
            vartypes = {1:"discrete", 2:"continuous"}
            self.data = data
            self.rindx = range(len(self.data))
            self.setTable()
            self.checksendpredictions()
            self.datalabel = "%d instances" % len(data)
        self.checkenable()

    def setPredictor(self, predictor, id):
        """handles incoming classifier (prediction, as could be a regressor as well)"""

        def getoutvar(predictors):
            """return outcome variable, if consistent among predictors, else None"""
            if not len(predictors):
                return None
            ov = predictors[0].classVar
            for predictor in predictors[1:]:
                if ov != predictor.classVar:
                    self.warning(0, "Mismatch in class variable (e.g., predictors %s and %s)" % (predictors[0].name, predictor.name))
                    return None
            return ov

        # remove the classifier with id, if empty
        if not predictor:
            if self.predictors.has_key(id):
                del self.predictors[id]
                if len(self.predictors) == 0:
                    self.clear()
                else:
                    self.predictorlabel = "%d" % len(self.predictors)
            return

        # set the classifier
        self.predictors[id] = predictor
        self.predictorlabel = "%d" % len(self.predictors)

        # set the outcome variable
        ov = getoutvar(self.predictors.values())
        if len(self.predictors) and not ov:
            self.tasklabel = "N/A (type mismatch)"
            self.classes = []
            self.selectedClasses = []
            self.clear()
            self.outvar = None
            return
        self.warning(0) # clear all warnings

        if ov != self.outvar:
            self.outvar = ov
            # regression or classification?
            if self.outvar.varType == orange.VarTypes.Continuous:
                self.copt.hide();
                self.tasklabel = "Regression"
            else:
                self.copt.show()
                self.classes = [str(v) for v in self.outvar.values]
                self.selectedClasses = []
                self.tasklabel = "Classification"

        if self.data:
            self.setTable()
            self.table.show()
            self.checksendpredictions()
        self.checkenable()

    ##############################################################################
    # Ouput signals

    def checksendpredictions(self):
        if len(self.predictors) and self.sendOnChange:
            self.sendpredictions()

    def sendpredictions(self):
        if not self.data or not self.outvar:
            self.send("Predictions", None)
            return

        # predictions, data set with class predictions
        classification = self.outvar.varType == orange.VarTypes.Discrete

        metas = []
        if classification:
            if len(self.selectedClasses):
                for c in self.predictors.values():
                    m = [orange.FloatVariable(name="%s(%s)" % (c.name, str(self.outvar.values[i])),
                                              getValueFrom = lambda ex, rw, cindx=i, c=c: orange.Value(c(ex, c.GetProbabilities)[cindx])) \
                         for i in self.selectedClasses]
                    metas.extend(m)
            if self.showClass:
                mc = [orange.EnumVariable(name="%s" % c.name, values = self.outvar.values,
                                         getValueFrom = lambda ex, rw, c=c: orange.Value(c(ex)))
                      for c in self.predictors.values()]
                metas.extend(mc)
        else:
            # regression
            mc = [orange.FloatVariable(name="%s" % c.name, 
                                       getValueFrom = lambda ex, rw, c=c: orange.Value(c(ex)))
                  for c in self.predictors.values()]
            metas.extend(mc)

        domain = orange.Domain(self.data.domain.attributes + [self.data.domain.classVar])
        domain.addmetas(self.data.domain.getmetas())
        for m in metas:
            domain.addmeta(orange.newmetaid(), m)
        predictions = orange.ExampleTable(domain, self.data)
        predictions.name = self.data.name
        self.send("Predictions", predictions)

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":    
    a = QApplication(sys.argv)
    ow = OWPredictions()
    ow.show()

    import orngTree

    dataset = orange.ExampleTable('../../doc/datasets/iris.tab')
#    dataset = orange.ExampleTable('../../doc/datasets/auto-mpg.tab')
    ind = orange.MakeRandomIndices2(p0=0.5)(dataset)
    data = dataset.select(ind, 0)
    test = dataset.select(ind, 1)
    testnoclass = orange.ExampleTable(orange.Domain(test.domain.attributes, False), test)        
    tree = orngTree.TreeLearner(data)
    tree.name = "tree"
    maj = orange.MajorityLearner(data)
    maj.name = "maj"
    knn = orange.kNNLearner(data, k = 10)
    knn.name = "knn"

    if 0: # data set only
        ow.setData(test)
    if 0: # two predictors, test data with class
        ow.setPredictor(maj, 1)
        ow.setPredictor(tree, 2)
        ow.setData(test)
    if 0: # two predictors, test data with no class
        ow.setPredictor(maj, 1)
        ow.setPredictor(tree, 2)
        ow.setData(testnoclass)
    if 1: # three predictors
        ow.setPredictor(tree, 1)
        ow.setPredictor(maj, 2)
        ow.setData(data)
        ow.setPredictor(knn, 3)
    if 0: # just classifier, no data
        ow.setData(data)
        ow.setPredictor(maj, 1)
        ow.setPredictor(knn, 2)
    if 0: # change data set
        ow.setPredictor(maj, 1)
        ow.setPredictor(tree, 2)
        ow.setData(testnoclass)
        data = orange.ExampleTable('../../doc/datasets/titanic.tab')
        tree = orngTree.TreeLearner(data)
        tree.name = "tree"
        ow.setPredictor(tree, 2)
        ow.setData(data)

    a.exec_()
    ow.saveSettings()
