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
import orange

from OWDataTable import ExampleTableModel, getCached

def safe_call(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception, ex:
            print >> sys.stderr, func.__name__, "call error", ex 
            return QVariant()
    return wrapper

class PyTableModel(QAbstractTableModel):
    """ A general list-of-lists table holding arbitrary Python objects.
    To view it in a item view you must subclass an QItemDelegate
    
    Arguments:
        - `table`: A 2D table of any python objects
        - `headers`: A list of strings for view headers
        - `parent`: models parent (default None)
        
    Examples::
        view = QTableView()
        view.setModel(PyTableView([[1, 2, 3], [1, 2, 3]], ["One, "Two", "Three"], parent=view)
    """
    def __init__(self, table=None, headers=None, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._table = [[]] if table is None else table
        self._header = [None] * len(self._table) if headers is None else headers
        
    @safe_call
    def data(self, index, role=Qt.DisplayRole):
        row, column = index.row(), index.column()
        if role == Qt.DisplayRole or role == Qt.EditRole:
            val = self._table[row][column]
            return QVariant(val)
        else:
            return QVariant() #QAbstractTableModel.data(self, index, role)
        
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._table)
        
    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return max([len(row) for row in self._table]) if self._table else 0
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical and  role == Qt.DisplayRole:
            return QVariant(QString(str(section + 1)))
        elif orientation == Qt.Horizontal and role == Qt.DisplayRole:
            header = self._header[section] if section < len(self._header) else str(section)
            return QVariant(QString(header)) if header is not None else QVariant()
        else:
            return QVariant() #QAbstractTableModel.headerData(self, section, orientation, role)
        
    def sort(self, column, order=Qt.AscendingOrder):
        self._table.sort(key=lambda row: row[column], reverse=order == Qt.DescendingOrder)
        self.reset()
        
        
class PredictionTableModel(PyTableModel):
    """ Item model for classifier predictions
    """
    def __init__(self, prediction_results, *args, **kwargs):
        """ prediciton_results [(classifier, list-of-predictions) ...]
        """
        PyTableModel.__init__(self, *args, **kwargs)
        self.prediction_results = prediction_results
        self._header = [p[0].name for p in prediction_results]
        
        self._table = [[] for i in range(max([len(p) for c, p in prediction_results] or [0]))]
        for c, pred in prediction_results:
            for i, p in enumerate(pred):
                self._table[i].append(p)

    
class PredictionItemDelegete(QStyledItemDelegate):
    """ Item delegate for prediction results i.e. (class, probabilities) 
    tuples as returned by classifiers with orange.Both return value
    
    Optional arguments:
        - `showProbs`: a list of `True` or `False` values for each class value.
        These are the probabilities that will be displayed (default all False
        i.e. no probabilities shown)
        - `deciamals`: number of decimals to show
    """
    def __init__(self, parent=None, *kwargs):
        QStyledItemDelegate.__init__(self, parent)
        self.__dict__.update(kwargs)
    
    def displayText(self, value, locale):
        pred = value.toPyObject()
        if type(pred) >= tuple:
            cls, prob = pred
        elif type(pred) >= orange.Value:
            cls, prob = pred, None
        elif type(pred) >= orange.Distribution:
            cls, prob = pred.modus(), pred
        else:
            return QString("")
        text = ""
        if prob and any(getattr(self, "showProbs", [])):
            fmt = "%%.%if" % getattr(self, "decimals", 2)
            text = " : ".join(fmt % f for f, show in zip(prob, self.showProbs) if show)
            if getattr(self, "showClass", True):
                text += " -> "
        if getattr(self, "showClass", True):
            text += str(cls)
        return QString(text)

class OWPredictions(OWWidget):
    settingsList = ["showProb", "showClass", "ShowAttributeMethod", "sendOnChange", "precision"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Predictions")

        self.callbackDeposit = []
        self.inputs = [("Data", ExampleTable, self.setData), ("Predictors", orange.Classifier, self.setPredictor, Multiple)]
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
        self.doPrediction = True
        self.outvar = None # current output variable (set by the first predictor/data set send in)

        self.data = None
        self.changedFlag = False
        
        self.loadSettings()

        # GUI - Options

        # Options - classification
        ibox = OWGUI.widgetBox(self.controlArea, "Info")
        OWGUI.label(ibox, self, "Data: %(datalabel)s")
        OWGUI.label(ibox, self, "Predictors: %(predictorlabel)s")
        OWGUI.label(ibox, self, "Task: %(tasklabel)s")
        OWGUI.separator(self.controlArea)
        
        self.copt = OWGUI.widgetBox(self.controlArea, "Options (classification)")
        self.copt.setDisabled(1)
        cb = OWGUI.checkBox(self.copt, self, 'showProb', "Show predicted probabilities", callback=self.setPredictionDelegate)#self.updateTableOutcomes)

#        self.lbClasses = OWGUI.listBox(self.copt, self, selectionMode = QListWidget.MultiSelection, callback = self.updateTableOutcomes)
        ibox = OWGUI.indentedBox(self.copt, sep=OWGUI.checkButtonOffsetHint(cb))
        self.lbcls = OWGUI.listBox(ibox, self, "selectedClasses", "classes",
                                   callback=[self.setPredictionDelegate, self.checksendpredictions],
#                                   callback=[self.updateTableOutcomes, self.checksendpredictions],
                                   selectionMode=QListWidget.MultiSelection)
        self.lbcls.setFixedHeight(50)

        OWGUI.spin(ibox, self, "precision", 1, 6, label="No. of decimals: ",
                   orientation=0, callback=self.setPredictionDelegate) #self.updateTableOutcomes)
        
        cb.disables.append(ibox)
        ibox.setEnabled(bool(self.showProb))

        OWGUI.checkBox(self.copt, self, 'showClass', "Show predicted class",
                       callback=[self.setPredictionDelegate, self.checksendpredictions])
#                       callback=[self.updateTableOutcomes, self.checksendpredictions])

        OWGUI.separator(self.controlArea)

        self.att = OWGUI.widgetBox(self.controlArea, "Data attributes")
        OWGUI.radioButtonsInBox(self.att, self, 'ShowAttributeMethod', ['Show all', 'Hide all'], callback=lambda :self.setDataModel(self.data)) #self.updateAttributes)
        self.att.setDisabled(1)
        OWGUI.rubber(self.controlArea)

        OWGUI.separator(self.controlArea)
        self.outbox = OWGUI.widgetBox(self.controlArea, "Output")
        
        b = self.commitBtn = OWGUI.button(self.outbox, self, "Send Predictions", callback=self.sendpredictions, default=True)
        cb = OWGUI.checkBox(self.outbox, self, 'sendOnChange', 'Send automatically')
        OWGUI.setStopper(self, b, cb, "changedFlag", callback=self.sendpredictions)
        OWGUI.checkBox(self.outbox, self, "doPrediction", "Replace/add predicted class",
                       tooltip="Apply the first predictor to input examples and replace/add the predicted value as the new class variable.",
                       callback=self.checksendpredictions)

        self.outbox.setDisabled(1)

        ## GUI table

        self.splitter = splitter = QSplitter(Qt.Horizontal, self.mainArea)
        self.dataView = QTableView()
        self.predictionsView = QTableView()
        
        self.dataView.verticalHeader().setDefaultSectionSize(22)
        self.dataView.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.dataView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dataView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.predictionsView.verticalHeader().setDefaultSectionSize(22)
        self.predictionsView.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.predictionsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.predictionsView.verticalHeader().hide()
        
        
        def syncVertical(value):
            """ sync vertical scroll positions of the two views
            """
            v1 = self.predictionsView.verticalScrollBar().value()
            if v1 != value:
                self.predictionsView.verticalScrollBar().setValue(value)
            v2 = self.dataView.verticalScrollBar().value()
            if v2 != value:
                self.dataView.verticalScrollBar().setValue(v1)
                
        self.connect(self.dataView.verticalScrollBar(), SIGNAL("valueChanged(int)"), syncVertical)
        self.connect(self.predictionsView.verticalScrollBar(), SIGNAL("valueChanged(int)"), syncVertical)
        
        splitter.addWidget(self.dataView)
        splitter.addWidget(self.predictionsView)
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(False)
        self.mainArea.layout().addWidget(splitter)
        
        self.spliter_restore_state = -1, 0
        self.dataModel = None
        self.predictionsModel = None
        
        self.resize(800, 600)
        
        self.handledAllSignalsFlag = False
        
        
    def updateSpliter(self):
        if not (self.dataModel and self.dataModel.columnCount() and \
                self.predictionsModel and self.predictionsModel.columnCount()):
            return
        def width(view):
            h_header = view.horizontalHeader()
            v_header = view.verticalHeader()
            return h_header.length() + v_header.width()
        
        def widthForColumns(view, start=-sys.maxint, end=sys.maxint):
            h_header = view.horizontalHeader()
            v_header = view.verticalHeader()
            return sum([h_header.sectionSize(ind) for ind in range(h_header.count())[start : end]] or [0]) + v_header.width()
        
        if self.ShowAttributeMethod == 1:
            w1, w2 = self.splitter.sizes()
            w = width(self.dataView) + 4
            self.splitter.setSizes([w, w1 + w2 - w])
            self.dataView.setMaximumWidth(w)
            
            state, w = self.spliter_restore_state
            if state == 0: # save the dataView width on change from 'show all' to 'hide all'
                self.spliter_restore_state = 1, w1
        else:
            w1, w2 = self.splitter.sizes()
            state, w = self.spliter_restore_state
            if state == 1: # restore dataView on change from 'hide all' to 'show all'
                w = min(w, (w1 + w2)*2 / 3)
            else:
                w1, w2 = self.splitter.sizes()
                w = widthForColumns(self.dataView, -2) + 4
                w = min(w,  (w1 + w2) / 2)
                w = max(w,  min(w1 + w2 - widthForColumns(self.predictionsView) - 20, w1 + w2 - w))
            self.splitter.setSizes([w, w1 + w2 - w])
            self.dataView.setMaximumWidth(16777215) # This is QWidget's max width
            
            self.spliter_restore_state = 0, w
            
    def setDataModel(self, data):
        if data is not None and self.outvar is not None:
            if self.ShowAttributeMethod == 1: # Show only the class column
                data = orange.ExampleTable(orange.Domain([self.outvar]), data)
            elif not data.domain.classVar: # add outVar as class (all unknown values) to data
                domain = orange.Domain(data.domain.attributes + [self.outvar])
                domain.addmetas(data.domain.getmetas())
                data = orange.ExampleTable(domain, data)
                
            dist = getCached(data, orange.DomainBasicAttrStat, (data,))
            self.dataModel = ExampleTableModel(data, dist, None)
            self.dataView.setModel(self.dataModel)
            self.dataView.setItemDelegate(OWGUI.TableBarItem(self, data, color = Qt.lightGray))
            count = self.dataModel.columnCount()
            if count:
                self.dataView.scrollTo(self.dataModel.index(0, count - 1))
            self.dataView.show()
        else:
            self.clear()
        self.updateSpliter()
            
    def setPredictionModel(self, classifiers, data):
        predictions = [(c, [c(ex, orange.GetBoth) for ex in self.data]) for c in classifiers]
        self.predictionsModel = PredictionTableModel(predictions)
        self.predictionsView.setModel(self.predictionsModel)
        self.setPredictionDelegate()
        self.predictionsView.show()
        
        
    def setPredictionDelegate(self):
        delegate = PredictionItemDelegete(self)
        delegate.showProbs = [self.showProb and i in self.selectedClasses for i in range(len(self.classes))]
        delegate.decimals = self.precision
        delegate.showClass = self.showClass
        self.predictionsView.setItemDelegate(delegate)
        self.predictionsView.resizeColumnsToContents()
        self.updateSpliter()

#    def sort(self, col):
#        "sorts the table by column col"
#        self.sortby = - self.sortby
#        self.table.sortItems(col, self.sortby>=0)
#
#        # the table may be sorted, figure out data indices
#        for i in range(len(self.data)):
#            self.rindx[int(str(self.table.item(i,0).text()))-1] = i
#        for (i, indx) in enumerate(self.rindx):
#            self.vheader.setLabel(i, self.table.item(i,0).text())

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
        self.dataModel = PyTableModel([[]])
        self.dataView.setModel(self.dataModel)
        self.predictionsModel = PyTableModel([[]])
        self.predictionsView.setModel(self.predictionsModel)
        
        self.dataView.hide()
        self.predictionsView.hide()
        

    ##############################################################################
    # Input signals

    def handleNewSignals(self):
        self.handledAllSignalsFlag = True
        if self.data:
            self.setDataModel(self.data)
            self.setPredictionModel(self.predictors.values(), self.data)
        self.checksendpredictions()

    def setData(self, data):
        self.handledAllSignalsFlag = False
        if not data:
            self.data = data
            self.datalabel = "N/A"
            self.clear()
        else:
            vartypes = {1:"discrete", 2:"continuous"}
            self.data = data
            self.rindx = range(len(self.data))
            self.datalabel = "%d instances" % len(data)
            
        self.checkenable()
        self.changedFlag = True

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

        self.handledAllSignalsFlag = False
        
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
                self.copt.hide()
                self.tasklabel = "Regression"
            else:
                self.copt.show()
                self.classes = [str(v) for v in self.outvar.values]
                self.selectedClasses = []
                self.tasklabel = "Classification"
                
        self.checkenable()
        self.changedFlag = True

    ##############################################################################
    # Ouput signals

    def checksendpredictions(self):
        if self.sendOnChange:
            self.sendpredictions()
        else:
            self.changedFlag = True

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
                    m = [orange.FloatVariable(name=str("%s(%s)" % (c.name, str(self.outvar.values[i]))),
                                              getValueFrom = lambda ex, rw, cindx=i, c=c: orange.Value(c(ex, c.GetProbabilities)[cindx])) \
                         for i in self.selectedClasses]
                    metas.extend(m)
            if self.showClass:
                mc = [orange.EnumVariable(name=str(c.name), values = self.outvar.values,
                                         getValueFrom = lambda ex, rw, c=c: orange.Value(c(ex)))
                      for c in self.predictors.values()]
                metas.extend(mc)
        else:
            # regression
            mc = [orange.FloatVariable(name="%s" % c.name, 
                                       getValueFrom = lambda ex, rw, c=c: orange.Value(c(ex)))
                  for c in self.predictors.values()]
            metas.extend(mc)
                
        classVar = self.outvar
        domain = orange.Domain(self.data.domain.attributes + [classVar])
        domain.addmetas(self.data.domain.getmetas())
        for m in metas:
            domain.addmeta(orange.newmetaid(), m)
        predictions = orange.ExampleTable(domain, self.data)
        if self.doPrediction:
            c = self.predictors.values()[0]
            for ex in predictions:
                ex[classVar] = c(ex)
                
        predictions.name = self.data.name
        self.send("Predictions", predictions)
        
        self.changedFlag = False

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
    
#    ow.setData(test)
#    
#    ow.setPredictor(maj, 1)
    
    

    if 1: # data set only
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
        
    ow.handleNewSignals()

    a.exec_()
    ow.saveSettings()
