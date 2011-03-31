"""<name>Preprocess</name>
<description>Construct and apply data preprocessors</description>
<icon>icons/Preprocess.png</icon>
<priority>2200</priority>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
from OWItemModels import PyListModel, ListSingleSelectionModel, ModelActionsWidget
import OWGUI, OWGUIEx

import orange
import orngWrap
import orngSVM

import sys, os
import math

from Orange.preprocess import *

def _gettype(obj):
    """ Return type of obj. If obj is type return obj.
    """
    if isinstance(obj, type):
        return obj
    else:
        return type(obj)
        
def _pyqtProperty(type, **kwargs):
    # check for Qt version, 4.4 supports only C++ classes 
    if qVersion() >= "4.5":
        return pyqtProperty(type, **kwargs)
    else:
        if "user" in kwargs:
            del kwargs["user"]
        return property(**kwargs)

## Preprocessor item editor widgets
class BaseEditor(OWBaseWidget):
    def __init__(self, parent=None):
        OWBaseWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
            
class DiscretizeEditor(BaseEditor):
    DISCRETIZERS = [("Entropy-MDL discretization", orange.EntropyDiscretization, {}),
                    ("Equal frequency discretization", orange.EquiDistDiscretization, {"numberOfIntervals":3}),
                    ("Equal width discretization", orange.EquiNDiscretization, {"numberOfIntervals":3}),
                    ("Remove continuous attributes", type(None), {})]
    def __init__(self, parent=None):
        BaseEditor.__init__(self, parent)
        self.discInd = 0
        self.numberOfIntervals = 3
#        box = OWGUI.widgetBox(self, "Discretize")
        rb = OWGUI.radioButtonsInBox(self, self, "discInd", [], box="Discretize", callback=self.onChange)
        for label, _, _ in self.DISCRETIZERS[:-1]:
            OWGUI.appendRadioButton(rb, self, "discInd", label)
        self.sliderBox = OWGUI.widgetBox(OWGUI.indentedBox(rb, sep=OWGUI.checkButtonOffsetHint(rb.buttons[-1])), "Num. of intervals (for equal width/frequency)")
        OWGUI.hSlider(self.sliderBox, self, "numberOfIntervals", callback=self.onChange, minValue=1)
        OWGUI.appendRadioButton(rb, self, "discInd", self.DISCRETIZERS[-1][0])
        OWGUI.rubber(rb)
        
        self.updateSliderBox()
        
    def updateSliderBox(self):
        self.sliderBox.setEnabled(self.discInd in [1, 2])
        
    def onChange(self):
        self.updateSliderBox()
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def getDiscretizer(self):
        if self.discInd == 0:
            preprocessor = Preprocessor_discretizeEntropy(method=orange.EntropyDiscretization())
        elif self.discInd in [1, 2]:
            name, disc, kwds = self.DISCRETIZERS[self.discInd]
            preprocessor = Preprocessor_discretize(method=disc(**dict([(key, getattr(self, key, val)) for key, val in kwds.items()])))
        elif self.discInd == 3:
            preprocessor = Preprocessor_removeContinuous()
        return preprocessor
    
    def setDiscretizer(self, discretizer):
        disc = dict([(val, i) for i, (_, val, _) in enumerate(self.DISCRETIZERS)])
        self.discInd = disc.get(_gettype(discretizer.method), 3)
        _, d, kwargs = self.DISCRETIZERS[self.discInd]
        for key, val in kwargs.items():
            setattr(self, key, getattr(discretizer.method, key, val))
            
        self.updateSliderBox()
        
    data = _pyqtProperty(Preprocessor_discretize,
                        fget=getDiscretizer,
                        fset=setDiscretizer,
                        user=True)
    
class ContinuizeEditor(BaseEditor):
    CONTINUIZERS = [("Most frequent is base", orange.DomainContinuizer.FrequentIsBase),
                    ("One attribute per value", orange.DomainContinuizer.NValues),
                    ("Ignore multinomial attributes", orange.DomainContinuizer.Ignore),
                    ("Ignore all discrete attributes", None),
                    ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                    ("Divide by number of values",orange.DomainContinuizer.AsNormalizedOrdinal)]
    
    TREATMENT_TO_IND = dict([(val, i) for i, (_, val) in enumerate(CONTINUIZERS)])
    
    def __init__(self, parent=None):
        BaseEditor.__init__(self, parent)
        self.contInd = 0
        
        b = OWGUI.radioButtonsInBox(self, self, "contInd", [name for name, _ in self.CONTINUIZERS], box="Continuize", callback=self.onChange)
        OWGUI.rubber(b)
        
    def onChange(self):
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def getContinuizer(self):
        if self.contInd in [0, 1, 2, 4, 5]:
            preprocessor = Preprocessor_continuize(multinomialTreatment=self.CONTINUIZERS[self.contInd][1])
        elif self.contInd == 3:
            preprocessor = Preprocessor_removeDiscrete()
        return preprocessor
    
    def setContinuizer(self, continuizer):
        if isinstance(continuizer, Preprocessor_removeDiscrete):
            self.contInd = 3 #Ignore all discrete
        elif isinstance(continuizer,Preprocessor_continuize):
            self.contInd = self.TREATMENT_TO_IND.get(continuizer.multinomialTreatment, 3)
    
    data = _pyqtProperty(Preprocessor_continuize,
                        fget=getContinuizer,
                        fset=setContinuizer,
                        user=True)
    
class ImputeEditor(BaseEditor):
    IMPUTERS = [("Average/Most frequent", orange.MajorityLearner),
                ("Model-based imputer", orange.BayesLearner),
                ("Random values", orange.RandomLearner),
                ("Remove examples with missing values", None)]
    
    def __init__(self, parent):
        BaseEditor.__init__(self, parent)
        
        self.methodInd = 0
        b = OWGUI.radioButtonsInBox(self, self, "methodInd", [label for label, _ in self.IMPUTERS], box="Impute", callback=self.onChange)
        OWGUI.rubber(b)
        
    def onChange(self):
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def getImputer(self):
        if self.methodInd in [0, 1, 2]:
            learner = self.IMPUTERS[self.methodInd][1]()
            imputer = Preprocessor_imputeByLearner(learner=learner)
        elif self.methodInd == 3:
            imputer = orange.Preprocessor_dropMissing()
        return imputer
            
    
    def setImputer(self, imputer):
        self.methodInd = 0
        if isinstance(imputer, Preprocessor_imputeByLearner):
            learner = imputer.learner
            dd = dict([(t, i) for i, (_, t) in enumerate(self.IMPUTERS)])
            self.methodInd = dd.get(_gettype(learner), 0)
        elif isinstance(imputer, orange.Preprocessor_dropMissing):
            self.methodInd = 3
            
    data = _pyqtProperty(Preprocessor_imputeByLearner,
                        fget=getImputer,
                        fset=setImputer,
                        user=True)
    
class FeatureSelectEditor(BaseEditor):
    MEASURES = [("ReliefF", orange.MeasureAttribute_relief),
                ("Information Gain", orange.MeasureAttribute_info),
                ("Gain ratio", orange.MeasureAttribute_gainRatio),
                ("Gini Gain", orange.MeasureAttribute_gini),
                ("Log Odds Ratio", orange.MeasureAttribute_logOddsRatio),
                ("Linear SVM weights", orngSVM.MeasureAttribute_SVMWeights)]
    
    FILTERS = [Preprocessor_featureSelection.bestN,
               Preprocessor_featureSelection.bestP]
    
    def __init__(self, parent=None):
        BaseEditor.__init__(self, parent)
        
        self.measureInd = 0
        self.selectBy = 0
        self.bestN = 10
        self.bestP = 10
        
        box = OWGUI.radioButtonsInBox(self, self, "selectBy", [], "Feature selection", callback=self.onChange)
        
        OWGUI.comboBox(box, self, "measureInd",  items= [name for (name, _) in self.MEASURES], label="Measure", callback=self.onChange)
        
        hbox1 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb1 = OWGUI.appendRadioButton(box, self, "selectBy", "Best", insertInto=hbox1, callback=self.onChange)
        self.spin1 = OWGUI.spin(OWGUI.widgetBox(hbox1), self, "bestN", 1, 10000, step=1, controlWidth=75, callback=self.onChange, posttext="features")
        OWGUI.rubber(hbox1)
        
        hbox2 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb2 = OWGUI.appendRadioButton(box, self, "selectBy", "Best", insertInto=hbox2, callback=self.onChange)
        self.spin2 = OWGUI.spin(OWGUI.widgetBox(hbox2), self, "bestP", 1, 100, step=1, controlWidth=75, callback=self.onChange, posttext="% features")
        OWGUI.rubber(hbox2)
        
        self.updateSpinStates()
        
        OWGUI.rubber(box)
        
    def updateSpinStates(self):
        self.spin1.setDisabled(bool(self.selectBy))
        self.spin2.setDisabled(not bool(self.selectBy))
        
    def onChange(self):
        self.updateSpinStates()
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def setFeatureSelection(self, fs):
        select = dict([(filter, i) for i, filter in enumerate(self.FILTERS)])
        
        measures = dict([(measure, i) for i, (_, measure) in enumerate(self.MEASURES)])
        
        self.selectBy = select.get(fs.filter, 0)
        self.measureInd = measures.get(fs.measure, 0)
        if self.selectBy:
            self.bestP = fs.limit
        else:
            self.bestN = fs.limit
            
        self.updateSpinStates()
    
    def getFeatureSelection(self):
        return Preprocessor_featureSelection(measure=self.MEASURES[self.measureInd][1],
                                             filter=self.FILTERS[self.selectBy],
                                             limit=self.bestP if self.selectBy  else self.bestN)
    
    data = _pyqtProperty(Preprocessor_featureSelection,
                        fget=getFeatureSelection,
                        fset=setFeatureSelection,
                        user=True)
        
class SampleEditor(BaseEditor):
    FILTERS = [Preprocessor_sample.selectNRandom,
               Preprocessor_sample.selectPRandom]
    def __init__(self, parent=None):
        BaseEditor.__init__(self, parent)
        self.methodInd = 0
        self.sampleN = 100
        self.sampleP = 25
        
        box = OWGUI.radioButtonsInBox(self, self, "methodInd", [], box="Sample", callback=self.onChange)
        
        w1 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb1 = OWGUI.appendRadioButton(box, self, "methodInd", "Sample", insertInto=w1)
        self.sb1 = OWGUI.spin(OWGUI.widgetBox(w1), self, "sampleN", min=1, max=100000, step=1, controlWidth=75, callback=self.onChange, posttext="data instances")
        OWGUI.rubber(w1)
        
        w2 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb2 = OWGUI.appendRadioButton(box, self, "methodInd", "Sample", insertInto=w2)
        self.sb2 = OWGUI.spin(OWGUI.widgetBox(w2), self, "sampleP", min=1, max=100, step=1, controlWidth=75, callback=self.onChange, posttext="% data instances")
        OWGUI.rubber(w2)
        
        self.updateSpinStates()
        
        OWGUI.rubber(box)
        
    def updateSpinStates(self):
        self.sb1.setEnabled(not self.methodInd)
        self.sb2.setEnabled(self.methodInd)
        
    def onChange(self):
        self.updateSpinStates()
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def getSampler(self):
        return Preprocessor_sample(filter=self.FILTERS[self.methodInd],
                                   limit=self.sampleN if self.methodInd == 0 else self.sampleP)
    
    def setSampler(self, sampler):
        filter = dict([(s, i) for i, s in enumerate(self.FILTERS)])
        self.methodInd = filter.get(sampler.filter, 0)
        if self.methodInd == 0:
            self.sampleN = sampler.limit
        else:
            self.sampleP = sampler.limit
            
        self.updateSpinStates()
            
    data = _pyqtProperty(Preprocessor_sample,
                        fget=getSampler,
                        fset=setSampler,
                        user=True)
    
def _funcName(func):
    return func.__name__
    
class PreprocessorItemDelegate(QStyledItemDelegate):
        
    #Preprocessor name replacement rules
    REPLACE = {Preprocessor_discretize: "Discretize ({0.method})",
               Preprocessor_discretizeEntropy: "Discretize (entropy)",
               Preprocessor_removeContinuous: "Discretize (remove continuous)",
               Preprocessor_continuize: "Continuize ({0.multinomialTreatment})",
               Preprocessor_removeDiscrete: "Continuize (remove discrete)",
               Preprocessor_impute: "Impute ({0.model})",
               Preprocessor_imputeByLearner: "Impute ({0.learner})",
               Preprocessor_dropMissing: "Remove missing",
               Preprocessor_featureSelection: "Feature selection ({0.measure}, {0.filter}, {0.limit})",
               Preprocessor_sample: "Sample ({0.filter}, {0.limit})",
               orange.EntropyDiscretization: "entropy",
               orange.EquiNDiscretization: "freq, {0.numberOfIntervals}",
               orange.EquiDistDiscretization: "width, {0.numberOfIntervals}",
               orange.RandomLearner: "random",  
               orange.BayesLearner: "bayes  model",
               orange.MajorityLearner: "average",
               orange.MeasureAttribute_relief: "ReliefF",
               orange.MeasureAttribute_info: "Info gain",
               orange.MeasureAttribute_gainRatio: "Gain ratio",
               orange.MeasureAttribute_gini: "Gini",
               orange.MeasureAttribute_logOddsRatio: "Log Odds",
               orngSVM.MeasureAttribute_SVMWeights: "Linear SVM weights",
               type(lambda : None): _funcName}
    
    import re
    INSERT_RE = re.compile(r"{0\.(\w+)}")
    
    def __init__(self, parent=None):
        QStyledItemDelegate.__init__(self, parent)
        
    def displayText(self, value, locale):
        try:
            p = value.toPyObject()
            return self.format(p)
        except Exception, ex:
            return repr(ex)
        
    def format(self, obj):
        def replace(match):
            attr = match.groups()[0]
            if hasattr(obj, attr):
                return self.format(getattr(obj, attr))
        
        text = self.REPLACE.get(_gettype(obj), str(obj))
        if hasattr(text, "__call__"):
            return text(obj)
        else:
            return self.INSERT_RE.sub(replace, text)
        
class PreprocessorSchema(object):
    """ Preprocessor schema holds a saved a named preprocessor list for display.
    """
    def __init__(self, name="New schema", preprocessors=[], selectedPreprocessor=0, modified=False):
        self.name = name
        self.preprocessors = preprocessors
        self.selectedPreprocessor = selectedPreprocessor
        self.modified = modified
        
class PreprocessorSchemaDelegate(QStyledItemDelegate):
    @classmethod
    def asSchema(cls, obj):
        if isinstance(obj, PreprocessorSchema):
            return obj
        elif isinstance(obj, tuple):
            return PreprocessorSchema(*obj)
        
    def displayText(self, value, locale):
        schema = self.asSchema(value.toPyObject())
        try:
            if schema.modified:
                return QString("*" + schema.name)
            else:
                return QString(schema.name)
        except Exception:
            return QString("Invalid schema")
        
    def paint(self, painter, option, index):
        schema = self.asSchema(index.data(Qt.DisplayRole).toPyObject())
        if getattr(schema, "modified", False):
            option = QStyleOptionViewItemV4(option)
            option.palette.setColor(QPalette.Text, QColor(Qt.red))
            option.palette.setColor(QPalette.Highlight, QColor(Qt.darkRed))
        QStyledItemDelegate.paint(self, painter, option, index)
        
    def createEditor(self, parent, option, index):
        return QLineEdit(parent)
    
    def setEditorData(self, editor, index):
        schema = self.asSchema(index.data().toPyObject())
        editor.setText(schema.name)
        
    def setModelData(self, editor, model, index):
        schema = self.asSchema(index.data().toPyObject())
        schema.name = editor.text()
        model.setData(index, QVariant(schema))
        
        
class PySortFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, filter_fmt=None, sort_fmt=None, parent=None):
        QSortFilterProxyModel.__init__(self, parent)
        self.filter_fmt = filter_fmt
        self.sort_fmt = sort_fmt
        
    if sys.version < "2.6":
        import re
        INSERT_RE = re.compile(r"{0\.(\w+)}")
        def format(self, fmt, *args, **kwargs):
            # a simple formating function for python 2.5
            def replace(match):
                attr = match.groups()[0]
                if hasattr(args[0], attr):
                    return str(getattr(args[0], attr))
            return self.INSERT_RE.sub(replace, fmt)
        
    else:
        def format(self, fmt, *args, **kwargs):
            return fmt.format(*args, **kwargs)
        
    def lessThen(self, left, right):
        left = self.sourceModel().data(left)
        right = self.sourceModel().data(right)
        
        left, right = left.toPyObject(), right.toPyObject()
        
        if self.sort_fmt is not None:
            left = self.format(self.sort_fmt, left)
            right = self.format(self.sort_fmt, right)
        
        return left < right
    
    def filterAcceptsRow(self, sourceRow, sourceParent):
        index = self.sourceModel().index(sourceRow, 0, sourceParent)
        
        value = index.data().toPyObject()
        if self.filter_fmt:
            value = self.format(self.filter_fmt, value)
            
        regexp = self.filterRegExp()
        return regexp.indexIn(str(value)) >= 0

class OWPreprocess(OWWidget):
    contextHandlers = {"": PerfectDomainContextHandler("", [""])}
    settingsList = ["allSchemas", "lastSelectedSchemaIndex"]
    
    # Default preprocessors
    preprocessors =[("Discretize", Preprocessor_discretizeEntropy, {}),
                    ("Continuize", Preprocessor_continuize, {}),
                    ("Impute", Preprocessor_impute, {}),
                    ("Feature selection", Preprocessor_featureSelection, {}),
                    ("Sample", Preprocessor_sample, {})]
    
    # Editor widgets for preprocessors
    EDITORS = {Preprocessor_discretize: DiscretizeEditor,
               Preprocessor_discretizeEntropy: DiscretizeEditor,
               Preprocessor_removeContinuous: DiscretizeEditor,
               Preprocessor_continuize: ContinuizeEditor,
               Preprocessor_removeDiscrete: ContinuizeEditor,
               Preprocessor_impute: ImputeEditor,
               Preprocessor_imputeByLearner: ImputeEditor,
               Preprocessor_dropMissing: ImputeEditor,
               Preprocessor_featureSelection: FeatureSelectEditor,
               Preprocessor_sample: SampleEditor,
               type(None): QWidget}
    
    def __init__(self, parent=None, signalManager=None, name="Preprocess"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Example Table", ExampleTable, self.setData)] #, ("Learner", orange.Learner, self.setLearner)]
        self.outputs = [("Preprocess", orngWrap.PreprocessedLearner), ("Preprocessed Example Table", ExampleTable)] #, ("Preprocessor", orange.Preprocessor)]
        
        self.autoCommit = False
        self.changedFlag = False
        
#        self.allSchemas = [PreprocessorSchema("Default" , [Preprocessor_discretize(method=orange.EntropyDiscretization()), Preprocessor_dropMissing()])]
        self.allSchemas = [("Default" , [Preprocessor_discretizeEntropy(method=orange.EntropyDiscretization()), Preprocessor_dropMissing()], 0)]
        
        self.lastSelectedSchemaIndex = 0
        
        self.preprocessorsList =  PyListModel([], self)
        
        box = OWGUI.widgetBox(self.controlArea, "Preprocessors", addSpace=True)
        box.layout().setSpacing(1)
        
        self.setStyleSheet("QListView::item { margin: 1px;}")
        self.preprocessorsListView = QListView()
        self.preprocessorsListSelectionModel = ListSingleSelectionModel(self.preprocessorsList, self)
        self.preprocessorsListView.setItemDelegate(PreprocessorItemDelegate(self))
        self.preprocessorsListView.setModel(self.preprocessorsList)
        
        self.preprocessorsListView.setSelectionModel(self.preprocessorsListSelectionModel)
        self.preprocessorsListView.setSelectionMode(QListView.SingleSelection)
        
        self.connect(self.preprocessorsListSelectionModel, SIGNAL("selectedIndexChanged(QModelIndex)"), self.onPreprocessorSelection)
        self.connect(self.preprocessorsList, SIGNAL("dataChanged(QModelIndex, QModelIndex)"), lambda arg1, arg2: self.commitIf)
        
        box.layout().addWidget(self.preprocessorsListView)
        
        self.addPreprocessorAction = QAction("+",self)
        self.addPreprocessorAction.pyqtConfigure(toolTip="Add a new preprocessor to the list")
        self.removePreprocessorAction = QAction("-", self)
        self.removePreprocessorAction.pyqtConfigure(toolTip="Remove selected preprocessor from the list")
        self.removePreprocessorAction.setEnabled(False)
        
        self.connect(self.preprocessorsListSelectionModel, SIGNAL("selectedIndexChanged(QModelIndex)"), lambda index:self.removePreprocessorAction.setEnabled(index.isValid()))
        
        actionsWidget = ModelActionsWidget([self.addPreprocessorAction, self.removePreprocessorAction])
        actionsWidget.layout().setSpacing(1)
        actionsWidget.layout().addStretch(10)
        
        box.layout().addWidget(actionsWidget)
        
        self.connect(self.addPreprocessorAction, SIGNAL("triggered()"), self.onAddPreprocessor)
        self.connect(self.removePreprocessorAction, SIGNAL("triggered()"), self.onRemovePreprocessor)
        
        box = OWGUI.widgetBox(self.controlArea, "Saved Schemas", addSpace=True)
        
        self.schemaFilterEdit = OWGUIEx.LineEditFilter(self)
        box.layout().addWidget(self.schemaFilterEdit)
        
        self.schemaList = PyListModel([], self, flags=Qt.ItemIsSelectable | Qt.ItemIsEditable| Qt.ItemIsEnabled)
        self.schemaListProxy = PySortFilterProxyModel(filter_fmt="{0.name}", parent=self)
        self.schemaListProxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.schemaListProxy.setSourceModel(self.schemaList)
        self.schemaListView = QListView()
        self.schemaListView.setItemDelegate(PreprocessorSchemaDelegate(self))
#        self.schemaListView.setModel(self.schemaList)
        self.schemaListView.setModel(self.schemaListProxy)
        self.connect(self.schemaFilterEdit, SIGNAL("textEdited(QString)"), self.schemaListProxy.setFilterRegExp)
        box.layout().addWidget(self.schemaListView)
        
        self.schemaListSelectionModel = ListSingleSelectionModel(self.schemaListProxy, self)
        self.schemaListView.setSelectionMode(QListView.SingleSelection)
        self.schemaListView.setSelectionModel(self.schemaListSelectionModel)
        
        self.connect(self.schemaListSelectionModel, SIGNAL("selectedIndexChanged(QModelIndex)"), self.onSchemaSelection)
        
        self.addSchemaAction = QAction("+", self)
        self.addSchemaAction.pyqtConfigure(toolTip="Add a new preprocessor schema")
        self.updateSchemaAction = QAction("Update", self)
        self.updateSchemaAction.pyqtConfigure(toolTip="Save changes made in the current schema")
        self.removeSchemaAction = QAction("-", self)
        self.removeSchemaAction.pyqtConfigure(toolTip="Remove selected schema")
        
        self.updateSchemaAction.setEnabled(False)
        self.removeSchemaAction.setEnabled(False)
        
        actionsWidget = ModelActionsWidget([])
        actionsWidget.addAction(self.addSchemaAction)
        actionsWidget.addAction(self.updateSchemaAction).setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        actionsWidget.addAction(self.removeSchemaAction)
        actionsWidget.layout().setSpacing(1)
        
        box.layout().addWidget(actionsWidget)
        
        self.connect(self.addSchemaAction, SIGNAL("triggered()"), self.onAddSchema)
        self.connect(self.updateSchemaAction, SIGNAL("triggered()"), self.onUpdateSchema)
        self.connect(self.removeSchemaAction, SIGNAL("triggered()"), self.onRemoveSchema)
        
        self.addPreprocessorsMenuActions = actions = []
        for name, pp, kwargs in self.preprocessors:
            action = QAction(name, self)
            self.connect(action, SIGNAL("triggered()"), lambda pp=pp, kwargs=kwargs:self.addPreprocessor(pp(**kwargs)))
            actions.append(action)
            
        box = OWGUI.widgetBox(self.controlArea, "Output")
        cb = OWGUI.checkBox(box, self, "autoCommit", "Commit on any change", callback=self.commitIf)
        b = OWGUI.button(box, self, "Commit", callback=self.commit)
        OWGUI.setStopper(self, b, cb, "changedFlag", callback=self.commitIf)
        
        self.mainAreaStack = QStackedLayout()
        self.stackedEditorsCache = {}
        
        OWGUI.widgetBox(self.mainArea, orientation=self.mainAreaStack)
        
        self.data = None
        self.learner = None
        
        self.loadSettings()
        self.activateLoadedSettings()
        
    def activateLoadedSettings(self):
        try:
            self.allSchemas = [PreprocessorSchemaDelegate.asSchema(obj) for obj in self.allSchemas]
            for s in self.allSchemas:
                s.modified = False
            self.schemaList.wrap(self.allSchemas)
            self.schemaListSelectionModel.select(self.schemaList.index(min(self.lastSelectedSchemaIndex, len(self.schemaList) - 1)), QItemSelectionModel.ClearAndSelect)
            self.commit()
        except Exception, ex:
            print repr(ex)
            
    def setData(self, data=None):
        self.data = data
#        self.commit()
    
    def setLearner(self, learner=None):
        self.learner = learner
#        self.commit()

    def handleNewSignals(self):
        self.commit()
        
    def selectedSchemaIndex(self):
        rows = self.schemaListSelectionModel.selectedRows()
        rows = [self.schemaListProxy.mapToSource(row) for row in rows]
        if rows:
            return rows[0]
        else:
            return QModelIndex()
    
    def addPreprocessor(self, prep):
        self.preprocessorsList.append(prep)
        self.preprocessorsListSelectionModel.select(self.preprocessorsList.index(len(self.preprocessorsList)-1),
                                                    QItemSelectionModel.ClearAndSelect)
        self.commitIf()
        self.setSchemaModified(True)
    
    def onAddPreprocessor(self):
        action = QMenu.exec_(self.addPreprocessorsMenuActions, QCursor.pos())
    
    def onRemovePreprocessor(self):
        index = self.preprocessorsListSelectionModel.selectedRow()
        if index.isValid():
            row = index.row()
            del self.preprocessorsList[row]
            newrow = min(max(row - 1, 0), len(self.preprocessorsList) - 1)
            if newrow > -1:
                self.preprocessorsListSelectionModel.select(self.preprocessorsList.index(newrow), QItemSelectionModel.ClearAndSelect)
            self.commitIf()
            self.setSchemaModified(True)
            
    def onPreprocessorSelection(self, index):
        if index.isValid():
            pp = self.preprocessorsList[index.row()]
            self.currentSelectedIndex = index.row()
            self.showEditWidget(pp)
        else:
            self.showEditWidget(None)
        
    def onSchemaSelection(self, index):
        self.updateSchemaAction.setEnabled(index.isValid())
        self.removeSchemaAction.setEnabled(index.isValid())
        if index.isValid():
            self.lastSelectedSchemaIndex = index.row()
            self.setActiveSchema(index.data().toPyObject())
    
    def onAddSchema(self):
        schema = list(self.preprocessorsList)
        self.schemaList.append(PreprocessorSchema("New schema", schema, self.preprocessorsListSelectionModel.selectedRow().row()))
        index = self.schemaList.index(len(self.schemaList) - 1)
        index = self.schemaListProxy.mapFromSource(index)
        self.schemaListSelectionModel.setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
        self.schemaListView.edit(index)
    
    def onUpdateSchema(self):
#        index = self.schemaListSelectionModel.selectedRow()
        index = self.selectedSchemaIndex()
        if index.isValid():
            row = index.row()
            schema = self.schemaList[row]
            self.schemaList[row] = PreprocessorSchema(schema.name, list(self.preprocessorsList),
                                                      self.preprocessorsListSelectionModel.selectedRow().row())
    
    def onRemoveSchema(self):
#        index = self.schemaListSelectionModel.selectedRow()
        index = self.selectedSchemaIndex()
        if index.isValid():
            row = index.row()
            del self.schemaList[row]
            newrow = min(max(row - 1, 0), len(self.schemaList) - 1)
            if newrow > -1:
                self.schemaListSelectionModel.select(self.schemaListProxy.mapFromSource(self.schemaList.index(newrow)),
                                                     QItemSelectionModel.ClearAndSelect)
                
    def setActiveSchema(self, schema):
        if schema.modified and hasattr(schema, "_tmp_preprocessors"):
            self.preprocessorsList[:] = list(schema._tmp_preprocessors)
        else:
            self.preprocessorsList[:] = list(schema.preprocessors)
        self.preprocessorsListSelectionModel.select(schema.selectedPreprocessor, QItemSelectionModel.ClearAndSelect)
        self.commitIf()
        
    def showEditWidget(self, pp):
        w = self.stackedEditorsCache.get(type(pp), None)
        if w is None:
            w = self.EDITORS[type(pp)](self.mainArea)
            self.stackedEditorsCache[type(pp)] = w
            self.connect(w, SIGNAL("dataChanged"), self.setEditedPreprocessor)
            self.mainAreaStack.addWidget(w)
        self.mainAreaStack.setCurrentWidget(w)
        w.data = pp
        w.show()
        
    def setEditedPreprocessor(self, pp):
        self.preprocessorsList[self.preprocessorsListSelectionModel.selectedRow().row()] = pp
        
        self.setSchemaModified(True)
        self.commitIf()
#        self.onUpdateSchema()
        
    def setSchemaModified(self, state):
#        index = self.schemaListSelectionModel.selectedRow()
        index = self.selectedSchemaIndex()
        if index.isValid():
            row = index.row()
            self.schemaList[row].modified = True
            self.schemaList[row]._tmp_preprocessors = list(self.preprocessorsList)
            self.schemaList.emitDataChanged([row])
        
    def commitIf(self):
        if self.autoCommit:
            self.commit()
        else:
            self.changedFlag = True
            
    def commit(self):
        wrap = orngWrap.PreprocessedLearner(list(self.preprocessorsList))
        if self.data is not None:
            data = wrap.processData(self.data)
            self.send("Preprocessed Example Table", data)
        self.send("Preprocess", wrap)
            
#        self.send("Preprocessor", Preprocessor_preprocessorList(list(self.preprocessorsList)))
        self.changedFlag = False
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPreprocess()
    w.setData(orange.ExampleTable("../../doc/datasets/iris"))
    w.show()
    app.exec_()
    w.saveSettings()
    