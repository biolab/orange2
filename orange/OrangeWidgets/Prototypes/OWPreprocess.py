"""<name>Preprocess</name>
<description>Construct and apply data preprocessors</description>
<icon>icons/Preprocess.png</icon>
"""

from OWWidget import *
from OWItemModels import PyListModel, ListSingleSelectionModel, ModelActionsWidget
import OWGUI, OWGUIEx

import orange
import orngWrap

import sys, os 

from orange import Preprocessor_discretize, Preprocessor_imputeByLearner, Preprocessor_dropMissing

import copy_reg

def _orange__reduce(self):
    """ A default __reduce__ method for orange types.
    
    Example::
        class NewOrangeType(orange.Learner):
            __reduce__ = _orange__reduce()
    """ 
    return type(self), (), dict(self.__dict__)
    
def _orange__new(base=orange.Preprocessor):
    """ Return an orange 'schizofrenic' __new__ class method.
    
    Arguments:
        - `base`: base orange class (default orange.Preprocessor)
         
    Example::
        class NewOrangeType(orange.Learner):
            __new__ = _orange__new()
        
    """
    def _orange__new_wrapped(cls, data=None, **kwargs):
        self = base.__new__(cls, **kwargs)
        if data:
            self.__init__(kwargs)
            return self.__call__(data)
        else:
            return self
    return _orange__new_wrapped

class Preprocessor_removeContinuous(Preprocessor_discretize):
    """ A preprocessor that removes all continuous attributes.
    """
    __new__ = _orange__new(Preprocessor_discretize)
    __reduce__ = _orange__reduce
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.domain.attributes if attr.varType == orange.VarTypes.Discrete]
        domain = orange.Domain(attrs, data.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
                
class Preprocessor_continuize(orange.Preprocessor):
    """ A preprocessor that continuizes a discrete domain (and optionaly normalizes it).
    See DomainContinuizer_ for list of accepted arguments.  
    """
    __new__ = _orange__new()
    __reduce__ = _orange__reduce
    
    def __init__(self, zeroBased=True, multinomialTreatment=orange.DomainContinuizer.NValues, normalizeContinuous=False, **kwargs):
        self.zeroBased = zeroBased
        self.multinomialTreatment = multinomialTreatment
        self.normalizeContinuous = normalizeContinuous
            
    def __call__(self, data, weightId=0):
        continuizer = orange.DomainContinuizer(zeroBased=self.zeroBased,
                                               multinomialTreatment=self.multinomialTreatment,
                                               normalizeContinuous=self.normalizeContinuous,
                                               classTreatment=orange.DomainContinuizer.Ignore)
        c_domain = continuizer(data, weightId)
        return data.translate(c_domain)
    
class Preprocessor_removeDiscrete(Preprocessor_continuize):
    """ A Preprocessor that removes all discrete attributes from the domain.
    """
    __new__ = _orange__new(Preprocessor_continuize)
    
    def __call__(self, data, weightId=None):
        attrs = [attr for attr in data.attributes if attr.varType == orange.VarTypes.Continuous]
        domain = orange.Domain(attrs, data.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
         
class Preprocessor_impute(orange.Preprocessor):
    """ A preprocessor that imputes unknown values using a learner.
    Arguments:
        - `model`: a learner class
    """
    __new__ = _orange__new()
    __reduce__ = _orange__reduce
    
    def __init__(self, model=None, **kwargs):
        self.model = orange.MajorityLearner if model is None else model
        
    def __call__(self, data, weightId=0):
        return orange.Preprocessor_imputeByLearner(data, learner=self.model)

def bestN(attrMeasures, N=10):
    """ Return best N attributes 
    """
    return attrMeasures[-N:]

def bestP(attrMeasures, P=10):
    """ Return best P percent of attributes
    """
    count = len(attrMeasures)
    return  attrMeasures[-max(count * 100 / P, 1):]

class Preprocessor_featureSelection(orange.Preprocessor):
    """ A preprocessor that runs feature selection using an feature scoring function.
    Arguments:
        
    """
    __new__ = _orange__new()
    __reduce__ = _orange__reduce
    
    bestN = staticmethod(bestN)
    bestP = staticmethod(bestP)
    
    def __init__(self, measure=orange.MeasureAttribute_relief, filter=None, limit=10):
        self.measure = measure
        self.filter = filter if filter is not None else self.bestN
        self.limit = 10
    
    def attrScores(self, data):
        measures = [(self.measure(attr, data), attr) for attr in data.domain.attributes]
        return measures
         
    def __call__(self, data, weightId=None):
        measures = self.attrScores(data)
        attrs = [attr for _, attr in self.filter(measures)]
        domain = orange.Domain(attrs, data.domain.classVar)
        domain.addmetas(data.domain.getmetas())
        return orange.ExampleTable(domain, data)
    
def selectNRandom(examples, N=10):
    """ Select N random examples
    """
    import random
    return random.sample(examples, N)

def selectPRandom(examples, P=10):
    """ Select P percent random examples
    """
    import random
    count = len(examples)
    return random.select(examples, max(count * P / 100, 1))

class Preprocessor_sample(orange.Preprocessor):
    __new__ = _orange__new()
    __reduce__ = _orange__reduce
    
#    @staticmethod
#    def selectNRandom(examples, N=10):
#        """ Select N random examples
#        """
#        import random
#        return random.sample(examples, N)
#    
#    @staticmethod
#    def selectPRandom(examples, P=10):
#        """ Select P percent random examples
#        """
#        import random
#        count = len(examples)
#        return random.select(examples, max(count * P / 100, 1))

    selectNRandom = staticmethod(selectNRandom)
    selectPRandom = staticmethod(selectPRandom)
    
    def __init__(self, filter=None, limit=10):
        self.filter = filter if filter is not None else self.selectNRandom
        self.limit = limit
        
    def __call__(self, data, weightId=None):
        return orange.ExampleTable(data.domain, self.filter(data, self.limit))
    

class Preprocessor_preprocessorList(orange.Preprocessor):
    """ A preprocessor wrapping a sequence of other preprocessors
    """
    
    __new__ = _orange__new()
    __reduce__ = _orange__reduce
    
    def __init__(self, preprocessors=[]):
        self.preprocessors = preprocessors
        
    def __call__(self, data, weightId=None):
        import orange
        hadWeight = hasWeight = weightId is not None
        for preprocessor in self.preprocessors:
            t = preprocessor(data, weightId) if hasWeight else preprocessor(data)
            if isinstance(t, tuple):
                data, weightId = t
                hasWeight = True
            else:
                data = t
        if hadWeight:
            return data, weightId
        else:
            return data

## Preprocessor item editor widgets
class DiscretizeEditor(OWBaseWidget):
    DISCRETIZERS = [("Entropy-MDL discretization", orange.EntropyDiscretization, {}),
                    ("Equal frequency discretization", orange.EquiDistDiscretization, {"numberOfIntervals":3}),
                    ("Equal width discretization", orange.EquiNDiscretization, {"numberOfIntervals":3}),
                    ("Remove continuous attributes", type(None), {})]
    def __init__(self, parent=None):
        OWBaseWidget.__init__(self, parent)
        self.discInd = 0
        self.numberOfIntervals = 3
        self.setLayout(QVBoxLayout())
#        box = OWGUI.widgetBox(self, "Discretize")
        rb = OWGUI.radioButtonsInBox(self, self, "discInd", [], box="Discretize", callback=self.onChange)
        for label, _, _ in self.DISCRETIZERS[:-1]:
            OWGUI.appendRadioButton(rb, self, "discInd", label)
        OWGUI.hSlider(OWGUI.indentedBox(rb), self, "numberOfIntervals", "Num. of intervals (for equal width/frequency)", callback=self.onChange)
        OWGUI.appendRadioButton(rb, self, "discInd", self.DISCRETIZERS[-1][0])
        OWGUI.rubber(rb)
        
    def onChange(self):
        self.emit(SIGNAL("dataChanged"), self.data)
        
    def getDiscretizer(self):
        if self.discInd in [0, 1, 2]:
            name, disc, kwds = self.DISCRETIZERS[self.discInd]
            preprocessor = Preprocessor_discretize(method=disc(**dict([(key, getattr(self, key, val)) for key, val in kwds.items()])))
        elif self.discInd == 3:
            preprocessor = Preprocessor_removeContinuous()
        return preprocessor
    
    def setDiscretizer(self, discretizer):
        self.discType = None
        self.discInd = 3
        for i, (_, disc, kwargs) in enumerate(self.DISCRETIZERS):
            if isinstance(discretizer.method, disc):
                self.discType = disc
                self.discInd = i
                for key, val in kwargs.items():
                    setattr(self, key, getattr(discretizer.method, key, val))
                break
        
    data = pyqtProperty(Preprocessor_discretize,
                        fget=getDiscretizer,
                        fset=setDiscretizer,
                        user=True)
    
class ContinuizeEditor(OWBaseWidget):
    CONTINUIZERS = [("Most frequent is base", orange.DomainContinuizer.FrequentIsBase),
                    ("One attribute per value", orange.DomainContinuizer.NValues),
                    ("Ignore multinomial attributes", orange.DomainContinuizer.Ignore),
                    ("Ignore all discrete attributes", None),
                    ("Treat as ordinal", orange.DomainContinuizer.AsOrdinal),
                    ("Divide by number of values",orange.DomainContinuizer.AsNormalizedOrdinal)]
    
    TREATMENT_TO_IND = dict([(val, i) for i, (_, val) in enumerate(CONTINUIZERS)])
    
    def __init__(self, parent=None):
        OWBaseWidget.__init__(self, parent)
        self.contInd = 0
        self.setLayout(QVBoxLayout())
        
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
        if not isinstance(continuizer, Preprocessor_continuize):
            self.contInd = 3 #Ignore all discrete
            return
        self.contInd = self.TREATMENT_TO_IND.get(continuizer.multinomialTreatment, 3)
    
    data = pyqtProperty(Preprocessor_continuize,
                        fget=getContinuizer,
                        fset=setContinuizer,
                        user=True)
    
class ImputeEditor(OWBaseWidget):
    IMPUTERS = [("Average/Most frequent", orange.MajorityLearner),
                ("Model-based imputer", orange.BayesLearner),
                ("Random values", orange.RandomLearner),
                ("Remove examples with missing values", None)]
    
    def __init__(self, parent):
        OWBaseWidget.__init__(self, parent)
        
        self.methodInd = 0
        self.setLayout(QVBoxLayout())
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
            self.methodInd = dd.get(type(learner), 0)
        elif isinstance(imputer, orange.Preprocessor_dropMissing):
            self.methodInd = 3
            
    data = pyqtProperty(Preprocessor_imputeByLearner,
                        fget=getImputer,
                        fset=setImputer,
                        user=True)
    
class FeatureSelectEditor(OWBaseWidget):
    MEASURES = [("ReliefF", orange.MeasureAttribute_relief),
                ("Information Gain", orange.MeasureAttribute_info),
                ("Gain ratio", orange.MeasureAttribute_gainRatio),
                ("Gini Gain", orange.MeasureAttribute_gini),
                ("Log Ods Ratio", orange.MeasureAttribute_logOddsRatio)]
    FILTERS = [Preprocessor_featureSelection.bestN,
               Preprocessor_featureSelection.bestP]
    
    def __init__(self, parent=None):
        OWBaseWidget.__init__(self, parent)
        
        self.measureInd = 0
        self.selectBy = 0
        self.bestN = 10
        self.bestP = 10
        self.setLayout(QVBoxLayout())
#        box = OWGUI.widgetBox(self, "Feature selection")
        box = OWGUI.radioButtonsInBox(self, self, "selectBy", [], "Feature selection", callback=self.onChange)
        
        OWGUI.comboBox(box, self, "measureInd",  items= [name for (name, _) in self.MEASURES], label="Measure", callback=self.onChange)
        
        hbox1 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb1 = OWGUI.appendRadioButton(box, self, "selectBy", "Best", insertInto=hbox1, callback=self.onChange)
        self.spin1 = OWGUI.spin(OWGUI.widgetBox(hbox1), self, "bestN", 1, 1000, step=1, callback=self.onChange)
        hbox2 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb2 = OWGUI.appendRadioButton(box, self, "selectBy", "Best", insertInto=hbox2, callback=self.onChange)
        self.spin2 = OWGUI.spin(OWGUI.widgetBox(hbox2), self, "bestP", 1, 100, step=1, callback=self.onChange)
        OWGUI.rubber(box)
        
    def onChange(self):
        self.spin1.setDisabled(bool(self.selectBy))
        self.spin2.setDisabled(not bool(self.selectBy))
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
    
    def getFeatureSelection(self):
        return Preprocessor_featureSelection(measure=self.MEASURES[self.measureInd][1],
                                             filter=self.FILTERS[self.selectBy],
                                             limit=self.bestP if self.selectBy  else self.bestN)
    
    data = pyqtProperty(Preprocessor_featureSelection,
                        fget=getFeatureSelection,
                        fset=setFeatureSelection,
                        user=True)
        
class SampleEditor(OWBaseWidget):
    FILTERS = [Preprocessor_sample.selectNRandom,
               Preprocessor_sample.selectPRandom]
    def __init__(self, parent=None):
        OWBaseWidget.__init__(self, parent)
        self.methodInd = 0
        self.sampleN = 100
        self.sampleP = 25
        
        self.setLayout(QVBoxLayout())
        box = OWGUI.radioButtonsInBox(self, self, "methodInd", [], box="Sample", callback=self.onChange)
        
        w1 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb1 = OWGUI.appendRadioButton(box, self, "methodInd", "", insertInto=w1)
        sb1 = OWGUI.spin(w1, self, "sampleN", min=1, max=100000, step=1, callback=self.onChange, posttext="data instances")
        
        w2 = OWGUI.widgetBox(box, orientation="horizontal", margin=0)
        rb2 = OWGUI.appendRadioButton(box, self, "methodInd", "", insertInto=w2)
        sb2 = OWGUI.spin(w2, self, "sampleP", min=1, max=100000, step=1, callback=self.onChange, posttext="% data instances")
        
        OWGUI.rubber(box)
    
    def onChange(self):
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
            
    data = pyqtProperty(Preprocessor_sample,
                        fget=getSampler,
                        fset=setSampler,
                        user=True)
    
def _funcName(func):
    return func.__name__
    
class PreprocessorItemDelegate(QStyledItemDelegate):
        
    #Preprocessor name replacement rules
    REPLACE = {Preprocessor_discretize: "Discretize ({0.method})",
               Preprocessor_removeContinuous: "Discretize (remove continuous)",
               Preprocessor_continuize: "Continuize ({0.multinomialTreatment})",
               Preprocessor_removeDiscrete: "Continuize (remove discrete)",
               Preprocessor_impute: "Impute ({0.model})",
               Preprocessor_imputeByLearner: "Impute ({0.learner})",
               Preprocessor_dropMissing: "Remove missing",
               Preprocessor_featureSelection: "Feature selection ({0.filter}, {0.limit})",
               Preprocessor_sample: "Sample ({0.filter}, {0.limit})",
               orange.EntropyDiscretization: "entropy",
               orange.EquiNDiscretization: "freq, {0.numberOfIntervals}",
               orange.EquiDistDiscretization: "width, {0.numberOfIntervals}",
               orange.RandomLearner: "random",
               orange.BayesLearner: "bayes  model",
               orange.MajorityLearner: "average",
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
        
        text = self.REPLACE.get(type(obj), str(obj))
        if hasattr(text, "__call__"):
            return text(obj)
        else:
            return self.INSERT_RE.sub(replace, text)
        
class NamedTupleItemDelegate(QStyledItemDelegate):
    """ Item delegate for displaying the name of (name, [...]) structured tuples
    """
    def displayText(self, value, locale):
        try:
            obj = value.toPyObject()
            return str(obj[0])
        except Exception, ex:
            return repr(ex)
        
    def createEditor(self, parent, option, index):
        return QLineEdit(parent)
    
    def setEditorData(self, editor, index):
        t = index.data().toPyObject()
        editor.setText(t[0])
        
    def setModelData(self, editor, model, index):
        t = tuple(index.data().toPyObject())
        name = editor.text()
        model.setData(index, QVariant((name,) + t[1:]))
                

class OWPreprocess(OWWidget):
    contextHandlers = {"": PerfectDomainContextHandler("", [""])}
    settingsList = ["allSchemas", "lastSelectedSchemaIndex"]
    
    preprocessors =[("Discretize", Preprocessor_discretize, dict(method=orange.EntropyDiscretization())),
                    ("Continuize", Preprocessor_continuize, {}),
                    ("Impute", Preprocessor_impute, {}),
                    ("Feature selection", Preprocessor_featureSelection, {}),
                    ("Sample", Preprocessor_sample, {})]
    
    EDITORS = {Preprocessor_discretize: DiscretizeEditor,
               Preprocessor_removeContinuous: DiscretizeEditor,
               Preprocessor_continuize: ContinuizeEditor,
               Preprocessor_removeDiscrete: ContinuizeEditor,
               Preprocessor_impute: ImputeEditor,
               Preprocessor_imputeByLearner: ImputeEditor,
               Preprocessor_dropMissing: ImputeEditor,
               Preprocessor_featureSelection: FeatureSelectEditor,
               Preprocessor_sample: SampleEditor,
               type(None): QWidget}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Preprocess")
        
        self.inputs = [("Example Table", ExampleTable, self.setData), ("Learner", orange.Learner, self.setLearner)]
        self.outputs = [("Wrapped Learner", orange.Learner), ("Preprocessed Example Table", ExampleTable), ("Preprocessor", orange.Preprocessor)]
        
        self.autoCommit = False
        self.changedFlag = False
        
        ## [(name, [list of preprocessors], selected preprocessor index), ...]
        self.allSchemas = [("Default" , [Preprocessor_discretize(method=orange.EntropyDiscretization()), Preprocessor_dropMissing()], 0)]
        
        self.lastSelectedSchemaIndex = 0
        
        self.preprocessorsList =  PyListModel([], self)
        
        box = OWGUI.widgetBox(self.controlArea, "Preprocessors", addSpace=True)
        
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
        self.removePreprocessorAction = QAction("-", self)
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
        self.schemaListView = QListView()
        self.schemaListView.setItemDelegate(NamedTupleItemDelegate(self))
        self.schemaListView.setModel(self.schemaList)
        box.layout().addWidget(self.schemaListView)
        
        self.schemaListSelectionModel = ListSingleSelectionModel(self.schemaList, self)
        self.schemaListView.setSelectionMode(QListView.SingleSelection)
        self.schemaListView.setSelectionModel(self.schemaListSelectionModel)
        
        self.connect(self.schemaListSelectionModel, SIGNAL("selectedIndexChanged(QModelIndex)"), self.onSchemaSelection)
        
        self.addSchemaAction = QAction("+", self)
        self.updateSchemaAction = QAction("Update", self)
        self.removeSchemaAction = QAction("-", self)
        
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
            self.connect(action, SIGNAL("triggered()"), lambda pp=pp:self.addPreprocessor(pp(**kwargs)))
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
            self.schemaList._list = self.allSchemas
            self.schemaList.reset()
            self.schemaListSelectionModel.select(self.schemaList.index(self.lastSelectedSchemaIndex), QItemSelectionModel.ClearAndSelect)
        except Exception, ex:
            print repr(ex)
            
    def setData(self, data=None):
        self.data = data
        self.commitIf()
    
    def setLearner(self, learner=None):
        self.learner = learner
        self.commitIf()
    
    def addPreprocessor(self, prep):
        self.preprocessorsList.append(prep)
        self.preprocessorsListSelectionModel.select(self.preprocessorsList.index(len(self.preprocessorsList)-1),
                                                    QItemSelectionModel.ClearAndSelect)
    
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
            self.setActiveSchema(*index.data().toPyObject())
    
    def onAddSchema(self):
        schema = list(self.preprocessorsList)
        self.schemaList.append(("New schema", schema, self.preprocessorsListSelectionModel.selectedRow().row()))
        index = self.schemaList.index(len(self.schemaList) - 1)
        self.schemaListSelectionModel.setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
        self.schemaListView.edit(index)
    
    def onUpdateSchema(self):
        index = self.schemaListSelectionModel.selectedRow()
        i = index.row()
        name, old, index = self.schemaList[i]
        self.schemaList[i] = (name, list(self.preprocessorsList), self.preprocessorsListSelectionModel.selectedRow().row())
    
    def onRemoveSchema(self):
        index = self.schemaListSelectionModel.selectedRow()
        if index.isValid():
            row = index.row()
            del self.schemaList[row]
            newrow = min(max(row - 1, 0), len(self.schemaList) - 1)
            if newrow > -1:
                self.schemaListSelectionModel.select(self.schemaList.index(row), QItemSelectionModel.ClearAndSelect)
                
    def setActiveSchema(self, name, schema, selectedIndex):
        print name, schema, selectedIndex
        self.preprocessorsList._list = list(schema)
        self.preprocessorsList.reset()
        self.preprocessorsListSelectionModel.select(selectedIndex, QItemSelectionModel.ClearAndSelect)
        
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
        self.onUpdateSchema()
        
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
        if self.learner is not None:
            learner = wrap.wrapLearner(self.learner)
            self.send("Wrapped Learner", learner)
            
        self.send("Preprocessor", Preprocessor_preprocessorList(list(self.preprocessorsList)))
        self.changedFlag = False
            
            
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPreprocess()
    w.setData(orange.ExampleTable("../../doc/datasets/iris"))
    w.show()
    app.exec_()
    w.saveSettings()
        
        