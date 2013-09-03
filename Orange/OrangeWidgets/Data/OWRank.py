"""
<name>Rank</name>
<description>Ranks and filters attributes by their relevance.</description>
<icon>icons/Rank.svg</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1102</priority>
"""

from collections import namedtuple
from functools import partial

import pkg_resources

from OWWidget import *

import OWGUI

import Orange
from Orange.feature import scoring
from Orange.regression import earth
from Orange.classification import svm
from Orange.ensemble import forest


def is_discrete(var):
    return isinstance(var, Orange.feature.Discrete)


def is_continuous(var):
    return isinstance(var, Orange.feature.Continuous)


def is_class_discrete(data):
    return is_discrete(data.domain.class_var)


def is_class_continuous(data):
    return is_continuous(data.domain.class_var)


def table(shape, fill=None):
    """ Return a 2D table with shape filed with ``fill``
    """
    return [[fill for j in range(shape[1])] for i in range(shape[0])]


MEASURE_PARAMS = {
    earth.ScoreEarthImportance: [
        {"name": "t",
         "type": int,
         "display_name": "Num. models.",
         "range": (1, 20),
         "default": 10,
         "doc": "Number of models to train for feature scoring."},
        {"name": "terms",
         "type": int,
         "display_name": "Max. num of terms",
         "range": (3, 200),
         "default": 10,
         "doc": "Maximum number of terms in the forward pass"},
        {"name": "degree",
         "type": int,
         "display_name": "Max. term degree",
         "range": (1, 3),
         "default": 2,
         "doc": "Maximum degree of terms included in the model."}
    ],
    scoring.Relief: [
        {"name": "k",
         "type": int,
         "display_name": "Neighbours",
         "range": (1, 20),
         "default": 10,
         "doc": "Number of neighbors to consider."},
        {"name":"m",
         "type": int,
         "display_name": "Examples",
         "range": (20, 100),
         "default": 20,
         "doc": ""}
        ],
    forest.ScoreFeature: [
        {"name": "trees",
         "type": int,
         "display_name": "Num. of trees",
         "range": (20, 100),
         "default": 100,
         "doc": "Number of trees in the random forest."}
        ]
    }


_score_meta = namedtuple(
    "_score_meta",
    ["name",
     "shortname",
     "score",
     "params",
     "supports_regression",
     "supports_classification",
     "handles_discrete",
     "handles_continuous"]
)


class score_meta(_score_meta):
    # Add sensible defaults to __new__
    def __new__(cls, name, shortname, score, params=None,
                supports_regression=True, supports_classification=True,
                handles_continuous=True, handles_discrete=True):
        return _score_meta.__new__(
            cls, name, shortname, score, params,
            supports_regression, supports_classification,
            handles_discrete, handles_continuous
        )


# Default scores.
SCORES = [
    score_meta(
        "ReliefF", "ReliefF", scoring.Relief,
        params=MEASURE_PARAMS[scoring.Relief],
        handles_continuous=True,
        handles_discrete=True),
    score_meta(
        "Information Gain", "Inf. gain", scoring.InfoGain,
        params=None,
        supports_regression=False,
        supports_classification=True,
        handles_continuous=False,
        handles_discrete=True),
    score_meta(
        "Gain Ratio", "Gain Ratio", scoring.GainRatio,
        params=None,
        supports_regression=False,
        handles_continuous=False,
        handles_discrete=True),
    score_meta(
        "Gini Gain", "Gini", scoring.Gini,
        params=None,
        supports_regression=False,
        supports_classification=True,
        handles_continuous=False),
    score_meta(
        "Log Odds Ratio", "log OR", Orange.core.MeasureAttribute_logOddsRatio,
        params=None,
        supports_regression=False,
        handles_continuous=False),
    score_meta(
        "MSE", "MSE", scoring.MSE,
        params=None,
        supports_classification=False,
        handles_continuous=False),
    score_meta(
        "Linear SVM Weights", "SVM weight", svm.ScoreSVMWeights,
        params=None),
    score_meta(
        "Random Forests", "RF", forest.ScoreFeature,
        params=MEASURE_PARAMS[forest.ScoreFeature]),
    score_meta(
        "Earth Importance", "Earth imp.", earth.ScoreEarthImportance,
        params=MEASURE_PARAMS[earth.ScoreEarthImportance],
    )
]

_DEFAULT_SELECTED = set(m.name for m in SCORES[:6])


class MethodParameter(object):
    def __init__(self, name="", type=None, display_name="Parameter",
                 range=None, default=None, doc=""):
        self.name = name
        self.type = type
        self.display_name = display_name
        self.range = range
        self.default = default
        self.doc = doc


def measure_parameters(measure):
    return [MethodParameter(**args) for args in (measure.params or [])]


def param_attr_name(measure, param):
    """Name of the OWRank widget's member where the parameter is stored.
    """
    return "param_" + measure.__name__ + "_" + param.name


def drop_exceptions(iterable, exceptions=(Exception,)):
    iterable = iter(iterable)
    while True:
        try:
            yield next(iterable)
        except StopIteration:
            raise
        except BaseException as ex:
            if not isinstance(ex, exceptions):
                raise


def load_ep_drop_exceptions(entry_point):
    for ep in pkg_resources.iter_entry_points(entry_point):
        try:
            yield ep.load()
        except Exception:
            log = logging.getLogger(__name__)
            log.debug("", exc_info=True)


def all_measures():
    iter_ep = load_ep_drop_exceptions("orange.widgets.feature_score")
    scores = [m for m in iter_ep if isinstance(m, score_meta)]
    return SCORES + scores


class OWRank(OWWidget):
    settingsList = [
        "nDecimals", "nIntervals", "sortBy", "nSelected",
        "selectMethod", "autoApply", "showDistributions",
        "distColorRgb"
    ]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Rank")

        self.inputs = [("Data", ExampleTable, self.setData)]
        self.outputs = [("Reduced Data", ExampleTable, Default + Single)]

        self.nDecimals = 3
        self.nIntervals = 4
        self.sortBy = 2
        self.selectMethod = 2
        self.nSelected = 5
        self.autoApply = True
        self.showDistributions = 1
        self.distColorRgb = (220, 220, 220, 255)
        self.distColor = QColor(*self.distColorRgb)

        self.all_measures = all_measures()

        self.selectedMeasures = dict(
            [(name, True) for name in _DEFAULT_SELECTED] +
            [(m.name, False)
             for m in self.all_measures[len(_DEFAULT_SELECTED):]]
        )

        self.data = None

        self.methodParamAttrs = []
        for m in self.all_measures:
            params = measure_parameters(m)
            for p in params:
                name_mangled = param_attr_name(m.score, p)
                setattr(self, name_mangled, p.default)
                self.methodParamAttrs.append(name_mangled)

        self.settingsList = self.settingsList + self.methodParamAttrs

        self.loadSettings()

        self.discMeasures = [m for m in self.all_measures
                             if m.supports_classification]
        self.contMeasures = [m for m in self.all_measures
                             if m.supports_regression]

        self.stackedLayout = QStackedLayout()
        self.stackedLayout.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = OWGUI.widgetBox(self.controlArea, margin=0,
                                             orientation=self.stackedLayout,
                                             addSpace=True)

        # Discrete class scoring
        discreteBox = OWGUI.widgetBox(self.stackedWidget, "Scoring",
                                      addSpace=False,
                                      addToLayout=False)
        self.stackedLayout.addWidget(discreteBox)

        # Continuous class scoring
        continuousBox = OWGUI.widgetBox(self.stackedWidget, "Scoring",
                                        addSpace=False,
                                        addToLayout=False)
        self.stackedLayout.addWidget(continuousBox)

        def measure_control(container, measure):
            """Construct UI control for `measure` (measure_meta instance).
            """
            name = measure.name
            params = measure_parameters(measure)
            if params:
                hbox = OWGUI.widgetBox(container, orientation="horizontal")
                OWGUI.checkBox(hbox, self.selectedMeasures, name, name,
                               callback=partial(self.measuresSelectionChanged,
                                                measure),
                               tooltip="Enable " + name)

                smallWidget = OWGUI.SmallWidgetLabel(
                    hbox, pixmap=1, box=name + " Parameters",
                    tooltip="Show " + name + "Parameters")

                for param in params:
                    OWGUI.spin(smallWidget.widget, self,
                               param_attr_name(measure.score, param),
                               param.range[0], param.range[-1],
                               label=param.display_name,
                               tooltip=param.doc,
                               callback=partial(
                                   self.measureParamChanged, measure, param),
                               callbackOnReturn=True)

                OWGUI.button(smallWidget.widget, self, "Load defaults",
                             callback=partial(self.loadMeasureDefaults,
                                              measure))
            else:
                OWGUI.checkBox(container, self.selectedMeasures, name, name,
                               callback=partial(self.measuresSelectionChanged,
                                                measure),
                               tooltip="Enable " + name)

        for measure in self.all_measures:
            if measure.supports_classification:
                measure_control(discreteBox, measure)

            if measure.supports_regression:
                measure_control(continuousBox, measure)

        OWGUI.comboBox(
            discreteBox, self, "sortBy", label="Sort by  ",
            items=["No Sorting", "Attribute Name", "Number of Values"] +
                  [m.name for m in self.discMeasures],
            orientation=0, valueType=int,
            callback=self.sortingChanged)

        OWGUI.comboBox(
            continuousBox, self, "sortBy", label="Sort by  ",
            items=["No Sorting", "Attribute Name", "Number of Values"] +
                  [m.name for m in self.contMeasures],
            orientation=0, valueType=int,
            callback=self.sortingChanged)

        box = OWGUI.widgetBox(self.controlArea, "Discretization",
                              addSpace=True)
        OWGUI.spin(box, self, "nIntervals", 2, 20,
                   label="Intervals: ",
                   orientation=0,
                   tooltip="Disctetization for measures which cannot score "
                           "continuous attributes.",
                   callback=self.discretizationChanged,
                   callbackOnReturn=True)

        box = OWGUI.widgetBox(self.controlArea, "Precision", addSpace=True)
        OWGUI.spin(box, self, "nDecimals", 1, 6, label="No. of decimals: ",
                   orientation=0, callback=self.decimalsChanged)

        box = OWGUI.widgetBox(self.controlArea, "Score bars",
                              orientation="horizontal", addSpace=True)
        self.cbShowDistributions = OWGUI.checkBox(
            box, self, "showDistributions", 'Enable',
            callback=self.showDistributionsChanged
        )

        OWGUI.rubber(box)
        box = OWGUI.widgetBox(box, orientation="horizontal")
        wl = OWGUI.widgetLabel(box, "Color: ")
        OWGUI.separator(box)
        self.colButton = OWGUI.toolButton(
            box, self, callback=self.changeColor, width=20, height=20,
            debuggingEnabled=0)

        self.cbShowDistributions.disables.extend([wl, self.colButton])
        self.cbShowDistributions.makeConsistent()

        selMethBox = OWGUI.widgetBox(self.controlArea, "Select attributes",
                                     addSpace=True)
        self.clearButton = OWGUI.button(selMethBox, self, "Clear",
                                        callback=self.clearSelection)
        self.clearButton.setDisabled(True)

        buttonGrid = QGridLayout()
        selMethRadio = OWGUI.radioButtonsInBox(
            selMethBox, self, "selectMethod", [],
            callback=self.selectMethodChanged)

        b1 = OWGUI.appendRadioButton(
            selMethRadio, self, "selectMethod", "All",
            insertInto=selMethRadio,
            callback=self.selectMethodChanged,
            addToLayout=False)

        b2 = OWGUI.appendRadioButton(
            selMethRadio, self, "selectMethod", "Manual",
            insertInto=selMethRadio,
            callback=self.selectMethodChanged,
            addToLayout=False)

        b3 = OWGUI.appendRadioButton(
            selMethRadio, self, "selectMethod", "Best ranked",
            insertInto=selMethRadio,
            callback=self.selectMethodChanged,
            addToLayout=False)

        spin = OWGUI.spin(OWGUI.widgetBox(selMethRadio, addToLayout=False),
                          self, "nSelected", 1, 100, orientation=0,
                          callback=self.nSelectedChanged)
        buttonGrid.addWidget(b1, 0, 0)
        buttonGrid.addWidget(b2, 1, 0)
        buttonGrid.addWidget(b3, 2, 0)
        buttonGrid.addWidget(spin, 2, 1)
        selMethRadio.layout().addLayout(buttonGrid)
        OWGUI.separator(selMethBox)

        applyButton = OWGUI.button(
            selMethBox, self, "Commit", callback=self.apply, default=True)
        autoApplyCB = OWGUI.checkBox(
            selMethBox, self, "autoApply", "Commit automatically")
        OWGUI.setStopper(
            self, applyButton, autoApplyCB, "dataChanged", self.apply)

        OWGUI.rubber(self.controlArea)

        # Discrete and continuous table views are stacked
        self.ranksViewStack = QStackedLayout()
        self.mainArea.layout().addLayout(self.ranksViewStack)

        self.discRanksView = QTableView()
        self.ranksViewStack.addWidget(self.discRanksView)
        self.discRanksView.setSelectionBehavior(QTableView.SelectRows)
        self.discRanksView.setSelectionMode(QTableView.MultiSelection)
        self.discRanksView.setSortingEnabled(True)

        self.discRanksModel = QStandardItemModel(self)
        self.discRanksModel.setHorizontalHeaderLabels(
            ["Attribute", "#"] + [m.shortname for m in self.discMeasures]
        )

        self.discRanksProxyModel = MySortProxyModel(self)
        self.discRanksProxyModel.setSourceModel(self.discRanksModel)
        self.discRanksView.setModel(self.discRanksProxyModel)

        self.discRanksView.setColumnWidth(1, 20)
        self.discRanksView.sortByColumn(2, Qt.DescendingOrder)
        self.connect(self.discRanksView.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     self.onSelectionChanged)
        self.connect(self.discRanksView,
                     SIGNAL("pressed(const QModelIndex &)"),
                     self.onSelectItem)
        self.connect(self.discRanksView.horizontalHeader(),
                     SIGNAL("sectionClicked(int)"),
                     self.headerClick)

        self.contRanksView = QTableView()
        self.ranksViewStack.addWidget(self.contRanksView)
        self.contRanksView.setSelectionBehavior(QTableView.SelectRows)
        self.contRanksView.setSelectionMode(QTableView.MultiSelection)
        self.contRanksView.setSortingEnabled(True)

        self.contRanksModel = QStandardItemModel(self)
        self.contRanksModel.setHorizontalHeaderLabels(
            ["Attribute", "#"] + [m.shortname for m in self.contMeasures]
        )

        self.contRanksProxyModel = MySortProxyModel(self)
        self.contRanksProxyModel.setSourceModel(self.contRanksModel)
        self.contRanksView.setModel(self.contRanksProxyModel)

        self.discRanksView.setColumnWidth(1, 20)
        self.contRanksView.sortByColumn(2, Qt.DescendingOrder)
        self.connect(self.contRanksView.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     self.onSelectionChanged)
        self.connect(self.contRanksView,
                     SIGNAL("pressed(const QModelIndex &)"),
                     self.onSelectItem)
        self.connect(self.contRanksView.horizontalHeader(),
                     SIGNAL("sectionClicked(int)"),
                     self.headerClick)

        # Switch the current view to Discrete
        self.switchRanksMode(0)
        self.resetInternals()
        self.updateDelegates()
        self.updateVisibleScoreColumns()

        self.resize(690, 500)
        self.updateColor()

        self.measure_scores = table((len(self.measures), 0), None)

    def switchRanksMode(self, index):
        """
        Switch between discrete/continuous mode
        """
        self.ranksViewStack.setCurrentIndex(index)
        self.stackedLayout.setCurrentIndex(index)

        if index == 0:
            self.ranksView = self.discRanksView
            self.ranksModel = self.discRanksModel
            self.ranksProxyModel = self.discRanksProxyModel
            self.measures = self.discMeasures
        else:
            self.ranksView = self.contRanksView
            self.ranksModel = self.contRanksModel
            self.ranksProxyModel = self.contRanksProxyModel
            self.measures = self.contMeasures

        self.updateVisibleScoreColumns()

    def setData(self, data):
        self.error(0)
        self.resetInternals()
        self.data = self.isDataWithClass(data) and data or None
        if self.data:
            attrs = self.data.domain.attributes
            self.usefulAttributes = \
                [attr for attr in attrs
                 if is_discrete(attr) or is_continuous(attr)]

            if is_class_continuous(self.data):
                self.switchRanksMode(1)
            elif is_class_discrete(self.data):
                self.switchRanksMode(0)
            else:
                # String or other.
                self.error(0, "Cannot handle class variable type %r" %
                           type(self.data.domain.class_var).__name__)

            self.ranksModel.setRowCount(len(attrs))
            for i, a in enumerate(attrs):
                if is_discrete(a):
                    v = len(a.values)
                else:
                    v = "C"
                item = PyStandardItem()
                item.setData(QVariant(v), Qt.DisplayRole)
                self.ranksModel.setItem(i, 1, item)
                item = PyStandardItem(a.name)
                item.setData(QVariant(i), OWGUI.SortOrderRole)
                self.ranksModel.setItem(i, 0, item)

            self.ranksView.resizeColumnToContents(1)

            self.measure_scores = table((len(self.measures),
                                         len(attrs)), None)
            self.updateScores()
            if is_class_discrete(self.data):
                self.setLogORTitle()
            self.ranksView.setSortingEnabled(self.sortBy > 0)

        self.applyIf()

    def updateScores(self, measuresMask=None):
        """
        Update the current computed scores.

        If `measuresMask` is given it must be an list of bool values
        indicating what measures should be recomputed.

        """
        if not self.data:
            return

        measures = self.measures
        # Invalidate all warnings
        self.warning(range(max(len(self.discMeasures),
                               len(self.contMeasures))))

        if measuresMask is None:
            # Update all selected measures
            measuresMask = [self.selectedMeasures.get(m.name)
                            for m in measures]

        for measure_index, (meas, mask) in enumerate(zip(measures, measuresMask)):
            if not mask:
                continue

            params = measure_parameters(meas)
            estimator = meas.score()
            if params:
                for p in params:
                    setattr(estimator, p.name,
                            getattr(self, param_attr_name(meas.score, p)))

            if not meas.handles_continuous:
                data = self.getDiscretizedData()
                attr_map = data.attrDict
                data = self.data
            else:
                attr_map, data = {}, self.data

            attr_scores = []
            for i, attr in enumerate(data.domain.attributes):
                attr = attr_map.get(attr, attr)
                s = None
                if attr is not None:
                    try:
                        s = estimator(attr, data)
                    except Exception, ex:
                        self.warning(measure_index, "Error evaluating %r: %r" %
                                     (meas.name, str(ex)))
                    if meas.name == "Log Odds Ratio" and s is not None:
                        # Hardcoded values returned by log odds measure
                        if s == -999999:
                            attr = u"-\u221E"
                        elif s == 999999:
                            attr = u"\u221E"
                        else:
                            attr = attr.values[1]
                        s = ("%%.%df" % self.nDecimals + " (%s)") % (s, attr)
                attr_scores.append(s)
            self.measure_scores[measure_index] = attr_scores

        self.updateRankModel(measuresMask)
        self.ranksProxyModel.invalidate()

        if self.selectMethod in [0, 2]:
            self.autoSelection()

    def updateRankModel(self, measuresMask=None):
        """
        Update the rankModel.
        """
        values = []
        for i, scores in enumerate(self.measure_scores):
            values_one = []
            for j, s in enumerate(scores):
                if isinstance(s, float):
                    values_one.append(s)
                else:
                    values_one.append(None)
                item = self.ranksModel.item(j, i + 2)
                if not item:
                    item = PyStandardItem()
                    self.ranksModel.setItem(j, i + 2, item)
                item.setData(QVariant(s), Qt.DisplayRole)
            values.append(values_one)

        for i, vals in enumerate(values):
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                vmin, vmax = min(valid_vals), max(valid_vals)
                for j, v in enumerate(vals):
                    if v is not None:
                        # Set the bar ratio role for i-th measure.
                        ratio = float((v - vmin) / ((vmax - vmin) or 1))
                        item = self.ranksModel.item(j, i + 2)
                        if self.showDistributions:
                            item.setData(QVariant(ratio), OWGUI.BarRatioRole)
                        else:
                            item.setData(QVariant(), OWGUI.BarRatioRole)

        self.ranksView.resizeColumnsToContents()
        self.ranksView.setColumnWidth(1, 20)
        self.ranksView.resizeRowsToContents()

    def showDistributionsChanged(self):
        # This should be handled by the delegates only (must always set the BarRatioRole
        self.updateRankModel()
        # Need to update the selection
        self.autoSelection()

    def changeColor(self):
        color = QColorDialog.getColor(self.distColor, self)
        if color.isValid():
            self.distColorRgb = color.getRgb()
            self.updateColor()

    def updateColor(self):
        self.distColor = QColor(*self.distColorRgb)
        w = self.colButton.width() - 8
        h = self.colButton.height() - 8
        pixmap = QPixmap(w, h)
        painter = QPainter()
        painter.begin(pixmap)
        painter.fillRect(0, 0, w, h, QBrush(self.distColor))
        painter.end()
        self.colButton.setIcon(QIcon(pixmap))
        self.updateDelegates()

    def resetInternals(self):
        self.data = None
        self.discretizedData = None
        self.attributeOrder = []
        self.selected = []
        self.measured = {}
        self.usefulAttributes = []
        self.dataChanged = False
        self.lastSentAttrs = None
        self.ranksModel.setRowCount(0)

    def onSelectionChanged(self, *args):
        """
        Called when the ranks view selection changes.
        """
        selected = self.selectedAttrs()
        self.clearButton.setEnabled(bool(selected))
        self.applyIf()

    def onSelectItem(self, index):
        """
        Called when the user selects/unselects an item in the table view.
        """
        self.selectMethod = 1  # Manual
        self.clearButton.setEnabled(bool(self.selectedAttrs()))
        self.applyIf()

    def clearSelection(self):
        self.ranksView.selectionModel().clear()

    def selectMethodChanged(self):
        if self.selectMethod in [0, 2]:
            self.autoSelection()

    def nSelectedChanged(self):
        self.selectMethod = 2
        self.selectMethodChanged()

    def getDiscretizedData(self):
        if not self.discretizedData:
            discretizer = Orange.feature.discretization.EqualFreq(
                numberOfIntervals=self.nIntervals)
            contAttrs = [attr for attr in self.data.domain.attributes
                         if is_continuous(attr)]
            at = []
            attrDict = {}
            for attri in contAttrs:
                try:
                    nattr = discretizer(attri, self.data)
                    at.append(nattr)
                    attrDict[attri] = nattr
                except:
                    pass
            domain = Orange.data.Domain(at, self.data.domain.class_var)
            self.discretizedData = self.data.translate(domain)
            self.discretizedData.setattr("attrDict", attrDict)
        return self.discretizedData

    def discretizationChanged(self):
        self.discretizedData = None
        self.updateScores([not m.handles_continuous for m in self.measures])
        self.autoSelection()

    def measureParamChanged(self, measure, param=None):
        index = self.measures.index(measure)
        mask = [i == index for i, _ in enumerate(self.measures)]
        self.updateScores(mask)

    def loadMeasureDefaults(self, measure):
        params = measure_parameters(measure)
        for i, p in enumerate(params):
            setattr(self, param_attr_name(measure.score, p), p.default)
        self.measureParamChanged(measure)

    def autoSelection(self):
        selModel = self.ranksView.selectionModel()
        rowCount = self.ranksModel.rowCount()
        columnCount = self.ranksModel.columnCount()
        model = self.ranksProxyModel
        if self.selectMethod == 0:
            selection = QItemSelection(
                model.index(0, 0),
                model.index(rowCount - 1, columnCount - 1)
            )
            selModel.select(selection, QItemSelectionModel.ClearAndSelect)
        if self.selectMethod == 2:
            nSelected = min(self.nSelected, rowCount)
            selection = QItemSelection(
                model.index(0, 0),
                model.index(nSelected - 1, columnCount - 1)
            )
            selModel.select(selection, QItemSelectionModel.ClearAndSelect)

    def headerClick(self, index):
        self.sortBy = index + 1
        if not self.ranksView.isSortingEnabled():
            # The sorting is disabled ("No sorting|" selected by user)
            self.sortingChanged()

        if index > 1 and self.selectMethod == 2:
            # Reselect the top ranked attributes
            self.autoSelection()
        self.sortBy = index + 1
        return

    def sortingChanged(self):
        """
        Sorting was changed by user (through the Sort By combo box.)
        """
        self.updateSorting()
        self.autoSelection()

    def updateSorting(self):
        """
        Update the sorting of the model/view.
        """
        self.ranksProxyModel.invalidate()
        if self.sortBy == 0:
            self.ranksProxyModel.setSortRole(OWGUI.SortOrderRole)
            self.ranksProxyModel.sort(0, Qt.DescendingOrder)
            self.ranksView.setSortingEnabled(False)

        else:
            self.ranksProxyModel.setSortRole(Qt.DisplayRole)
            self.ranksView.sortByColumn(self.sortBy - 1, Qt.DescendingOrder)
            self.ranksView.setSortingEnabled(True)

    def setLogORTitle(self):
        var = self.data.domain.classVar
        if len(var.values) == 2:
            title = "log OR (for %r)" % var.values[1][:10]
        else:
            title = "log OR"

        index = [m.name for m in self.discMeasures].index("Log Odds Ratio")

        item = PyStandardItem(title)
        self.ranksModel.setHorizontalHeaderItem(index + 2, item)

    def measuresSelectionChanged(self, measure=None):
        """Measure selection has changed. Update column visibility.
        """
        if measure is None:
            # Update all scores
            measuresMask = None
        else:
            # Update scores for shown column if they are not yet computed.
            shown = self.selectedMeasures.get(measure.name, False)
            index = self.measures.index(measure)
            if all(s is None for s in self.measure_scores[index]) and shown:
                measuresMask = [m == measure for m in self.measures]
            else:
                measuresMask = [False] * len(self.measures)
        self.updateScores(measuresMask)

        self.updateVisibleScoreColumns()

    def updateVisibleScoreColumns(self):
        """
        Update the visible columns of the scores view.
        """
        for i, measure in enumerate(self.measures):
            shown = self.selectedMeasures.get(measure.name)
            self.ranksView.setColumnHidden(i + 2, not shown)

    def sortByColumn(self, col):
        if col < 2:
            self.sortBy = 1 + col
        else:
            self.sortBy = 3 + self.selectedMeasures[col - 2]
        self.sortingChanged()

    def decimalsChanged(self):
        self.updateDelegates()
        self.ranksView.resizeColumnsToContents()

    def updateDelegates(self):
        self.contRanksView.setItemDelegate(
            OWGUI.ColoredBarItemDelegate(
                self,
                decimals=self.nDecimals,
                color=self.distColor)
        )

        self.discRanksView.setItemDelegate(
            OWGUI.ColoredBarItemDelegate(
                self,
                decimals=self.nDecimals,
                color=self.distColor)
        )

    def sendReport(self):
        self.reportData(self.data)
        self.reportRaw(OWReport.reportTable(self.ranksView))

    def applyIf(self):
        if self.autoApply:
            self.apply()
        else:
            self.dataChanged = True

    def apply(self):
        selected = self.selectedAttrs()
        if not self.data or not selected:
            self.send("Reduced Data", None)
        else:
            domain = Orange.data.Domain(selected, self.data.domain.classVar)
            domain.addmetas(self.data.domain.getmetas())
            data = Orange.data.Table(domain, self.data)
            self.send("Reduced Data", data)
        self.dataChanged = False

    def selectedAttrs(self):
        if self.data:
            inds = self.ranksView.selectionModel().selectedRows(0)
            source = self.ranksProxyModel.mapToSource
            inds = map(source, inds)
            inds = [ind.row() for ind in inds]
            return [self.data.domain.attributes[i] for i in inds]
        else:
            return []


class PyStandardItem(QStandardItem):
    """A StandardItem subclass for python objects.
    """
    def __init__(self, *args):
        QStandardItem.__init__(self, *args)
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def __lt__(self, other):
        my = self.data(Qt.DisplayRole).toPyObject()
        other = other.data(Qt.DisplayRole).toPyObject()
        if my is None:
            return True
        return my < other


class MySortProxyModel(QSortFilterProxyModel):
    def headerData(self, section, orientation, role):
        """ Don't map headers.
        """
        source = self.sourceModel()
        return source.headerData(section, orientation, role)

    def lessThan(self, left, right):
        role = self.sortRole()
        left = left.data(role).toPyObject()
        right = right.data(role).toPyObject()
        return left < right


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWRank()
    ow.setData(Orange.data.Table("wine.tab"))
    ow.setData(Orange.data.Table("zoo.tab"))
    ow.setData(Orange.data.Table("servo.tab"))
    ow.setData(Orange.data.Table("iris.tab"))
#    ow.setData(orange.ExampleTable("auto-mpg.tab"))
    ow.show()
    a.exec_()
    ow.saveSettings()
