from collections import namedtuple

import OWGUI
from OWWidget import *

import Orange
import Orange.data
import Orange.feature
from Orange import distance


NAME = "Distances"
DESCRIPTION = "Computes a distance matrix from a set of data instances."
ICON = "icons/Distance.svg"
PRIORITY = 1300

INPUTS = [("Data", Orange.data.Table, "set_data")]
OUTPUTS = [("Distances", Orange.misc.SymMatrix)]


Metric = namedtuple(
    "Metric",
    ["name", "constructor", "normalize"]
)

METRICS = [
    Metric("Euclidean", distance.Euclidean, True),
    Metric("Pearson Correlation", distance.PearsonR, False),
    Metric("Absolute Pearson Correlation", distance.PearsonRAbsolute, False),
    Metric("Spearman Rank Correlation", distance.SpearmanR, False),
    Metric("Absolute Spearman Rank Correlation",
           distance.SpearmanRAbsolute, False),
    Metric("Manhattan", distance.Manhattan, True),
    Metric("Hamming", distance.Hamming, False),
    Metric("Relief", distance.Relief, False)
]


class _CanceledError(Exception):
    pass


class OWDistances(OWWidget):
    settingsList = ["axis", "metric", "normalize", "label", "autocommit"]
    contextHandlers = {"": DomainContextHandler("", ["label"])}

    def __init__(self, parent=None, signalManager=None, title="Distances"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False, resizingEnabled=False)

        self.axis = 0
        self.metric = 0
        self.normalize = True
        self.label = 0
        self.autocommit = False
        self._canceled = False
        self.loadSettings()

        self.labels = []
        self.data = None
        self.data_t = None
        self.matrix = None
        self.metric = min(max(self.metric, 0), len(METRICS) - 1)
        self._invalidated = False

        box = OWGUI.widgetBox(self.controlArea, "Distances between",
                              addSpace=True)
        OWGUI.radioButtonsInBox(
            box, self, "axis", ["rows", "columns"],
            callback=self.distAxisChanged
        )

        box = OWGUI.widgetBox(self.controlArea, "Distance Metric",
                              addSpace=True)

        cb = OWGUI.comboBox(
            box, self, "metric",
            items=[m.name for m in METRICS],
            tooltip=("Choose metric to measure pairwise distances between "
                     "examples."),
            callback=self.distMetricChanged,
            valueType=str
        )

        cb.setMinimumWidth(170)

        box = OWGUI.widgetBox(box, "Settings", flat=True)

        self.normalizeCB = OWGUI.checkBox(
            box, self, "normalize",
            "Normalized",
            callback=self.distMetricChanged
        )

        self.normalizeCB.setEnabled(self.metric in [0, 3])

        self.labelCombo = OWGUI.comboBox(
            self.controlArea, self, "label",
            box="Label",
            items=[],
            tooltip="Attribute used for matrix item labels",
            callback=self.invalidate,
        )
        self.labelCombo.setDisabled(True)
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Commit")
        cb = OWGUI.checkBox(box, self, "autocommit", "Commit on any change")
        b = OWGUI.button(box, self, "Commit", callback=self.commit,
                         default=True)
        OWGUI.setStopper(self, b, cb, "_invalidated", self.commit)

    def sendReport(self):
        metric = METRICS[self.metric]
        if metric.normalize and self.normalize:
            metric = "Normalized " + metric.name
        else:
            metric = metric.name

        if self.axis == 0:
            axis = "rows"
        else:
            axis = "columns"

        self.reportSettings("Settings",
                            [("By", axis),
                             ("Metrics", metric)])
        self.reportData(self.data)

    def distMetricChanged(self):
        self.normalizeCB.setEnabled(METRICS[self.metric].normalize)
        self.matrix = None

        if self.data is not None:
            self.invalidate()

    def distAxisChanged(self):
        self._updateLabelCB()
        self.matrix = None

        if self.data is not None:
            self.invalidate()

    def invalidate(self):
        if self.autocommit:
            self.commit()
        else:
            self._invalidated = True

    def _updateLabelCB(self):
        self.labelCombo.clear()
        self.labels = []
        if not self.data:
            self.labelCombo.setEnabled(False)
            return

        domain = self.data.domain
        attributes = domain.attributes

        if self.axis:
            self.labelCombo.addItem("Column Id")
            self.labelCombo.insertSeparator(1)
            attributes = self.data.domain.attributes
            keys = reduce(
                set.union,
                [attr.attributes.keys() for attr in attributes],
                set()
            )
            keys = sorted(keys)
            self.labels = keys
            self.labelCombo.addItems(sorted(keys))
            self.labelCombo.setCurrentIndex(self.label)
        else:
            variables = list(domain.getmetas().values()) + \
                        list(domain.variables)
            names = [v.name for v in variables]
            self.labels = names
            self.labelCombo.addItems(names)

        self.label = min(self.label, len(self.labels))
        self.labelCombo.setCurrentIndex(self.label)

        # TODO: ctx. dep. label index
        self.labelCombo.setEnabled(True)

    def labeledItems(self):
        if not self.data:
            return

        if self.axis:
            if self.label == 0:
                labels = [v.name for v in self.data.domain.attributes]
            else:
                key = unicode(self.labelCombo.currentText())
                labels = [v.attributes.get(key, "")
                          for v in self.data.domain.attributes]
            items = labels
        else:
            vname = self.labels[self.label]
            labels = [str(inst[vname]) for inst in self.data]
            output_data = Orange.data.Table(self.data)
            for inst, name in zip(output_data, labels):
                inst.name = name

            items = output_data

        return items

    def progressBarSet(self, value):
        if self._canceled:
            raise _CanceledError

        super(OWDistances, self).progressBarSet(value)

    def computeMatrix(self):
        self.warning(1)

        if not self.data:
            self.matrix = None
            return None

        data = self.data
        domain = data.domain
        metric = METRICS[self.metric]

        constructor = metric.constructor()

        if self.axis:
            if domain_has_discrete_attributes(domain):
                self.warning(1, "Input domain contains discrete attributes.")

            if self.data_t is None:
                self.data_t = transpose(self.data)

            data = self.data_t
        else:
            data = self.data

        if metric.normalize:
            constructor.normalize = bool(self.normalize)

        self.error(0)
        self.progressBarInit()
        try:
            matrix = distance.distance_matrix(
                data, constructor, self.progressBarSet
            )
        except Orange.core.KernelException, ex:
            self.error(0, "Could not create distance matrix! %s" % str(ex))
            matrix = None

        self.progressBarFinished()

        if matrix:
            matrix.setattr('items', data)

        self.matrix = matrix
        return matrix

    def commit(self):
        if self.matrix is None:
            try:
                self.matrix = self.computeMatrix()
            except _CanceledError:
                return

        matrix = self.matrix
        if matrix is not None:
            items = self.labeledItems()
            matrix.setattr("items", items)

        self.send("Distances", matrix)
        self._invalidated = False

    def set_data(self, data):
        if data and len(data.domain.attributes):
            self.data = data
            self.data_t = None
            self.matrix = None
            self._updateLabelCB()
            self.commit()
        else:
            self.data = None
            self.data_t = None
            self.matrix = None
            self.labelCombo.clear()
            self.labels = []
            self.send("Distances", None)

    def onDeleteWidget(self):
        super(OWDistances, self).onDeleteWidget()
        self._canceled = True


def domain_has_discrete_attributes(domain):
    return any(isinstance(attr, Orange.feature.Discrete)
               for attr in domain.attributes)


def domain_has_continuous_attributes(domain):
    return any(isinstance(attr, Orange.feature.Continuous)
               for attr in domain.attributes)


def transpose(data):
    """
    Transpose the `data` (works on attributes part only).
    """
    domain = data.domain
    N = len(data)
    if not all(isinstance(attr,
                          (Orange.feature.Continuous, Orange.feature.Discrete))
               for attr in domain.attributes):
        raise TypeError

    trans_domain = Orange.data.Domain(
        [Orange.feature.Continuous("F%i" % (i + 1)) for i in range(N)]
    )
    X, = data.to_numpy_MA("A")
    return Orange.data.Table(trans_domain, X.T)


if __name__ == "__main__":
    data = Orange.data.Table('glass')
    a = QApplication(sys.argv)
    ow = OWDistances()
    ow.show()
    ow.set_data(data)
    a.exec_()
    ow.saveSettings()
