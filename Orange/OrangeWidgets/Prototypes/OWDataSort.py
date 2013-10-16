from operator import itemgetter

from PyQt4.QtGui import (
    QListView, QStyledItemDelegate, QApplication, QComboBox
)

from PyQt4.QtCore import Qt

import Orange

from OWItemModels import VariableListModel
from OWWidget import OWWidget
from OWContexts import DomainContextHandler
import OWGUI


NAME = "Data Sort"
DESCRIPTION = "Sort instances in a data table"
ICON = "icons/DataSort.svg"


INPUTS = [("Data", Orange.data.Table, "setData")]
OUTPUTS = [("Data", Orange.data.Table)]


SortOrderRole = next(OWGUI.OrangeUserRole)


def toInt(variant):
    if type(variant).__name__ == "QVariant":
        value, ok = variant.toInt()
        if ok:
            return value
        else:
            raise TypeError()
    else:
        return int(variant)


def toSortOrder(variant):
    try:
        order = toInt(variant)
    except TypeError:
        order = Qt.AscendingOrder
    return order


class SortParamDelegate(QStyledItemDelegate):

    def initStyleOption(self, option, index):
        QStyledItemDelegate.initStyleOption(self, option, index)

        order = toSortOrder(index.data(SortOrderRole))

        if order == Qt.AscendingOrder:
            option.text = option.text + " (Ascending)"
        else:
            option.text = option.text + " (Descending)"

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        # Note: Qt.AscendingOrder == 0 and Qt.DescendingOrder == 1
        editor.addItems(["Ascending", "Descending"])
        editor.setFocusPolicy(Qt.StrongFocus)
        return editor

    def setEditorData(self, editor, index):
        order = toSortOrder(index.data(SortOrderRole))
        editor.setCurrentIndex(order)
        editor.activated.connect(lambda i: self._commit(editor))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex(), SortOrderRole)

    def _commit(self, editor):
        # Notify the view the editor finished.
        self.commitData.emit(editor)


class OWDataSort(OWWidget):
    contextHandlers = {
        "": DomainContextHandler(
            "", ["sortroles"]
        )
    }
    settingsList = ["autoCommit"]

    def __init__(self, parent=None, signalManger=None, title="Data Sort"):
        super(OWDataSort, self).__init__(parent, signalManger, title,
                                         wantMainArea=False)

        #: Mapping (feature.name, feature.var_type) to (sort_index, sort_order)
        #: where sirt index is the position of the feature in the sortByModel
        #: and sort_order the Qt.SortOrder flag
        self.sortroles = {}

        self.autoCommit = False
        self._outputChanged = False

        box = OWGUI.widgetBox(self.controlArea, "Sort By Features")
        self.sortByView = QListView()
        self.sortByView.setItemDelegate(SortParamDelegate(self))
        self.sortByView.setSelectionMode(QListView.ExtendedSelection)
        self.sortByView.setDragDropMode(QListView.DragDrop)
        self.sortByView.setDefaultDropAction(Qt.MoveAction)
        self.sortByView.viewport().setAcceptDrops(True)

        self.sortByModel = VariableListModel(
            flags=Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                  Qt.ItemIsDragEnabled | Qt.ItemIsEditable
        )
        self.sortByView.setModel(self.sortByModel)

        box.layout().addWidget(self.sortByView)

        box = OWGUI.widgetBox(self.controlArea, "Unused Features")
        self.unusedView = QListView()
        self.unusedView.setSelectionMode(QListView.ExtendedSelection)
        self.unusedView.setDragDropMode(QListView.DragDrop)
        self.unusedView.setDefaultDropAction(Qt.MoveAction)
        self.unusedView.viewport().setAcceptDrops(True)

        self.unusedModel = VariableListModel(
            flags=Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                  Qt.ItemIsDragEnabled
        )
        self.unusedView.setModel(self.unusedModel)

        box.layout().addWidget(self.unusedView)

        box = OWGUI.widgetBox(self.controlArea, "Output")
        cb = OWGUI.checkBox(box, self, "autoCommit", "Auto commit")
        b = OWGUI.button(box, self, "Commit", callback=self.commit)
        OWGUI.setStopper(self, b, cb, "_outputChanged", callback=self.commit)

    def setData(self, data):
        """
        Set the input data.
        """
        self._storeRoles()

        self.closeContext("")
        self.data = data

        if data is not None:
            self.openContext("", data)
            domain = data.domain
            features = (domain.variables + domain.class_vars +
                        domain.get_metas().values())
            sort_by = []
            unused = []

            for feat in features:
                hint = self.sortroles.get((feat.name, feat.var_type), None)
                if hint is not None:
                    index, order = hint
                    sort_by.append((feat, index, order))
                else:
                    unused.append(feat)

            sort_by = sorted(sort_by, key=itemgetter(1))
            self.sortByModel[:] = [feat for feat, _, _ in sort_by]
            self.unusedModel[:] = unused

            # Restore the sort orders
            for i, (_, _, order) in enumerate(sort_by):
                index = self.sortByModel.index(i, 0)
                self.sortByModel.setData(index, order, SortOrderRole)

        self.commit()

    def _invalidate(self):
        if self.autoCommit:
            self.commit()
        else:
            self._outputChanged = True

    def _sortingParams(self):
        params = []

        for i, feature in enumerate(self.sortByModel):
            index = self.sortByModel.index(i, 0)
            order = toSortOrder(index.data(SortOrderRole))
            params.append((feature, order))

        return params

    def commit(self):
        params = self._sortingParams()

        if self.data:
            instances = sorted(self.data, key=sort_key(params))

            data = Orange.data.Table(self.data.domain, instances)
        else:
            data = self.data

        self.send("Data", data)

    def _storeRoles(self):
        """
        Store the sorting roles back into the stored settings.
        """
        roles = {}
        for i, feature in enumerate(self.sortByModel):
            index = self.sortByModel.index(i, 0)
            order = toSortOrder(index.data(SortOrderRole))
            roles[(feature.name, feature.var_type)] = (i, int(order))

        self.sortroles = roles

    def getSettings(self, *args, **kwargs):
        self._storeRoles()
        return OWWidget.getSettings(self, *args, **kwargs)


def sort_key(params):
    def key(inst):
        return tuple(
            inst[feature] if order == Qt.AscendingOrder
                else rev_compare(inst[feature])
            for feature, order in params
        )
    return key


class rev_compare(object):
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return self.obj == other.obj

    def __lt__(self, other):
        return not self.obj < other.obj


if __name__ == "__main__":
    app = QApplication([])
    w = OWDataSort()
    data = Orange.data.Table("iris")
    w.setData(data)
    w.show()
    w.raise_()
    app.exec_()
    w.saveSettings()
