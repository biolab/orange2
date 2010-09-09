from PyQt4.QtCore import *
from PyQt4.QtGui import *

from functools import wraps

class PyListModel(QAbstractListModel):
    """ A model for displaying python list like objects in Qt item view classes
    """
    def __init__(self, iterable=[], parent=None, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled):
        QAbstractListModel.__init__(self, parent)
        self._list = []
        self._flags = flags 
        self.extend(iterable)
        
    def wrap(self, list):
        """ Wrap the list with this model. All changes to the model
        are done in place on the passed list
        """
        self._list = list
        self.reset()
    
    def index(self, row, column=0, parent=QModelIndex()):
        return QAbstractListModel.createIndex(self, row, column, parent)
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return QVariant(str(section))
    
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self)
    
    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 1
    
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return QVariant(self[index.row()])
        return QVariant()
    
    def parent(self, index=QModelIndex()):
        return QModelIndex()
    
    def setData(self, index, value, role=Qt.EditRole):
        obj = value.toPyObject()
        self[index.row()] = obj
        
    def flags(self, index):
        return Qt.ItemFlags(self._flags)
    
    def extend(self, iterable):
        list_ = list(iterable)
        self.beginInsertRows(QModelIndex(), len(self), len(self) + len(list_))
        self._list.extend(list_)
        self.endInsertRows()
    
    def append(self, item):
        self.extend([item])
        
    def insert(self, i, val):
        self.beginInsertRows(QModelIndex(), i, i + 1)
        self._list.insert(i, val)
        self.endInsertRows()
        
    def remove(self, val):
        i = self._list.index(val)
        self.__delitem__(i)
        
    def pop(self, i):
        item = self._list[i]
        self.__delitem__(i)
        return item
    
    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(self._list)
    
    def __getitem__(self, i):
        return self._list[i]
        
    def __add__(self, iterable):
        return PyListModel(self._list + iterable, self.parent())
    
    def __iadd__(self, iterable):
        self.extend(iterable)
        
    def __delitem__(self, i):
        self.beginRemoveRows(QModelIndex(), i, i)
        del self._list[i]
        self.endRemoveRows()
        
    def __delslice(self, i, j):
        self.beginRemoveRows(QModelIndex(), i, j)
        del self._list[i:j]
        self.endRemoveRows()
        
    def __setitem__(self, i, value):
        self._list[i] = value
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(i), self.index(i))
        
    def __setslice__(self, i, j, iterable):
        self._list[i:j] = iterable
        self.reset()
#        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(i), self.index(j))
        
    def reverse(self):
        self._list.reverse()
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(0), self.index(len(self) -1))
        
    def sort(self, *args, **kwargs):
        self._list.sort(*args, **kwargs)
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(0), self.index(len(self) -1))
        
    def __repr__(self):
        return "PyListModel(%s)" % repr(self._list)
    
    def __nonzero__(self):
        return len(self) != 0
    
    #for Python 3000
    def __bool__(self):
        return len(self) != 0

import OWGUI
import orange

class VariableListModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        i = index.row()
        if role == Qt.DisplayRole:
            return QVariant(self.__getitem__(i).name)
        elif role == Qt.DecorationRole:
            return QVariant(OWGUI.getAttributeIcons().get(self.__getitem__(i).varType, -1))
        elif role == Qt.EditRole:
            return QVariant(self.__getitem__(i))
        return QVariant()
        
    def setData(self, index, value, role):
        i = index.row()
        if role == Qt.EditRole:    
            self.__setitem__(i,  value.toPyObject()) 
        
class VariableEditor(QWidget):
    def __init__(self, var, parent):
        QWidget.__init__(self, parent)
        self.var = var
        layout = QHBoxLayout()
        self._attrs = OWGUI.getAttributeIcons()
        self.type_cb = QComboBox(self)
        for attr, icon in self._attrs.items():
            if attr != -1:
                self.type_cb.addItem(icon, str(attr))
        layout.addWidget(self.type_cb)
        
        self.name_le = QLineEdit(self)
        layout.addWidget(self.name_le)
        
        self.setLayout(layout)
        
        self.connect(self.type_cb, SIGNAL("currentIndexChanged(int)"), self.edited)
        self.connect(self.name_le, SIGNAL("editingFinished()"), self.edited)
    
    def edited(self, *args):
        self.emit(SIGNAL("edited()"))
         
    def setData(self, type, name):
        self.type_cb.setCurrentIndex(self._attr.keys().index(type))
        self.name_le.setText(name)
        
class EnumVariableEditor(VariableEditor):
    def __init__(self, var, parent):
        VariableEditor.__init__(self, var, parent)
        
class FloatVariableEditor(QLineEdit):
    
    def setVariable(self, var):
        self.setText(str(var.name))
        
    def getVariable(self):
        return orange.FloatVariable(str(self.text()))

    
class StringVariableEditor(QLineEdit):
    def setVariable(self, var):
        self.setText(str(var.name))
        
    def getVariable(self):
        return orange.StringVariable(str(self.text()))
        
class VariableDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        var = index.data(Qt.EditRole).toPyObject()
        if isinstance(var, orange.EnumVariable):
            return EnumVariableEditor(parent)
        elif isinstance(var, orange.FloatVariable):
            return FloatVariableEditor(parent)
        elif isinstance(var, orange.StringVariable):
            return StringVariableEditor(parent)
#        return VariableEditor(var, parent)
    
    def setEditorData(self, editor, index):
        var = index.data(Qt.EditRole).toPyObject()
        editor.variable = var
        
    def setModelData(self, editor, model, index):
        model.setData(index, QVariant(editor.variable), Qt.EditRole)
        
#    def displayText(self, value, locale):
#        return value.toPyObject().name
        
class ListSingleSelectionModel(QItemSelectionModel):
    """ Item selection model for list item models with single selection.
    
    Defines signal:
        - selectedIndexChanged(QModelIndex)
        
    """
    def __init__(self, model, parent=None):
        QItemSelectionModel.__init__(self, model, parent)
        self.connect(self, SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.onSelectionChanged)
        
    def onSelectionChanged(self, new, old):
        index = list(new.indexes())
        if index:
            index = index.pop()
        else:
            index = QModelIndex()
        self.emit(SIGNAL("selectedIndexChanged(QModelIndex)"), index)
        
    def selectedRow(self):
        """ Return QModelIndex of the selected row or invalid if no selection. 
        """
        rows = self.selectedRows()
        if rows:
            return rows[0]
        else:
            return QModelIndex()
        
    def select(self, index, flags=QItemSelectionModel.ClearAndSelect):
        if isinstance(index, int):
            index = self.model().index(index)
        return QItemSelectionModel.select(self, index, flags)
    
        
class ModelActionsWidget(QWidget):
    def __init__(self, actions=[], parent=None, direction=QBoxLayout.LeftToRight):
        QWidget.__init__(self, parent)
        self.actions = []
        self.buttons = []
        layout = QBoxLayout(direction)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        for action in actions:
            self.addAction(action)
        self.setLayout(layout)
            
    def actionButton(self, action):
        if isinstance(action, QAction):
            button = QToolButton()
            button.setDefaultAction(action)
            return button
        elif isinstance(action, QAbstractButton):
            return action
            
    def insertAction(self, ind, action, *args):
        button = self.actionButton(action)
        self.layout().insertWidget(ind, button, *args)
        self.buttons.insert(ind, button)
        self.actions.insert(ind, action)
        return button
        
    def addAction(self, action, *args):
        return self.insertAction(-1, action, *args)
    