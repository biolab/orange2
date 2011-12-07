"""
<name>Edit Domain</name>
<description>Edit domain variables</description>
<icon>icons/EditDomain.png</icon>
<contact>Ales Erjavec (ales.erjavec(@ at @)fri.uni-lj.si)</contact>
<priority>3125</priority>
"""

from OWWidget import *
from OWItemModels import VariableListModel, PyListModel

import OWGUI

import Orange

def is_discrete(var):
    return isinstance(var, Orange.data.variable.Discrete)

def is_continuous(var):
    return isinstance(var, Orange.data.variable.Continuous)

def get_qualified(module, name):
    module = __import__(module)
    return getattr(module, name)

def variable_description(var):
    var_type = type(var)
    if is_discrete(var):
        return (var_type.__module__,
                var_type.__name__,
                var.name, 
                (("values", tuple(var.values)),), 
                tuple(sorted(var.attributes.items())))
    else:
        return (var_type.__module__,
                var_type.__name__,
                var.name, 
                (), 
                tuple(sorted(var.attributes.items())))

def variable_from_description(description):
    module, type_name, name, kwargs, attrs = description
    type = get_qualified(module, type_name)
    var = type(name, **dict(list(kwargs)))
    var.attributes.update(attrs)
    return var
    
from PyQt4 import QtCore, QtGui

QtCore.Slot = QtCore.pyqtSlot
QtCore.Signal = QtCore.pyqtSignal

class PyStandardItem(QStandardItem):
    def __lt__(self, other):
        return id(self) < id(other)
    
class DictItemsModel(QStandardItemModel):
    def __init__(self, parent=None, dict={}):
        QStandardItemModel.__init__(self, parent)
        self.setHorizontalHeaderLabels(["Key", "Value"])
        self.set_dict(dict)
        
    def set_dict(self, dict):
        self._dict = dict
        self.clear()
        self.setHorizontalHeaderLabels(["Key", "Value"])
        for key, value in sorted(dict.items()):
            key_item = PyStandardItem(QString(key))
            value_item = PyStandardItem(QString(value))
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item.setFlags(value_item.flags() | Qt.ItemIsEditable)
            self.appendRow([key_item, value_item])
            
    def get_dict(self):
        dict = {}
        for row in range(self.rowCount()):
            key_item = self.item(row, 0)
            value_item = self.item(row, 1)
            dict[str(key_item.text())] = str(value_item.text())
        return dict

class VariableEditor(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setup_gui()
        
    def setup_gui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.main_form = QFormLayout()
        self.main_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.main_form)
        
        self._setup_gui_name()
        self._setup_gui_labels()
        
    def _setup_gui_name(self):
        self.name_edit = QLineEdit()
        self.main_form.addRow("Name", self.name_edit)
        self.name_edit.editingFinished.connect(self.on_name_changed)
        
    def _setup_gui_labels(self):
        self.labels_edit = QTreeView()
        self.labels_model = DictItemsModel()
        self.labels_edit.setModel(self.labels_model)
        
        self.labels_model.dataChanged.connect(self.on_labels_changed)
        self.main_form.addRow("Labels", self.labels_edit)
        
        # TODO: add/remove label buttons 
        
    def set_data(self, var):
        self.clear()
        self.var = var
        if var is not None:
            self.name_edit.setText(var.name)
            self.labels_model.set_dict(dict(var.attributes))
            
    def get_data(self):
        name = str(self.name_edit.text())
        labels = self.labels_model.get_dict()
        
        # Is the variable actually changed. 
        if not self.is_same():
            var = type(self.var)(name)
            var.attributes.update(labels)
            self.var = var
        else:
            var = self.var
        
        return var
    
    def is_same(self):
        """ Is the current model state the same as the input. 
        """
        name = str(self.name_edit.text())
        labels = self.labels_model.get_dict()
        
        return self.var and name == self.var.name and labels == self.var.attributes
            
    def clear(self):
        self.var = None
        self.name_edit.setText("")
        self.labels_model.set_dict({})
        
    def maybe_commit(self):
        if not self.is_same():
            self.commit()
            
    def commit(self):
        self.emit(SIGNAL("variable_changed()"))
        
    @QtCore.Slot()
    def on_name_changed(self):
        self.maybe_commit()
        
    @QtCore.Slot()
    def on_labels_changed(self):
        self.maybe_commit()
        
        
class DiscreteVariableEditor(VariableEditor):
    def setup_gui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.main_form = QFormLayout()
        self.main_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.main_form)
        
        self._setup_gui_name()
        self._setup_gui_values()
        self._setup_gui_labels()
        
    def _setup_gui_values(self):
        self.values_edit = QListView()
        self.values_model = PyListModel(flags=Qt.ItemIsSelectable | \
                                        Qt.ItemIsEnabled | Qt.ItemIsEditable)
        self.values_edit.setModel(self.values_model)
        
        self.values_model.dataChanged.connect(self.on_values_changed)
        self.main_form.addRow("Values", self.values_edit)

    def set_data(self, var):
        VariableEditor.set_data(self, var)
        self.values_model.wrap([])
        if var is not None:
            for v in var.values:
                self.values_model.append(v)
                
    def get_data(self):
        name = str(self.name_edit.text())
        labels = self.labels_model.get_dict()
        values = map(str, self.values_model)
        
        if not self.is_same():
            var = type(self.var)(name, values=values)
            var.attributes.update(labels)
            self.var = var
        else:
            var = self.var
            
        return var
            
    def is_same(self):
        """ Is the current model state the same as the input. 
        """
        values = map(str, self.values_model)
        return VariableEditor.is_same(self) and self.var.values == values
    
    def clear(self):
        VariableEditor.clear(self)
        self.values_model.wrap([])
        
    @QtCore.Slot()
    def on_values_changed(self):
        self.maybe_commit()
        
        
class ContinuousVariableEditor(VariableEditor):
    pass


class OWEditDomain(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["domain_change_hints", "selected_index"])}
    settingsList = []
    
    def __init__(self, parent=None, signalManager=None, title="Edit Domain"):
        OWWidget.__init__(self, parent, signalManager, title)
        
        self.inputs = [("Input Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Output Data", Orange.data.Table)]
        
        # Settings
        
        self.domain_change_hints = {}
        self.selected_index = 0
        self.auto_commit = False
        self.changed_flag = False
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        box = OWGUI.widgetBox(self.controlArea, "Domain Features")
        self.domain_view = QListView()
        self.domain_view.setSelectionMode(QListView.SingleSelection)
        
        self.domain_model = VariableListModel()
        
        self.domain_view.setModel(self.domain_model)
        
        self.connect(self.domain_view.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     self.on_selection_changed)
        
        box.layout().addWidget(self.domain_view)
        
        box = OWGUI.widgetBox(self.mainArea, "Edit Feature")
        self.editor_stack = QStackedWidget()
        box.layout().addWidget(self.editor_stack)
        
        box = OWGUI.widgetBox(self.controlArea, "Reset")
        
        OWGUI.button(box, self, "Reset selected",
                     callback=self.reset_selected,
                     tooltip="Reset changes made to the selected feature"
                     )
        
        OWGUI.button(box, self, "Reset all",
                     callback=self.reset_all,
                     tooltip="Reset all changes made to the domain"
                     )
        
        
        box = OWGUI.widgetBox(self.controlArea, "Commit")
        
        b = OWGUI.button(box, self, "&Commit",
                         callback=self.commit,
                         tooltip="Commit the data with the changed domain",
                         )
        
        cb = OWGUI.checkBox(box, self, "auto_commit",
                            label="Commit automatically",
                            tooltip="Commit the changed domain on any change",
                            callback=self.commit_if)
        
        OWGUI.setStopper(self, b, cb, "changed_flag",
                         callback=self.commit)
        
        self._editor_cache = {}
        
        self.resize(600, 500)
        
    def clear(self):
        self.data = None
        self.domain_model[:] = []
        self.domain_change_hints = {}
        self.clear_editor()
        
    def clear_editor(self):
        current = self.editor_stack.currentWidget()
        if current:
            QObject.disconnect(current, SIGNAL("variable_changed()"),
                               self.on_variable_changed)
            current.set_data(None)
        
    def set_data(self, data=None):
        self.closeContext("")
        self.clear()
        self.data = data
        if data is not None:
            input_domain = data.domain
            all_vars = list(input_domain.variables) + \
                       input_domain.getmetas().values()
            
            self.openContext("", data)
            
            edited_vars = []
            for var in all_vars:
                desc = variable_description(var)
                changed = self.domain_change_hints.get(desc, None)
                if changed is not None:
                    new = variable_from_description(changed)
                    new.source_variable = var
                    new.get_value_from = Orange.core.ClassifierFromVar(whichVar=var)
                    var = new
                edited_vars.append(var)
            
            self.all_vars = all_vars
            self.input_domain = input_domain
            
            self.domain_model[:] = edited_vars
            
            # Try to restore the variable selection
            index = self.selected_index
            if self.selected_index >= len(all_vars):
                index = 0 if len(all_vars) else -1
            if index >= 0:
                self.select_variable(index)
        
            self.changed_flag = True
            self.commit_if()
        else:
            # To force send None on output
            self.commit()
            
    def on_selection_changed(self, *args):
        i = self.selected_var_index()
        if i is not None:
            self.open_editor(i)
            self.selected_index = i
        
    def selected_var_index(self):
        rows = self.domain_view.selectionModel().selectedRows()
        if rows:
            return rows[0].row()
        else:
            return None
        
    def select_variable(self, index):
        sel_model = self.domain_view.selectionModel()
        sel_model.select(self.domain_model.index(index, 0),
                         QItemSelectionModel.ClearAndSelect)
        
    def open_editor(self, index):
        # First remove and clear the current editor if any
        self.clear_editor()
            
        var = self.domain_model[index]
        
        editor = self.editor_for_variable(var)
        editor.set_data(var)
        self.edited_variable_index = index
        
        QObject.connect(editor, SIGNAL("variable_changed()"),
                        self.on_variable_changed)
        self.editor_stack.setCurrentWidget(editor)
    
    def editor_for_variable(self, var):
        editor = None
        if is_discrete(var):
            editor = DiscreteVariableEditor
        elif is_continuous(var):
            editor = ContinuousVariableEditor
        else:
            editor = VariableEditor
            
        if type(var) not in self._editor_cache:
            editor = editor()
            self._editor_cache[type(var)] = editor
            self.editor_stack.addWidget(editor)
            
        return self._editor_cache[type(var)]
    
    def on_variable_changed(self):
        var = self.domain_model[self.edited_variable_index]
        editor = self.editor_stack.currentWidget()
        new_var = editor.get_data()
        
        self.domain_model[self.edited_variable_index] = new_var
        old_var = self.all_vars[self.edited_variable_index]
        
        self.domain_change_hints[variable_description(old_var)] = \
                    variable_description(new_var)
                    
        new_var.source_variable = old_var
        new_var.get_value_from = Orange.core.ClassifierFromVar(whichVar=old_var)
        self.commit_if()
         
    def reset_all(self):
        self.domain_change_hints = {}
        if self.data is not None:
            # To invalidate stored hints
            self.closeContext("")
            self.openContext("", self.data)
            self.domain_model[:] = self.all_vars
            self.select_variable(self.selected_index)
            
    def reset_selected(self):
        if self.data is not None:
            var = self.all_vars[self.selected_index]
            self.domain_model[self.selected_index] = var
            self.editor_stack.currentWidget().set_data(var)
            
    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.changed_flag = True
        
    def commit(self):
        new_data = None
        if self.data is not None:
            new_vars = list(self.domain_model)
            variables = new_vars[: len(self.input_domain.variables)]
            class_var = None
            if self.input_domain.class_var:
                class_var = variables[-1]
                variables = variables[:-1]
            
            new_metas = new_vars[len(self.input_domain.variables) :]
            new_domain = Orange.data.Domain(variables, class_var)
            
            # Assumes getmetas().items() order has not changed.
            # TODO: store them in set_data
            for (mid, _), new in zip(self.input_domain.getmetas().items(), 
                                       new_metas):
                new_domain.addmeta(mid, new)
                
            new_data = Orange.data.Table(new_domain, self.data)
        
        self.send("Output Data", new_data)
        self.changed_flag = False
            
        
def main():
    import sys
    app = QApplication(sys.argv)
    w = OWEditDomain()
    data = Orange.data.Table("iris")
#    data = Orange.data.Table("rep:GDS636.tab")
    w.set_data(data)
    w.show()
    rval = app.exec_()
    w.set_data(None)
    w.saveSettings()
    return rval
        
if __name__ == "__main__":
    import sys
    sys.exit(main())
    