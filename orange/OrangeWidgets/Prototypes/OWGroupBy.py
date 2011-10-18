"""<name>Group By</name>
<description>Group instances by selected columns</description>
<icons>icons/GroupBy.png</icons>

"""

from OWWidget import *
from OWItemModels import VariableListModel

import Orange
from Orange.data import Table, Domain, variable, utils
from Orange.statistics import distribution

import random
import math
from operator import itemgetter

def modus(values):
    dist = distribution.Distribution(values[0].variable)
    for v in values:
        dist.add(v)
    return dist.modus()

def mean(values):
    dist = distribution.Distribution(values[0].variable)
    for v in values:
        dist.add(v)
    return dist.average()

def geometric_mean(values):
    values = [float(v) for v in values if not v.is_special()]
    if values:
        prod = reduce(float.__mul__, values, 1.0)
        return math.pow(prod, 1.0/len(values))
    else:
        return "?"
    
def harmonic_mean(values):
    values = [float(v) for v in values if not v.is_special()]
    if values:
        hsum = sum(map(lambda v: 1.0 / (v or 1e-6), values))
        return len(values) / (hsum or 1e-6) 
    else:
        return "?"

def aggregate_func(func):
    if isinstance(func, basestring):
        mapping = {"random": random.choice,
                   "first": itemgetter(0),
                   "last": itemgetter(-1),
                   "modus": modus,
                   "mean": mean,
                   "geometric mean": geometric_mean,
                   "harmonic mean": harmonic_mean,
                   "join": lambda values: ", ".join(map(str, values))
                   }
        return mapping[func]
    return func
        
def group_by(table, group_attrs, aggregate_disc="first", aggregate_cont="mean",
             aggregate_string="join", attr_aggregate=None):
    if attr_aggregate is None:
        attr_aggregate = {}
    else:
        attr_aggregate = dict(attr_aggregate) # It is modified later
        
    all_vars = table.domain.variables + table.domain.getmetas().values()
    aggregate_vars = []
    for v in all_vars:
        if v not in group_attrs:
            if v in attr_aggregate:
                pass
            elif isinstance(v, variable.Continuous):
                attr_aggregate[v] = aggregate_cont
            elif isinstance(v, variable.Discrete):
                attr_aggregate[v] = aggregate_disc
            elif isinstance(v, variable.String):
                attr_aggregate[v] = aggregate_string
            else:
                raise TypeError(v)
            aggregate_vars.append(v)
            attr_aggregate[v] = aggregate_func(attr_aggregate[v])
            
    indices_map = utils.table_map(table, group_attrs, exclude_special=False)
    new_instances = []
    key_set = set()
    print group_attrs, indices_map
    for inst in table: # Iterate over the table instead of the inidces_map to preserve order
        key = tuple([str(inst[v]) for v in group_attrs])
        if key in key_set:
            continue # Already seen this group
        indices = indices_map[key]
        new_instance = Orange.data.Instance(inst) # Copy
        for v in aggregate_vars:
            values = [table[i][v] for i in indices] # Values to aggregate
            print attr_aggregate[v], values, " -> ", attr_aggregate[v](values)
            new_instance[v] = attr_aggregate[v](values)
        new_instances.append(new_instance)
        key_set.add(key)
    return Orange.data.Table(new_instances)

AggregateMethodRole = OWGUI.OrangeUserRole.next()

DISC_METHODS = \
    [("Modus", ),
     ("Random", ),
     ("First", ),
     ("Last", ),
     ]

CONT_METHODS = \
    [("Mean", ),
     ("Modus", ),
     ("Geometric Mean", ),
     ("Harmonic Mean", ),
     ("Random", ),
     ("First", ),
     ("Last", ),
     ]
    
STR_METHODS = \
    [("Join", ),
     ("Random", ),
     ("First", ),
     ("Last", ),
     ]
    
PYTHON_METHODS = \
    [("Random", ),
     ("First", ),
     ("Last", ),
     ]
    
DEFAULT_METHOD = "First"
    
    
class OWGroupBy(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["hints"])}
    settingsList = []
    def __init__(self, parent=None, signalManager=None, title="Group By"):
        OWWidget.__init__(self, parent, signalManager, title, 
                          wantMainArea=False)
        
        self.inputs = [("Input Data", Table, self.set_data)]
        self.outputs = [("Output Data", Table)]
        
        self.auto_commit = False
        self.hints = {}
        
        self.state_chaged_flag = False
        
        self.loadSettings()
        
        #############
        # Data Models
        #############
        
        self.group_list = VariableListModel(parent=self, 
                            flags=Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
                            Qt.ItemIsDragEnabled )
        self.aggregate_list = VariableAggragateModel(parent=self,
                            flags=Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
                            Qt.ItemIsDragEnabled | \
                            Qt.ItemIsEditable)
        
        self.aggregate_delegate = AggregateDelegate()
        
        #####
        # GUI
        #####
        
        box = OWGUI.widgetBox(self.controlArea, "Group By Attributes")
        self.group_view = QListView()
        self.group_view.setSelectionMode(QListView.ExtendedSelection)
        self.group_view.setDragDropMode(QListView.DragDrop)
        self.group_view.setModel(self.group_list)
#        self.group_view.setDragDropOverwriteMode(True)
        self.group_view.setDefaultDropAction(Qt.MoveAction)
        self.group_view.viewport().setAcceptDrops(True)
#        self.group_view.setDropIndicatorShown(True)
        self.group_view.setToolTip("A set of attributes to group by (drag \
values to 'Aggregate Attributes' to remove them from this group).")
        box.layout().addWidget(self.group_view)
        
        box = OWGUI.widgetBox(self.controlArea, "Aggregate Attributes")
        self.aggregate_view = AggregateListView()
        self.aggregate_view.setSelectionMode(QListView.ExtendedSelection)
        self.aggregate_view.setDragDropMode(QListView.DragDrop)
        self.aggregate_view.setItemDelegate(self.aggregate_delegate)
        self.aggregate_view.setModel(self.aggregate_list)
        self.aggregate_view.setEditTriggers(QListView.SelectedClicked)
#        self.aggregate_view.setDragDropOverwriteMode(False)
        self.aggregate_view.setDefaultDropAction(Qt.MoveAction)
        self.aggregate_view.viewport().setAcceptDrops(True)
        self.aggregate_view.setToolTip("Aggregated attributes.")
        
        box.layout().addWidget(self.aggregate_view)
        
        OWGUI.rubber(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Commit")
#        cb = OWGUI.checkBox(box, self, "auto_commit", "Commit on any change.",
#                            tooltip="Send the data on output on any change of settings or inputs.",
#                            callback=self.commit_if
#                            )
        b = OWGUI.button(box, self, "Commit", callback=self.commit, 
                         tooltip="Send data on output.", 
                         autoDefault=True)
#        OWGUI.setStopper(self, b, cb, "state_chaged_flag",
#                         callback=self.commit)
        
    def set_data(self, data=None):
        """ Set the input data for the widget.
        """
        self.update_hints()
        self.closeContext("")
        self.clear()
        if data is not None:
            self.init_with_data(data)
            self.openContext("", data)
            self.init_state_from_hints()
        
        self.commit_if()
        
    def init_with_data(self, data):
        """ Init widget state from data
        """
        attrs = data.domain.variables + data.domain.get_metas().values()
#        self.group_list.set_items(attrs)
        self.all_attrs = attrs
        self.hints = dict([((a.name, a.varType), ("group_by", "First")) for a in attrs])
        self.data = data
        
    def init_state_from_hints(self):
        """ Init the group and aggregate from hints (call after openContext) 
        """
        group = []
        aggregate = []
        aggregate_hint = {}
        for a in self.all_attrs:
            try:
                place, hint = self.hints.get((a.name, a.var_type), ("group_by", DEFAULT_METHOD))
            except Exception:
                place, hint = ("group_by", DEFAULT_METHOD)
            if place == "group_by":
                group.append(a)
            else:
                aggregate.append(a)
            aggregate_hint[a] = hint
        self.group_list[:] = group
        self.aggregate_list[:] = aggregate
        
        for i, a in enumerate(group):
            self.group_list.setData(self.group_list.index(i),
                                    aggregate_hint[a],
                                    AggregateMethodRole)
            
        for i, a in enumerate(aggregate):
            self.aggregate_list.setData(self.aggregate_list.index(i),
                                        aggregate_hint[a],
                                        AggregateMethodRole)
        
    def update_hints(self):
        for i, var in enumerate(self.group_list):
            self.hints[var.name, var.var_type] = \
                ("group_by", str(self.group_list.data( \
                                         self.group_list.index(i),
                                         AggregateMethodRole).toPyObject()))
                
        for i, var in enumerate(self.aggregate_list):
            self.hints[var.name, var.var_type] = \
                ("aggregate", str(self.aggregate_list.data( \
                                        self.aggregate_list.index(i),
                                        AggregateMethodRole).toPyObject()))
        
        
    def clear(self):
        """ Clear the widget state.
        """
        self.data = None
        self.group_list[:] = [] #clear()
        self.aggregate_list[:] = [] #.clear()
        self.all_attrs = []
        self.hints = {}
        
    def get_aggregates_from_hints(self):
        aggregates = {}
        for i, v in enumerate(self.aggregate_list):
            _, hint = self.hints.get((v.name, v.var_type), ("", DEFAULT_METHOD))
            
            aggregates[v] = hint.lower()
        return aggregates
    
    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.state_chaged_flag = True
            
    def commit(self):
        self.update_hints()
        if self.data is not None:
            group = list(self.group_list)
            aggregates = self.get_aggregates_from_hints()
            print aggregates
            data = group_by(self.data, group, attr_aggregate=aggregates)
        else:
            data = None
        self.send("Output Data", data)
        self.state_chaged_flag = False
        

def aggregate_options(var):
    if isinstance(var, variable.Discrete):
        items = [m[0] for m in DISC_METHODS]
    elif isinstance(var, variable.Continuous):
        items = [m[0] for m in CONT_METHODS]
    elif isinstance(var, variable.String):
        items = [m[0] for m in STR_METHODS]
    elif isinstance(var, variable.Python):
        items = [m[0] for m in PYTHON_METHODS]
    else:
        items = []
    return items


class AggregateDelegate(QStyledItemDelegate):
    def __init__(self, *args, **kwargs):
        QStyledItemDelegate.__init__(self, *args, **kwargs)
        
    def paint(self, painter, option, index):
        val = index.data(Qt.EditRole).toPyObject()
        method = index.data(AggregateMethodRole)
        if method.isValid():
            met = method.toPyObject()
        else:
            met = ""
        option.text = QString(val.name + str(met))
        QStyledItemDelegate.paint(self, painter, option, index)
        
    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.setFocusPolicy(Qt.StrongFocus)
        return editor
        
    def setEditorData(self, editor, index):
        var = index.data(Qt.EditRole).toPyObject() 
        options = aggregate_options(var)
        current = index.data(AggregateMethodRole).toPyObject()
        current_index = options.index(current) if current in options else 0
        editor.clear()
        editor.addItems(options)
        editor.setCurrentIndex(current_index)
        QObject.connect(editor, SIGNAL("activated(int)"), 
                    lambda i:self.emit(SIGNAL("commitData(QWidget)"), editor))
        
    def setModelData(self, editor, model, index):
        method = str(editor.currentText())
        model.setData(index, QVariant(method), AggregateMethodRole)
        
        
class AggregateListView(QListView):
    def contextMenuEvent(self, event):
        selection_model = self.selectionModel()
        model = self.model()
        rows = selection_model.selectedRows()
        rows = [r.row() for r in rows]
        if rows:
            options = [aggregate_options(model[i]) for i in rows]
            options = reduce(set.intersection, map(set, options[1:]), set(options[0]))
            menu = QMenu(self)
            for option in options:
                menu.addAction(option)
            selected = menu.exec_(event.globalPos())
            if selected:
                name = selected.text()
                for i in rows:
                    model.setData(model.index(i), QVariant(name), AggregateMethodRole)
                    
        
class VariableAggragateModel(VariableListModel):
    def data(self, index, role=Qt.DisplayRole):
        i = index.row()
        if role == Qt.DisplayRole:
            var_name = self.__getitem__(i).name
            met = self.data(index, AggregateMethodRole)
            if met.isValid():
                met = " (%s)" % str(met.toString())
            else:
                met = ""
            return QVariant(var_name + met)
        else:
            return VariableListModel.data(self, index, role=role)
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWGroupBy()
    w.show()
    data = Orange.data.Table("lenses.tab")
    w.set_data(data)
    w.set_data(None)
    w.set_data(data)
    app.exec_()
    w.set_data(None)
    w.saveSettings()
    