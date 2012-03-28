"""
<name>Multi-key Merge Data</name>
<description>Merge datasets based on values of selected tuples of attributes.</description>
<icon>icons/MergeData.png</icon>
<priority>100</priority>
<contact>Peter Husen (phusen@bmb.sdu.dk)</contact>
"""

from OWWidget import *
from OWItemModels import PyListModel, VariableListModel

#import OWGUI
import Orange


def slices(indices):
    """ Group the given integer indices into slices
    """
    indices = list(sorted(indices))
    if indices:
        first = last = indices[0]
        for i in indices[1:]:
            if i == last + 1:
                last = i
            else:
                yield first, last + 1
                first = last = i
        yield first, last + 1


def delslice(model, start, end):
    """ Delete the start, end slice (rows) from the model. 
    """
    if isinstance(model, PyListModel):
        model.__delslice__(start, end)
    elif isinstance(model, QAbstractItemModel):
        model.removeRows(start, end-start)
    else:
        raise TypeError(type(model))


class VariablesListItemModel(VariableListModel):
    """ An Qt item model for for list of orange.Variable objects.
    Supports drag operations
    """
        
    def flags(self, index):
        flags = VariableListModel.flags(self, index)
        if index.isValid():
            flags |= Qt.ItemIsDragEnabled
        else:
            flags |= Qt.ItemIsDropEnabled
        return flags
    
    ###########
    # Drag/Drop
    ###########
    
    MIME_TYPE = "application/x-Orange-VariableListModelData"
    
    def supportedDropActions(self):
        return Qt.MoveAction
    
    def supportedDragActions(self):
        return Qt.MoveAction
    
    def mimeTypes(self):
        return [self.MIME_TYPE]
    
    def mimeData(self, indexlist):
        descriptors = []
        vars = []
        item_data = []
        for index in indexlist:
            var = self[index.row()]
            descriptors.append((var.name, var.varType))
            vars.append(var)
            item_data.append(self.itemData(index))
        
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, QByteArray(str(descriptors)))
        mime._vars = vars
        mime._item_data = item_data
        return mime
    
    def dropMimeData(self, mime, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
    
        vars, item_data = self.items_from_mime_data(mime)
        if vars is None:
            return False
        
        if row == -1:
            row = len(self)
            
        self.__setslice__(row, row, vars)
        
        for i, data in enumerate(item_data):
            self.setItemData(self.index(row + i), data)
            
        return True
    
    def items_from_mime_data(self, mime):
        if not mime.hasFormat(self.MIME_TYPE):
            return None, None
        
        if hasattr(mime, "_vars"):
            vars = mime._vars
            item_data = mime._item_data
            return vars, item_data
        else:
            #TODO: get vars from orange.Variable.getExisting
            return None, None


class VariablesListItemView(QListView):
    """ A Simple QListView subclass initialized for displaying
    variables.
    """
    def __init__(self, parent=None):
        QListView.__init__(self, parent)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListView.DragDrop)
        if hasattr(self, "setDefaultDropAction"):
            # For compatibility with Qt version < 4.6
            self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.viewport().setAcceptDrops(True)
    
    def startDrag(self, supported_actions):
        indices = self.selectionModel().selectedIndexes()
        indices = [i for i in indices if i.flags() & Qt.ItemIsDragEnabled]
        if indices:
            data = self.model().mimeData(indices)
            if not data:
                return
            # rect = QRect()
            
            drag = QDrag(self)
            drag.setMimeData(data)
            
            default_action = Qt.IgnoreAction
            if hasattr(self, "defaultDropAction") and \
                    self.defaultDropAction() != Qt.IgnoreAction and \
                    supported_actions & self.defaultDropAction():
                default_action = self.defaultDropAction()
            elif supported_actions & Qt.CopyAction and dragDropMode() != QListView.InternalMove:
                default_action = Qt.CopyAction
            
            res = drag.exec_(supported_actions, default_action)
                
            if res == Qt.MoveAction:
                selected = self.selectionModel().selectedIndexes()
                rows = map(QModelIndex.row, selected)
                
                for s1, s2 in reversed(list(slices(rows))):
                    delslice(self.model(), s1, s2)
    
    def render_to_pixmap(self, indices):
        pass


from functools import partial
class OWMultiMergeData(OWWidget):
    contextHandlers = { "A": DomainContextHandler("A", [ContextField("domainA_role_hints")]),
                        "B": DomainContextHandler("B", [ContextField("domainB_role_hints")]) }

    def __init__(self, parent = None, signalManager = None, name = "Merge data"):
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0)  #initialize base class

        # set channels
        self.inputs = [("Data A", ExampleTable, self.set_dataA),
                       ("Data B", ExampleTable, self.set_dataB)]
        
        self.outputs = [("Merged Data A+B", ExampleTable),
                        ("Merged Data B+A", ExampleTable)]

        self.domainA_role_hints = {}
        self.domainB_role_hints = {}
        
        # data
        self.dataA = None
        self.dataB = None

        # load settings
        self.loadSettings()

        # ####
        # GUI
        # ####
        
        import sip
        sip.delete(self.controlArea.layout())
        
        layout = QGridLayout()
        layout.setMargin(0)

        # Available A attributes
        box = OWGUI.widgetBox(self.controlArea, "Available A attributes", addToLayout=False)
        self.available_attrsA = VariablesListItemModel()
        self.available_attrsA_view = VariablesListItemView()
        self.available_attrsA_view.setModel(self.available_attrsA)
        
        self.connect(self.available_attrsA_view.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     partial(self.update_interface_state,
                             self.available_attrsA_view))
        
        box.layout().addWidget(self.available_attrsA_view)
        layout.addWidget(box, 0, 0, 2, 1)


        # Used A Attributes
        box = OWGUI.widgetBox(self.controlArea, "Used A attributes", addToLayout=False)
        self.used_attrsA = VariablesListItemModel()
        self.used_attrsA_view = VariablesListItemView()
        self.used_attrsA_view.setModel(self.used_attrsA)
        self.connect(self.used_attrsA_view.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     partial(self.update_interface_state,
                             self.used_attrsA_view))
        
        box.layout().addWidget(self.used_attrsA_view)
        layout.addWidget(box, 0, 2, 1, 1)


        # Data A info
        box = OWGUI.widgetBox(self, 'Data A', orientation = "vertical", addToLayout=0)
        self.lblDataAExamples = OWGUI.widgetLabel(box, "num examples")
        self.lblDataAAttributes = OWGUI.widgetLabel(box, "num attributes")
        layout.addWidget(box, 1, 1, 1, 2)


        # Available B attributes
        box = OWGUI.widgetBox(self.controlArea, "Available B attributes", addToLayout=False)
        self.available_attrsB = VariablesListItemModel()
        self.available_attrsB_view = VariablesListItemView()
        self.available_attrsB_view.setModel(self.available_attrsB)
        
        self.connect(self.available_attrsB_view.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     partial(self.update_interface_state,
                             self.available_attrsB_view))
        
        box.layout().addWidget(self.available_attrsB_view)
        layout.addWidget(box, 2, 0, 2, 1)


        # Used B Attributes
        box = OWGUI.widgetBox(self.controlArea, "Used B attributes", addToLayout=False)
        self.used_attrsB = VariablesListItemModel()
        self.used_attrsB_view = VariablesListItemView()
        self.used_attrsB_view.setModel(self.used_attrsB)
        self.connect(self.used_attrsB_view.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     partial(self.update_interface_state,
                             self.used_attrsB_view))


        # Data B info
        box.layout().addWidget(self.used_attrsB_view)
        layout.addWidget(box, 2, 2, 1, 1)

        box = OWGUI.widgetBox(self, 'Data B', orientation = "vertical", addToLayout=0)
        self.lblDataBExamples = OWGUI.widgetLabel(box, "num examples")
        self.lblDataBAttributes = OWGUI.widgetLabel(box, "num attributes")
        layout.addWidget(box, 3, 1, 1, 2)


        # A buttons
        bbox = OWGUI.widgetBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 0, 1, 1, 1)
        
        self.up_attrA_button = OWGUI.button(bbox, self, "Up", 
                    callback = partial(self.move_up, self.used_attrsA_view))
        self.move_attrA_button = OWGUI.button(bbox, self, ">",
                    callback = partial(self.move_selected, self.used_attrsA_view))
        self.down_attrA_button = OWGUI.button(bbox, self, "Down",
                    callback = partial(self.move_down, self.used_attrsA_view))


        # B buttons
        bbox = OWGUI.widgetBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 2, 1, 1, 1)

        self.up_attrB_button = OWGUI.button(bbox, self, "Up",
                    callback = partial(self.move_up, self.used_attrsB_view))
        self.move_attrB_button = OWGUI.button(bbox, self, ">",
                    callback = partial(self.move_selected, self.used_attrsB_view))
        self.down_attrB_button = OWGUI.button(bbox, self, "Down",
                    callback = partial(self.move_down, self.used_attrsB_view))


        # Apply / reset
        bbox = OWGUI.widgetBox(self.controlArea, orientation="horizontal", addToLayout=False, margin=0)
        applyButton = OWGUI.button(bbox, self, "Apply", callback=self.commit)
        resetButton = OWGUI.button(bbox, self, "Reset", callback=self.reset)
        
        layout.addWidget(bbox, 4, 0, 1, 3)
        
        layout.setHorizontalSpacing(0)
        self.controlArea.setLayout(layout)
        
        self.data = None
        self.output_report = None

        self.resize(500, 600)
        
        # For automatic widget testing using
        self._guiElements.extend( \
                  [(QListView, self.available_attrsA_view),
                   (QListView, self.used_attrsA_view),
                   (QListView, self.available_attrsB_view),
                   (QListView, self.used_attrsB_view),
                  ])


    ############################################################################################################################################################
    ## Data input and output management
    ############################################################################################################################################################
    
    
    def set_dataA(self, data=None):
        self.update_domainA_role_hints()
        self.closeContext("A")
        self.dataA = data
        if data is not None:
            self.openContext("A", data)
            all_vars = data.domain.variables + data.domain.getmetas().values()
            
            var_sig = lambda attr: (attr.name, attr.varType)
            
            domain_hints = dict([(var_sig(attr), ("available", i)) \
                            for i, attr in enumerate(data.domain.attributes)])
            
            domain_hints.update(dict([(var_sig(attr), ("available", i)) \
                for i, attr in enumerate(data.domain.getmetas().values())]))
            
            if data.domain.class_var:
                domain_hints[var_sig(data.domain.class_var)] = ("available", 0)
                    
            domain_hints.update(self.domainA_role_hints) # update the hints from context settings
            
            attrs_for_role = lambda role: [(domain_hints[var_sig(attr)][1], attr) \
                    for attr in all_vars if domain_hints[var_sig(attr)][0] == role]
            
            available = [attr for place, attr in sorted(attrs_for_role("available"))]
            used = [attr for place, attr in sorted(attrs_for_role("used"))]
            
            self.available_attrsA[:] = available
            self.used_attrsA[:] = used
        else:
            self.available_attrsA[:] = []
            self.used_attrsA[:] = []
        
        self.updateInfoA()
        self.commit()


    def set_dataB(self, data=None):
        self.update_domainB_role_hints()
        self.closeContext("B")
        self.dataB = data
        if data is not None:
            self.openContext("B", data)
            all_vars = data.domain.variables + data.domain.getmetas().values()
            
            var_sig = lambda attr: (attr.name, attr.varType)
            
            domain_hints = dict([(var_sig(attr), ("available", i)) \
                            for i, attr in enumerate(data.domain.attributes)])
            
            domain_hints.update(dict([(var_sig(attr), ("available", i)) \
                for i, attr in enumerate(data.domain.getmetas().values())]))
            
            if data.domain.class_var:
                domain_hints[var_sig(data.domain.class_var)] = ("available", 0)
                    
            domain_hints.update(self.domainB_role_hints) # update the hints from context settings
            
            attrs_for_role = lambda role: [(domain_hints[var_sig(attr)][1], attr) \
                    for attr in all_vars if domain_hints[var_sig(attr)][0] == role]
            
            available = [attr for place, attr in sorted(attrs_for_role("available"))]
            used = [attr for place, attr in sorted(attrs_for_role("used"))]
            
            self.available_attrsB[:] = available
            self.used_attrsB[:] = used
        else:
            self.available_attrsB[:] = []
            self.used_attrsB[:] = []

        self.updateInfoB()
        self.commit()


    def updateInfoA(self):
        """Updates data A info box.
        """
        if self.dataA:
            self.lblDataAExamples.setText("%s example%s" % self._sp(self.dataA))
            self.lblDataAAttributes.setText("%s attribute%s" % self._sp(self.available_attrsA[:] + self.used_attrsA[:]))
        else:
            self.lblDataAExamples.setText("No data on input A.")
            self.lblDataAAttributes.setText("")


    def updateInfoB(self):
        """Updates data B info box.
        """
        if self.dataB:
            self.lblDataBExamples.setText("%s example%s" % self._sp(self.dataB))
            self.lblDataBAttributes.setText("%s attribute%s" % self._sp(self.available_attrsB[:] + self.used_attrsB[:]))
        else:
            self.lblDataBExamples.setText("No data on input B.")
            self.lblDataBAttributes.setText("")


    def update_domainA_role_hints(self):
        """ Update the domain hints to be stored in the widgets settings.
        """
        hints_from_model = lambda role, model: \
                [((attr.name, attr.varType), (role, i)) \
                 for i, attr in enumerate(model)]
        
        hints = dict(hints_from_model("available", self.available_attrsA))
        hints.update(hints_from_model("used", self.used_attrsA))
        self.domainA_role_hints = hints

    def update_domainB_role_hints(self):
        """ Update the domain hints to be stored in the widgets settings.
        """
        hints_from_model = lambda role, model: \
                [((attr.name, attr.varType), (role, i)) \
                 for i, attr in enumerate(model)]
        
        hints = dict(hints_from_model("available", self.available_attrsB))
        hints.update(hints_from_model("used", self.used_attrsB))
        self.domainB_role_hints = hints

    def move_rows(self, view, rows, offset):
        model = view.model()
        newrows = [min(max(0, row + offset), len(model) - 1) for row in rows]
        
        for row, newrow in sorted(zip(rows, newrows), reverse=offset > 0):
            model[row], model[newrow] = model[newrow], model[row]
            
        selection = QItemSelection()
        for nrow in newrows:
            index = model.index(nrow, 0)
            selection.select(index, index)
        view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def move_up(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, -1)
    
    def move_down(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, 1)
        
    def selected_rows(self, view):
        """ Return the selected rows in the view. 
        """
        rows = view.selectionModel().selectedRows()
        model = view.model()
        return [r.row() for r in rows]

    def move_selected(self, view):
        fromto = {
            self.available_attrsA_view: self.used_attrsA_view,
            self.available_attrsB_view: self.used_attrsB_view,
            self.used_attrsA_view: self.available_attrsA_view,
            self.used_attrsB_view: self.available_attrsB_view
        }
        if self.selected_rows(view):
            self.move_selected_from_to(view, fromto[view])
        else:
            self.move_selected_from_to(fromto[view], view)
    
    def move_selected_from_to(self, src, dst):
        self.move_from_to(src, dst, self.selected_rows(src))
        
    def move_from_to(self, src, dst, rows):
        src_model = src.model()
        attrs = [src_model[r] for r in rows]

        for s1, s2 in reversed(list(slices(rows))):
            del src_model[s1:s2]

        dst_model = dst.model()
        dst_model.extend(attrs)


    def reset(self):
        if self.dataA is not None:
            meta_attrsA = self.dataA.domain.getmetas().values()
            self.available_attrsA[:] = self.dataA.domain.variables + meta_attrsA
            self.used_attrsA[:] = []
            self.update_domainA_role_hints()
        if self.dataB is not None:
            meta_attrsB = self.dataB.domain.getmetas().values()
            self.available_attrsB[:] = self.dataB.domain.variables + meta_attrsB
            self.used_attrsB[:] = []
            self.update_domainB_role_hints()


    def update_interface_state(self, focus=None, selected=None, deselected=None):
        for view in [self.available_attrsA_view, self.used_attrsA_view,
                     self.available_attrsB_view, self.used_attrsB_view]:
            if view is not focus and not view.hasFocus():
                view.selectionModel().clear()
                
        availableA_selected = bool(self.available_attrsA_view.selectionModel().selectedRows())
        availableB_selected = bool(self.available_attrsB_view.selectionModel().selectedRows())
                              #bool(self.selected_rows(self.available_attrs_view))
                                 
        move_attrA_enabled = bool(availableA_selected or \
                                  self.used_attrsA_view.selectionModel().selectedRows())
        move_attrB_enabled = bool(availableB_selected or \
                                  self.used_attrsB_view.selectionModel().selectedRows())

        self.move_attrA_button.setEnabled(move_attrA_enabled)
        self.move_attrB_button.setEnabled(move_attrB_enabled)
        if move_attrA_enabled:
            self.move_attrA_button.setText(">" if availableA_selected else "<")
        if move_attrB_enabled:
            self.move_attrB_button.setText(">" if availableB_selected else "<")

    def commit(self):
        if self.dataA: self.update_domainA_role_hints()
        if self.dataB: self.update_domainB_role_hints()
        self.error(0)
        if self.dataA and self.dataB and list(self.used_attrsA) and list(self.used_attrsB):
            try:
                self.send("Merged Data A+B", self.merge(self.dataA, self.dataB, self.used_attrsA, self.used_attrsB))
                self.send("Merged Data B+A", self.merge(self.dataB, self.dataA, self.used_attrsB, self.used_attrsA))
            except Orange.core.KernelException, ex:
                self.error(0, "Cannot merge the two tables (%r)" % str(ex))
        else:
            self.send("Merged Data A+B", None)
            self.send("Merged Data B+A", None)

    ############################################################################################################################################################
    ## Utility functions
    ############################################################################################################################################################

    def _sp(self, l, capitalize=True):
        """Input: list; returns tuple (str(len(l)), "s"/"")
        """
        n = len(l)
        if n == 0:
            if capitalize:
                return "No", "s"
            else:
                return "no", "s"
        elif n == 1:
            return str(n), ''
        else:
            return str(n), 's'

    def merge(self, dataA, dataB, varsA, varsB):
        """ Merge two tables
        """
        
        vkey = lambda row, vs : tuple( row[v].native() for v in vs )
        
        val2idx = dict([( vkey(e,varsB) , i) for i, e in reversed(list(enumerate(dataB)))])
        
        #for key in ["?", "~", ""]:
        #    if key in val2idx:
        #        val2idx.pop(key)

        metasA = dataA.domain.getmetas().items()
        metasB = dataB.domain.getmetas().items()
        
        includedAttsB = [attrB for attrB in dataB.domain if attrB not in dataA.domain]
        includedMetaB = [(mid, meta) for mid, meta in metasB if (mid, meta) not in metasA]
        includedClassVarB = dataB.domain.classVar and dataB.domain.classVar not in dataA.domain
        
        reducedDomainB = Orange.data.Domain(includedAttsB, includedClassVarB)
        reducedDomainB.addmetas(dict(includedMetaB))
        
        
        mergingB = Orange.data.Table(reducedDomainB)
        
        for ex in dataA:
            ind = val2idx.get( vkey(ex,varsA), None)
            if ind is not None:
                mergingB.append(Orange.data.Instance(reducedDomainB, dataB[ind]))
                
            else:
                mergingB.append(Orange.data.Instance(reducedDomainB, ["?"] * len(reducedDomainB)))
                
        return Orange.data.Table([dataA, mergingB])
    
if __name__=="__main__":
    import sys
    a=QApplication(sys.argv)
    ow=OWMultiMergeData()
    ow.show()
    data = Orange.data.Table("iris.tab")
    data2 = Orange.data.Table("iris.tab")
    ow.set_dataA(data)
    ow.set_dataB(data2)
    a.exec_()


