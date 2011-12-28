"""
<name>Correlations</name>
<description>Compute all pairwise attribute correlations</description>
<icon>icons/Correlations.png</icon>
<contact>ales.erjavec(@ at @)fri.uni-lj.si</contact>

"""

from OWWidget import *

import OWGUI
import OWGraph

import Orange
from Orange.data import variable

def is_continuous(var):
    return isinstance(var, variable.Continuous)

def is_discrete(var):
    return isinstance(var, variable.Discrete)

def pairwise_pearson_correlations(data, vars=None):
    if vars is None:
        vars = list(data.domain.variables)
        
    matrix = Orange.core.SymMatrix(len(vars))
    
    for i in range(len(vars)):
        for j in  range(i + 1, len(vars)):
            matrix[i, j] = Orange.core.PearsonCorrelation(vars[i], vars[j], data, 0).r
            
    return matrix

def pairwise_spearman_correlations(data, vars=None):
    import numpy
    import statc
    
    if vars is None:
        vars = list(data.domain.variables)
    
    matrix = Orange.core.SymMatrix(len(vars))
    
    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]
    (data,) = data.to_numpy_MA("Ac")
    
    averages = numpy.ma.average(data, axis=0)
    
    for i, var_i in enumerate(indices):
        for j, var_j in enumerate(indices[i + 1:], i + 1):
            a = data[:, var_i].filled(averages[var_i])
            b = data[:, var_j].filled(averages[var_j])
            matrix[i, j] = statc.spearmanr(list(a), list(b))[0]
            
    return matrix

def target_pearson_correlations(data, vars=None, target_var=None):
    if vars is None:
        vars = list(data.domain.variables)
    
    if target_var is None:
        if is_continuous(data.domain.class_var):
            target_var = data.domain.class_var
        else:
            raise ValueError("A data with continuous class variable expected if 'target_var' is not explicitly declared.")
    
    correlations = []
    for var in vars:
        correlations.append(Orange.core.PearsonCorrelation(var, target_var, data, 0).r)
        
    return correlations


def target_spearman_correlations(data, vars=None, target_var=None):
    import numpy
    import statc
    
    if vars is None:
        vars = list(data.domain.variables)
    
    if target_var is None:
        if is_continuous(data.domain.class_var):
            target_var = data.domain.class_var
        else:
            raise ValueError("A data with continuous class variable expected if 'target_var' is not explicitly declared.")
    
    all_vars = list(data.domain.variables)
    indices = [all_vars.index(v) for v in vars]
    target_index = all_vars.index(target_var)
    (data,) = data.to_numpy_MA("Ac")
    
    averages = numpy.ma.average(data, axis=0)
    target_values = data[:, target_index].filled(averages[target_index])
    target_values = list(target_values)
    
    correlations = []
    for i, var_i in enumerate(indices):
        a = data[:,var_i].filled(averages[var_i])
        correlations.append(statc.spearmanr(list(a), target_values)[0])
        
    return correlations

    
def matrix_to_table(matrix, items=None):
    from Orange.data import variable
    if items is None:
        items = getattr(matrix, "items", None)
    if items is None:
        items = range(matrix.dim)
        
    items = map(str, items)
    
    attrs = [variable.Continuous(name) for name in items]
    domain = Orange.data.Domain(attrs, None)
    row_name = variable.String("Row name")
    domain.addmeta(Orange.data.new_meta_id(), row_name)
    
    table = Orange.data.Table(domain, [list(r) for r in matrix])
    for item, row in zip(items, table):
        row[row_name] = item
        
    return table

class CorrelationsItemDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        v = value.toPyObject()
        if isinstance(v, float):
            return QString("%.4f" % v)
        else:
            return QStyledItemDelegate.displayText(value, locale)
        
class CorrelationsTableView(QTableView):
    def sizeHint(self):
        hint = QTableView.sizeHint(self)
        h_header = self.horizontalHeader()
        v_header = self.verticalHeader()
        width = v_header.width() + h_header.length() + 4 + self.verticalScrollBar().width()
        height = v_header.length() + h_header.height() + 4 + self.horizontalScrollBar().height()
        return QSize(width, height)
    
class OWCorrelations(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["selected_index"])}
    
    settingsList = ["correlations_type", "pairwise_correlations", "splitter_state"]
    
    COR_TYPES = ["Pairwise Pearson correlation",
                 "Pairwise Spearman correlation",
                 "Correlate with class"
                 ]
    def __init__(self, parent=None, signalManager=None, title="Correlations"):
        # Call OWBaseWidget constructor to bypass OWWidget layout
        OWBaseWidget.__init__(self, parent, signalManager, title,)
#                          wantMainAre=False, noReport=True)
        
        self.inputs = [("Example Table", Orange.data.Table, self.set_data)]
        
        self.outputs = [("Correlations", Orange.data.Table),
                        ("Variables", AttributeList)]
        
        # Settings
        
        self.pairwise_correlations = True
        self.correlations_type = 0
        
        self.selected_index = None
        self.changed_flag = False
        self.auto_commit = True
        
        self.splitter_state = None
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        layout = QVBoxLayout(self)
        layout.setMargin(4)
        self.setLayout(layout)
        
        self.splitter = QSplitter()
        self.layout().addWidget(self.splitter)
        self.splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.controlArea = OWGUI.widgetBox(self.splitter, addToLayout=False)
        self.splitter.addWidget(self.controlArea)
        self.mainArea = OWGUI.widgetBox(self.splitter, addToLayout=False)
        self.mainArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.splitter.addWidget(self.mainArea)
        self.splitter.setSizes([1,1])
        
        if self.splitter_state is not None:
            try:
                self.splitter.restoreState(QByteArray(self.splitter_state))
            except Exception, ex:
                pass
            
        self.splitter.splitterMoved.connect(
                        self.on_splitter_moved
                        )
        
        box = OWGUI.widgetBox(self.controlArea, "Correlations")
        self.corr_radio_buttons = OWGUI.radioButtonsInBox(box, 
                                self, "correlations_type",
                                btnLabels=self.COR_TYPES, 
                                callback=self.on_corr_type_change,
                                )
        
        self.corr_table = CorrelationsTableView()
        self.corr_table.setSelectionMode(QTableView.SingleSelection)
        self.corr_table.setItemDelegate(CorrelationsItemDelegate(self))
        self.corr_table.setEditTriggers(QTableView.NoEditTriggers)
        self.corr_table.horizontalHeader().sectionClicked.connect(
                    self.on_horizontal_header_section_click
                    )
        self.corr_table.verticalHeader().sectionClicked.connect(
                    self.on_vertical_header_section_click
                    )
        self.corr_model = QStandardItemModel()
        self.corr_table.setModel(self.corr_model)
        
        self.corr_table.selectionModel().selectionChanged.connect(
                    self.on_table_selection_change
                    )
        
        self.controlArea.layout().addWidget(self.corr_table)
        
        self.corr_graph = CorrelationsGraph(self)
        self.corr_graph.showFilledSymbols = False
        
        self.mainArea.layout().addWidget(self.corr_graph)
        
        self.clear()
        
        self.resize(1000, 600)
        
    def clear(self):
        self.data = None
        self.cont_vars = None
        self.var_names = None
        self.selected_vars = None
        self.clear_computed()
        self.clear_graph()
        
    def clear_computed(self):
        self.corr_model.clear()
        self.set_all_pairwise_matrix(None, None)
        self.set_target_correlations(None, None)
        
    def clear_graph(self):
        self.corr_graph.clear()
        self.corr_graph.setData(None, None)
        self.corr_graph.replot()
        
    def set_data(self, data=None):
        self.closeContext("")
        self.clear()
        self.information(0)
        self.data = data
        if data is not None and len(filter(is_continuous, data.domain)) >= 2:
            self.set_variables_list(data)
            self.selected_index = None
            self.corr_graph.setData(data)
            self.openContext("", data)
            
            b = self.corr_radio_buttons.buttons[-1]
            if not is_continuous(data.domain.class_var):
                self.correlations_type = min(self.correlations_type, 1)
                b.setEnabled(False)
            else:
                b.setEnabled(True)
                
            if self.selected_index is None or \
                    any(n in self.data.domain for n in self.selected_index):
                self.selected_index = self.var_names[:2]
                
            self.run()
                
        elif data is not None:
            self.data = None
            self.information(0, "Need data with at least 2 continuous variables.")
            
        self.commit_if()
            
    def set_variables_list(self, data):
        vars = list(data.domain.variables)
        vars = [v for v in vars if is_continuous(v)]
        self.cont_vars = vars
        self.var_names = [v.name for v in vars]
    
    @property
    def target_variable(self):
        if self.data:
            return self.data.domain.class_var
        else:
            return None
        
    def run(self):
        if self.correlations_type < 2:
            if self.correlations_type == 0:
                matrix = pairwise_pearson_correlations(self.data, self.cont_vars)
            elif self.correlations_type == 1:
                matrix = pairwise_spearman_correlations(self.data, self.cont_vars)
            self.set_all_pairwise_matrix(matrix)
            
        elif is_continuous(self.target_variable):
            vars = [v for v in self.cont_vars if v != self.target_variable]
            p_corr = target_pearson_correlations(self.data, vars, self.target_variable)
            s_corr = target_spearman_correlations(self.data, vars, self.target_variable)
            correlations = map(list, zip(p_corr, s_corr))
            self.set_target_correlations(correlations, vars, self.target_variable)
            
    def set_all_pairwise_matrix(self, matrix, vars=None):
        self.matrix = matrix
        if matrix is not None:
            for i, row in enumerate(matrix):
                for j, e in enumerate(row):
                    item = QStandardItem()
                    if i != j:
                        item.setData(e, Qt.DisplayRole)
                    else:
                        item.setData(QVariant(QColor(Qt.gray)), Qt.BackgroundRole)
                    self.corr_model.setItem(i, j, item)
                    
            if vars is None:
                vars = self.cont_vars
            header = [v.name for v in vars]
            self.corr_model.setVerticalHeaderLabels(header)
            self.corr_model.setHorizontalHeaderLabels(header)
            
            self.corr_table.resizeColumnsToContents()
            self.corr_table.resizeRowsToContents()
            
            QTimer.singleShot(100, self.corr_table.updateGeometry)
#            self.corr_table.updateGeometry()
    
    def set_target_correlations(self, correlations, vars=None, target_var=None):
        self.target_correlations = correlations
        if correlations is not None:
            for i, row in enumerate(correlations):
                for j, c in enumerate(row):
                    item = QStandardItem()
                    item.setData(c, Qt.DisplayRole)
                    self.corr_model.setItem(i, j, item)
                
            if vars is None:
                vars = self.cont_vars
            
            v_header = [v.name for v in vars]
            h_header = ["Pearson", "Spearman"]
            self.corr_model.setVerticalHeaderLabels(v_header)
            self.corr_model.setHorizontalHeaderLabels(h_header)
            
            self.corr_table.resizeColumnsToContents()
            self.corr_table.resizeRowsToContents()
            
            QTimer.singleShot(100, self.corr_table.updateGeometry)
#            self.corr_table.updateGeometry()
            
    def set_selected_vars(self, x, y):
        x = self.cont_vars.index(x)
        y = self.cont_vars.index(y)
        if self.correlations_type == 2:
            y = 0
        
        model = self.corr_model
        sel_model = self.corr_table.selectionModel()
        sel_model.select(model.index(x, y),
                         QItemSelectionModel.ClearAndSelect)
    
    def on_corr_type_change(self):
        if self.data is not None:
            curr_selection = self.selected_vars
            self.clear_computed()
            self.run()
            
            if curr_selection:
                try:
                    self.set_selected_vars(*curr_selection)
                except Exception, ex:
                    import traceback
                    traceback.print_exc()
            
            self.commit_if()
        
    def on_table_selection_change(self, selected, deselected):
        indexes = self.corr_table.selectionModel().selectedIndexes()
        if indexes:
            index = indexes[0]
            i, j = index.row(), index.column()
            if self.correlations_type == 2 and is_continuous(self.target_variable):
                j = len(self.var_names) - 1
            
            self.corr_graph.updateData(self.var_names[i],
                           self.var_names[j],
                           self.data.domain.class_var.name \
                           if is_discrete(self.data.domain.class_var) else \
                           "(Same color)")
            vars = [self.cont_vars[i], self.cont_vars[j]]
        else:
            # TODO: Clear graph
            vars = None
        self.selected_vars = vars
        
        self.send("Variables", vars)
        
    def on_horizontal_header_section_click(self, section):
        sel_model = self.corr_table.selectionModel()
        indexes = sel_model.selectedIndexes()
        if indexes:
            index = indexes[0]
            i, j = index.row(), index.column()
            sel_index = self.corr_model.index(i, section)
            sel_model.setCurrentIndex(sel_index,
                                QItemSelectionModel.ClearAndSelect)
            
    def on_vertical_header_section_click(self, section):
        sel_model = self.corr_table.selectionModel()
        indexes = sel_model.selectedIndexes()
        if indexes:
            index = indexes[0]
            i, j = index.row(), index.column()
            sel_index = self.corr_model.index(section, j)
            sel_model.setCurrentIndex(sel_index,
                                QItemSelectionModel.ClearAndSelect)
            
    def on_splitter_moved(self, *args):
        self.splitter_state = str(self.splitter.saveState())
            
    def commit_if(self):
        if self.auto_commit:
            self.commit()
        else:
            self.changed_flag = True
    
    def commit(self):
        table = None
        if self.data is not None:
            if self.correlations_type == 2 and \
                    is_continuous(self.target_variable):
                pearson, _ = variable.make("Pearson", Orange.core.VarTypes.Continuous)
                spearman, _ = variable.make("Spearman", Orange.core.VarTypes.Continuous)
                row_name, _ = variable.make("Variable", Orange.core.VarTypes.String)
                
                domain = Orange.data.Domain([pearson, spearman], None)
                domain.addmeta(Orange.data.new_meta_id(), row_name)
                table = Orange.data.Table(domain, self.target_correlations)
                for inst, name in zip(table, self.var_names):
                    inst[row_name] = name
#            else:
#                table = matrix_to_table(self.matrix, self.var_names)
        
        self.send("Correlations", table)
        
from OWScatterPlotGraph import OWScatterPlotGraph

class CorrelationsGraph(OWScatterPlotGraph):
    def updateData(self, x_attr, y_attr, *args, **kwargs):
        OWScatterPlotGraph.updateData(self, x_attr, y_attr, *args, **kwargs)
        if not hasattr(self, "regresson_line"):
            self.regression_line = self.addCurve("regresson_line",
                                                  style=OWGraph.QwtPlotCurve.Lines,
                                                  symbol=OWGraph.QwtSymbol.NoSymbol,
                                                  autoScale=True)
        if isinstance(x_attr, basestring):
            x_index = self.attribute_name_index[x_attr]
        else:
            x_index = x_attr
            
        if isinstance(y_attr, basestring):
            y_index = self.attribute_name_index[y_attr]
        else:
            y_index = y_attr
        
        X = self.original_data[x_index]
        Y = self.original_data[y_index]
        
        valid = self.getValidList([x_index, y_index])
        
        X = X[valid]
        Y = Y[valid]
        x_min, x_max = self.attr_values[x_attr]
        
        import numpy
        X = numpy.array([numpy.ones_like(X), X]).T
        try:
            beta, _, _, _ = numpy.linalg.lstsq(X, Y)
        except numpy.linalg.LinAlgError:
            beta = [0, 0]
        
        y1 = beta[0] + x_min * beta[1]
        y2 = beta[0] + x_max * beta[1]
        
        self.regression_line.setData([x_min, x_max], [y1, y2])        
        self.replot()

def main():
    import sys
    app = QApplication(sys.argv)
    data = Orange.data.Table("housing")
#    data = Orange.data.Table("iris")
    w = OWCorrelations()
    w.set_data(None)
    w.set_data(data)
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
    
