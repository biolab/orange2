import os
import OWGUI

from owconstants import *

from PyQt4.QtGui import QWidget, QToolButton, QGroupBox, QVBoxLayout, QHBoxLayout, QIcon
from PyQt4.QtCore import Qt, pyqtSignal

class OrientedWidget(QWidget):
    def __init__(self, orientation, parent):
        QWidget.__init__(self, parent)
        if orientation == Qt.Vertical:
            self._layout = QVBoxLayout()
        else:
            self._layout = QHBoxLayout()
        self.setLayout(self._layout)

class StateButtonContainer(OrientedWidget):
    def __init__(self, gui, ids, orientation, parent):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        for i in ids:
            b = gui.tool_button(i, self)
            b.clicked.connect(self.button_clicked)
            self.buttons[i] = b
            self.layout().addWidget(b)
            
    def button_clicked(self, checked):
        sender = self.sender()
        for button in self.buttons.itervalues():
            button.setDown(button is sender)
            
    def button(self, id):
        return self.buttons[id]
                    
class AttributeChangeButton(QToolButton):
    def __init__(self, plot, attr_name, attr_value, parent):
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        self.plot = plot
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.clicked.connect(self.button_clicked)
        
    def button_clicked(self, clicked):
        setattr(self.plot, self.attr_name, self.attr_value)
    
    downChanged = pyqtSignal('bool')
    
    def setDown(self, down):
        self.downChanged.emit(down)
        QToolButton.setDown(self, down)
    
class CallbackButton(QToolButton):
    def __init__(self, plot, callback, parent):
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        if type(callback) == str:
            callback = getattr(plot, callback, None)
        if callback:
            self.clicked.connect(callback)
                    
class OWPlotGUI:
    '''
        This class contains functions to create common user interface elements (QWidgets)
        for configuration and interaction with the ``plot``. 
    '''
    def __init__(self, plot):
        self._plot = plot
        
    Antialiasing = 1
    ShowLegend = 2
    ShowFilledSymbols = 3
    ShowGridLines = 4
    PointSize = 5
    AlphaValue = 6
    UseAnimations = 7
    
    Zoom = 11
    Pan = 12
    Select = 13
    
    SelectionAdd = 21
    SelectionRemove = 22
    SelectionToggle = 23
    SelectionOne = 24
    
    SendSelection = 31
    ClearSelection = 32
    
    '''
        A map of 
        id : (name, attr_name, attr_value, icon_name)
    '''
    _attribute_buttons = {
        Zoom : ('Zoom', 'state', ZOOMING, 'Dlg_zoom'),
        Pan : ('Pan', 'state', PANNING, 'Dlg_pan_hand'),
        Select : ('Select', 'state', SELECT, 'Dlg_arrow'),
        SelectionAdd : ('Add to selection', 'selection_behavior', SELECTION_ADD, ''),
        SelectionRemove : ('Remove from selection', 'selection_behavior', SELECTION_REMOVE, ''),
        SelectionToggle : ('Toggle selection', 'selection_behavior', SELECTION_TOGGLE, ''),
        SelectionOne : ('Replace selection', 'selection_behavior', SELECTION_REPLACE, '')
    }
    
    _action_buttons = {
        SendSelection : ('Send selection', None, 'Dlg_send'),
        ClearSelection : ('Clear selection', 'clear_selection', 'Dlg_clear')
    }

    def _get_callback(self, name):
        return getattr(self._plot, name, self._plot.replot)
        
    def _check_box(self, widget, value, label, cb_name):
        OWGUI.checkBox(widget, self._plot, value, label, callback=self._get_callback(cb_name))    
        
    def antialiasing_check_box(self, widget):
        self._check_box(widget, 'use_antialiasing', 'Use antialiasing', 'update_antialiasing')
        
    def show_legend_check_box(self, widget):
        self._check_box(widget, 'show_legend', 'Show legend', 'update_legend')
    
    def filled_symbols_check_box(self, widget):
        self._check_box(widget, 'show_filled_symbols', 'Show filled symbols', 'update_filled_symbols')
        
    def grid_lines_check_box(self, widget):
        self._check_box(widget, 'show_grid', 'Show gridlines', 'update_grid')
    
    def animations_check_box(self, widget):
        self._check_box(widget, 'use_animations', 'Use animations', 'update_animations')
    
    def _slider(self, widget, value, label, min_value, max_value, step, cb_name):
        OWGUI.hSlider(widget, self._plot, value, label=label, minValue=min_value, maxValue=max_value, step=step, callback=self._get_callback(cb_name))
        
    def point_size_slider(self, widget):
        self._slider(widget, 'point_width', "Symbol size:   ", 1, 20, 1, 'update_point_size')
        
    def alpha_value_slider(self, widget):
        self._slider(widget, 'alpha_value', "Transparency: ", 0, 255, 10, 'update_alpha_value')
        
    def point_properties_box(self, widget):
        return self.create_box([
            self.PointSize, 
            self.AlphaValue
            ], widget, "Point properties")
        
    def plot_settings_box(self, widget):
        return self.create_box([
            self.ShowLegend,
            self.ShowFilledSymbols,
            self.ShowGridLines,
            self.UseAnimations,
            self.Antialiasing
            ], widget, "Plot settings")
        
    _functions = {
        Antialiasing : antialiasing_check_box,
        ShowLegend : show_legend_check_box,
        ShowFilledSymbols : filled_symbols_check_box,
        ShowGridLines : grid_lines_check_box,
        PointSize : point_size_slider,
        AlphaValue : alpha_value_slider,
        UseAnimations : animations_check_box
        }
        
    def add_widget(self, id, widget):
        if id in self._functions:
            self._functions[id](self, widget)
            
    def add_widgets(self, ids, widget):
        for id in ids:
            self.add_widget(id, widget)
            
    def create_box(self, ids, widget, name):
        box = OWGUI.widgetBox(widget, name)
        self.add_widgets(ids, widget)
        return box
        
    def tool_button(self, id, widget):
        if id in self._attribute_buttons:
            name, attr_name, attr_value, icon_name = self._attribute_buttons[id]
            b = AttributeChangeButton(self._plot, attr_name, attr_value, widget)
        elif id in self._action_buttons:
            name, cb, icon_name = self._action_buttons[id]
            b = CallbackButton(self._plot, cb, widget)
        else:
            return
        b.setToolTip(name)
        b.setIcon(QIcon(os.path.dirname(__file__) + "/../icons/" + icon_name + '.png'))
        if widget.layout():
            widget.layout().addWidget(b)
        return b
        
    def state_buttons(self, ids, orientation, widget):
        '''
            This function creates a set of checkable buttons and connects them so that only one
            may be checked at a time. 
        '''
        return StateButtonContainer(self, ids, orientation, widget)
        
    def zoom_select_toolbar(self, widget, orientation = Qt.Horizontal, send_selection_callback = None):
        o = 'vertial' if orientation == Qt.Vertical else 'horizontal'
        t = OWGUI.widgetBox(widget, 'Zoom / Select', orientation=o)
        zps = self.state_buttons([OWPlotGUI.Zoom, OWPlotGUI.Pan, OWPlotGUI.Select], orientation, t)
        t.layout().addWidget(zps)
        selection_modes = self.state_buttons([OWPlotGUI.SelectionOne, OWPlotGUI.SelectionAdd, OWPlotGUI.SelectionRemove], orientation, t)
        t.layout().addSpacing(10)
        t.layout().addWidget(selection_modes)
        zps.button(OWPlotGUI.Select).downChanged.connect(selection_modes.setEnabled)
        zps.button(OWPlotGUI.Select).downChanged.connect(selection_modes.button(OWPlotGUI.SelectionOne).click)
        zps.button(OWPlotGUI.Zoom).click()
        t.layout().addSpacing(10)
        self.tool_button(OWPlotGUI.ClearSelection, t)
        b = self.tool_button(OWPlotGUI.SendSelection, t)
        if send_selection_callback:
            b.clicked.connect(send_selection_callback)
        return t
