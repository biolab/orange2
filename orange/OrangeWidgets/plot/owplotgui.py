'''
    
.. index:: plot

######################################
GUI elements for plots (``owplotgui``)
######################################

.. autoclass:: OrientedWidget
    :show-inheritance:
    
.. autoclass:: StateButtonContainer
    :show-inheritance:
    
.. autoclass:: OWToolbar
    :show-inheritance:
    
.. autoclass:: OWButton
    :show-inheritance:

.. autoclass:: OrangeWidgets.plot.OWPlotGUI
    :members:

'''

import os
import OWGUI

from owconstants import *

from PyQt4.QtGui import QWidget, QToolButton, QGroupBox, QVBoxLayout, QHBoxLayout, QIcon
from PyQt4.QtCore import Qt, pyqtSignal, qDebug


class OrientedWidget(QWidget):
    '''
        A simple QWidget with a box layout that matches its ``orientation``. 
    '''
    def __init__(self, orientation, parent):
        QWidget.__init__(self, parent)
        if orientation == Qt.Vertical:
            self._layout = QVBoxLayout()
        else:
            self._layout = QHBoxLayout()
        self.setLayout(self._layout)

class OWToolbar(OrientedWidget):
    '''
        A toolbar is a container that can contain any number of buttons.  
        
        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`
        
        :param text: The name of this toolbar
        :type text: str
        
        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int
        
        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)
        
        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    '''
    def __init__(self, gui, text, orientation, buttons, parent):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        self.groups = {}
        i = 0
        n = len(buttons)
        while i < n:
            if buttons[i] == gui.StateButtonsBegin:
                state_buttons = []
                for j in range(i+1, n):
                    if buttons[j] == gui.StateButtonsEnd:
                        qDebug('Adding state buttons ' + repr(state_buttons) + ' to layout ' + repr(self.layout()))
                        s = gui.state_buttons(orientation, state_buttons, self)
                        self.buttons.update(s.buttons)
                        self.groups[buttons[i+1]] = s
                        i = j
                        break
                    else:
                        state_buttons.append(buttons[j])
            elif buttons[i] == gui.Spacing:
                self.layout().addSpacing(10)
            elif type(buttons[i] == int):
                self.buttons[buttons[i]] = gui.tool_button(buttons[i], self)
            elif len(buttons[i] == 4):
                gui.tool_button(buttons[i], self)
            else:
                self.buttons[buttons[i][0]] = gui.tool_button(buttons[i], self)
            i = i + 1
        self.layout().addStretch()


class StateButtonContainer(OrientedWidget):
    '''
        This class can contain any number of checkable buttons, of which only one can be selected at any time. 
    
        :param gui: Used to create containers and buttons
        :type gui: :obj:`.OWPlotGUI`
        
        :param buttons: A list of button identifiers to be added to this toolbar
        :type buttons: list of (int or tuple)
       
        :param orientation: The orientation of this toolbar, either Qt.Vertical or Qt.Horizontal
        :type tex: int
        
        :param parent: The toolbar's parent widget
        :type parent: :obj:`.QWidget`
    '''
    def __init__(self, gui, orientation, buttons, parent):
        OrientedWidget.__init__(self, orientation, parent)
        self.buttons = {}
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._clicked_button = None
        for i in buttons:
            b = gui.tool_button(i, self)
            b.clicked.connect(self.button_clicked)
            self.buttons[i] = b
            self.layout().addWidget(b)
            
    def button_clicked(self, checked):
        sender = self.sender()
        self._clicked_button = sender
        for button in self.buttons.itervalues():
            button.setDown(button is sender)
            
    def button(self, id):
        return self.buttons[id]
        
    def setEnabled(self, enabled):
        OrientedWidget.setEnabled(self, enabled)
        if enabled and self._clicked_button:
            self._clicked_button.click()
                    
class OWButton(QToolButton):
    '''
        A custom tool button that can set and attribute or call a function when clicked. 
        
        :param plot: The object whose attributes will be modified
        :type plot: :obj:`.OWPlot`
        
        :param attr_name: Name of attribute which will be set when the button is clicked. 
                          If this parameter is empty or None, no attribute will be set. 
        :type attr_name: str
        
        :param attr_value: The value that will be assigned to the ``attr_name`` when the button is clicked. 
        :type attr: any
        
        :param callback: Function to be called when the button is clicked. 
                         If a string is passed as ``callback``, a method by that name of ``plot`` will be called. 
                         If this parameter is empty or None, no function will be called
        :type callback: str or function
    '''
    def __init__(self, plot, attr_name, attr_value, callback, parent):
        QToolButton.__init__(self, parent)
        self.setMinimumSize(30, 30)
        self.plot = plot
        if type(callback) == str:
            callback = getattr(plot, callback, None)
        if callback:
            self.clicked.connect(callback)
        if attr_name:
            self.attr_name = attr_name
            self.attr_value = attr_value
            self.clicked.connect(self.set_attribute)
        
    def set_attribute(self, clicked):
        setattr(self.plot, self.attr_name, self.attr_value)
    
    downChanged = pyqtSignal('bool')
    
    def setDown(self, down):
        self.downChanged.emit(down)
        QToolButton.setDown(self, down)
                    
class OWPlotGUI:
    '''
        This class contains functions to create common user interface elements (QWidgets)
        for configuration and interaction with the ``plot``. 
        
        It provides shorter version of some methods in :obj:`.OWGUI` that are directly related to an :obj:`.OWPlot` object. 
    '''
    def __init__(self, plot):
        self._plot = plot
        
    Spacing = 0
        
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
    
    ZoomSelection = 15
    
    SelectionAdd = 21
    SelectionRemove = 22
    SelectionToggle = 23
    SelectionOne = 24
    
    SendSelection = 31
    ClearSelection = 32
    
    StateButtonsBegin = 35
    StateButtonsEnd = 36
    
    UserButton = 100
    
    default_zoom_select_buttons = [
        StateButtonsBegin,
            Zoom,
            Pan, 
            Select,
        StateButtonsEnd,
        Spacing,
        StateButtonsBegin,
            SelectionOne,
            SelectionAdd, 
            SelectionRemove,
        StateButtonsEnd,
        Spacing,
        SendSelection,
        ClearSelection
    ]
    
    _buttons = {
        Zoom : ('Zoom', 'state', ZOOMING, None, 'Dlg_zoom'),
        Pan : ('Pan', 'state', PANNING, None, 'Dlg_pan_hand'),
        Select : ('Select', 'state', SELECT, None, 'Dlg_arrow'),
        SelectionAdd : ('Add to selection', 'selection_behavior', SELECTION_ADD, None, ''),
        SelectionRemove : ('Remove from selection', 'selection_behavior', SELECTION_REMOVE, None, ''),
        SelectionToggle : ('Toggle selection', 'selection_behavior', SELECTION_TOGGLE, None, ''),
        SelectionOne : ('Replace selection', 'selection_behavior', SELECTION_REPLACE, None, ''),
        SendSelection : ('Send selection', None, None, 'send_selection', 'Dlg_send'),
        ClearSelection : ('Clear selection', None, None, 'clear_selection', 'Dlg_clear')
    }
    '''
        The list of built-in buttons. It is a map of 
        id : (name, attr_name, attr_value, callback, icon_name)
        
        .. seealso:: :meth:`.tool_button`
    '''

    def _get_callback(self, name):
        if type(name) == str:
            return getattr(self._plot, name, self._plot.replot)
        else:
            return name
        
    def _check_box(self, widget, value, label, cb_name):
        '''
            Adds a :obj:`.QCheckBox` to ``widget``. 
            When the checkbox is toggled, the attribute ``value`` of the plot object is set to the checkbox' check state,
            and the callback ``cb_name`` is called. 
        '''
        OWGUI.checkBox(widget, self._plot, value, label, callback=self._get_callback(cb_name))    
        
    def antialiasing_check_box(self, widget):
        '''
            Creates a check box that toggles the Antialiasing of the plot 
        '''
        self._check_box(widget, 'use_antialiasing', 'Use antialiasing', 'update_antialiasing')
        
    def show_legend_check_box(self, widget):
        '''
            Creates a check box that shows and hides the plot legend
        '''
        self._check_box(widget, 'show_legend', 'Show legend', 'update_legend')
    
    def filled_symbols_check_box(self, widget):
        self._check_box(widget, 'show_filled_symbols', 'Show filled symbols', 'update_filled_symbols')
        
    def grid_lines_check_box(self, widget):
        self._check_box(widget, 'show_grid', 'Show gridlines', 'update_grid')
    
    def animations_check_box(self, widget):
        '''
            Creates a check box that enabled or disables animations
        '''
        self._check_box(widget, 'use_animations', 'Use animations', 'update_animations')
    
    def _slider(self, widget, value, label, min_value, max_value, step, cb_name):
        OWGUI.hSlider(widget, self._plot, value, label=label, minValue=min_value, maxValue=max_value, step=step, callback=self._get_callback(cb_name))
        
    def point_size_slider(self, widget):
        '''
            Creates a slider that controls point size
        '''
        self._slider(widget, 'point_width', "Symbol size:   ", 1, 20, 1, 'update_point_size')
        
    def alpha_value_slider(self, widget):
        '''
            Creates a slider that controls point transparency
        '''
        self._slider(widget, 'alpha_value', "Transparency: ", 0, 255, 10, 'update_alpha_value')
        
    def point_properties_box(self, widget):
        '''
            Creates a box with controls for common point properties. 
            Currently, these properties are point size and transparency. 
        '''
        return self.create_box([
            self.PointSize, 
            self.AlphaValue
            ], widget, "Point properties")
        
    def plot_settings_box(self, widget):
        '''
            Creates a box with controls for common plot settings
        '''
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
        '''
            Creates a :obj:`.QGroupBox` with text ``name`` and adds it to ``widget``. 
            The ``ids`` argument is a list of widget ID's that will be added to this box
        '''
        box = OWGUI.widgetBox(widget, name)
        self.add_widgets(ids, widget)
        return box
        
    def tool_button(self, id, widget):
        '''
            Creates an :obj:`.OWButton` and adds it to the parent ``widget``. 
            
            :param id: If ``id`` is an ``int``, a button is constructed from the default table. 
                       Otherwise, ``id`` must be tuple of size 4 or 5 with members. The first is optional and 
                       is a unique identifier for the button, the second is its name,
                       the next three are the ``attr_name``, ``attr_value`` and ``callback`` arguments to :obj:`.OWButton`, 
                       and the last one is the icon name
            :type if: int or tuple
            
            :param widget: The button's parent widget
            :type widget: QWidget
        '''
        if type(id) == int:
            name, attr_name, attr_value, callback, icon_name = self._buttons[id]
        elif len(id) == 4:
            name, attr_name, attr_value, callback, icon_name = id
        else:
            id, name, attr_name, attr_value, callback, icon_name = id
        b = OWButton(self._plot, attr_name, attr_value, callback, widget)
        b.setToolTip(name)
        b.setIcon(QIcon(os.path.dirname(__file__) + "/../icons/" + icon_name + '.png'))
        if widget.layout() is not None:
            widget.layout().addWidget(b)
        return b
        
    def state_buttons(self, orientation, buttons, widget):
        '''
            This function creates a set of checkable buttons and connects them so that only one
            may be checked at a time. 
        '''
        c = StateButtonContainer(self, orientation, buttons, widget)
        if widget.layout() is not None:
            widget.layout().addWidget(c)
        return c
        
    def toolbar(self, widget, text, orientation, buttons):
        '''
            Creates an :obj:`.OWToolbar` with the specified ``text``, ``orientation`` and ``buttons`` and adds it to ``widget``. 
            
            .. seealso:: :obj:`.OWToolbar`
        '''
        t = OWToolbar(self, text, orientation, buttons, widget)
        if widget.layout() is not None:
            widget.layout().addWidget(t)
        return t
        
    def zoom_select_toolbar(self, widget, text = 'Zoom / Select', orientation = Qt.Horizontal, buttons = default_zoom_select_buttons):
        t = self.toolbar(widget, text, orientation, buttons)
        t.groups[self.SelectionOne].setEnabled(t.buttons[self.Select].isDown())
        t.buttons[self.Select].downChanged.connect(t.groups[self.SelectionOne].setEnabled)
        t.buttons[self.Select].click()
        t.buttons[self.SelectionOne].click()
        return t
    
