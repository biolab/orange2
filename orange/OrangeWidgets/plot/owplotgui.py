import OWGUI

class OWPlotGUI:
    def __init__(self, plot):
        self._plot = plot
        
    Antialiasing = 1
    ShowLegend = 2
    ShowFilledSymbols = 3
    ShowGridLines = 4
    PointSize = 5
    AlphaValue = 6
        
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
        
    def _slider(self, widget, value, label, min_value, max_value, step, cb_name):
        OWGUI.hSlider(widget, self._plot, value, label=label, minValue=min_value, maxValue=max_value, step=step, callback=self._get_callback(cb_name))
        
    def point_size_slider(self, widget):
        self._slider(widget, 'point_width', "Symbol size:   ", 1, 20, 1, 'update_point_size')
        
    def alpha_value_slider(self, widget):
        self._slider(widget, 'alpha_value', "Transparency: ", 0, 255, 10, 'update_alpha_value')
        
    def point_properties_box(self, widget):
        return self.create_box([self.PointSize, self.AlphaValue], widget, "Point Properties")
        
    _functions = {
        Antialiasing : antialiasing_check_box,
        ShowLegend : show_legend_check_box,
        ShowFilledSymbols : filled_symbols_check_box,
        ShowGridLines : grid_lines_check_box,
        PointSize : point_size_slider,
        AlphaValue : alpha_value_slider
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
