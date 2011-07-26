import OWGUI

class OWPlotGUI:
    def __init__(self, plot):
        self._plot = plot
        
    def antialiasing_check_box(self, widget):
        OWGUI.checkBox(widget, self._plot, 'use_antialiasing', 'Use antialiasing', callback = self._plot.update_antialiasing)
        
    def show_legend_check_box(self, widget):
        OWGUI.checkBox(widget, self._plot, 'show_legend', 'Show legend', callback = self._plot.update_legend)
    
    def filled_symbols_check_box(self, widget):
        OWGUI.checkBox(widget, self._plot, 'show_filled_symbols', 'Show filled symbols', callback = self._plot.update_filled_symbols)

