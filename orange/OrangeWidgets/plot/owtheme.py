from PyQt4.Qt import QFont, QColor

class PlotTheme(object):
    '''Collection of color and font settings.'''

    def __init__(self):
        self.labels_font = QFont('Helvetice', 8)
        self.helper_font = self.labels_font
        self.helpers_color = QColor(0, 0, 0, 255)
        self.background_color = QColor(255, 255, 255, 255)
        self.axis_title_font = QFont('Helvetica', 10, QFont.Bold)
        self.axis_font = QFont('Helvetica', 9)
        self.labels_color = QColor(0, 0, 0, 255)
        self.axis_color = QColor(30, 30, 30, 255)
        self.axis_values_color = QColor(30, 30, 30, 255)

