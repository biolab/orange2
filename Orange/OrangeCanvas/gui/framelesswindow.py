"""
A Frameless window widget

"""

from PyQt4.QtGui import QWidget, QPalette, QPainter, QStyleOption

from PyQt4.QtCore import Qt, pyqtProperty as Property

from .utils import is_transparency_supported, StyledWidget_paintEvent


class FramelessWindow(QWidget):
    """A Basic frameless window widget with rounded corners (if supported by
    the windowing system).

    """
    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.__radius = 6
        self.__isTransparencySupported = is_transparency_supported()
        self.setAttribute(Qt.WA_TranslucentBackground,
                          self.__isTransparencySupported)

    def setRadius(self, radius):
        """Set the window rounded border radius.
        """
        if self.__radius != radius:
            self.__radius = radius
            self.update()

    def radius(self):
        return self.__radius

    radius_ = Property(int, fget=radius, fset=setRadius)

    def paintEvent(self, event):
        if self.__isTransparencySupported:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)

            opt = QStyleOption()
            opt.init(self)
            rect = opt.rect
            p.setBrush(opt.palette.brush(QPalette.Window))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, self.__radius, self.__radius)
            p.end()
        else:
            StyledWidget_paintEvent(self, event)
