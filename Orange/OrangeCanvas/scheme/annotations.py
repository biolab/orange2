"""
Scheme Annotations

"""

from PyQt4.QtCore import QObject
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property


class BaseSchemeAnnotation(QObject):
    """Base class for scheme annotations.
    """
    geometry_changed = Signal()


class SchemeArrowAnnotation(BaseSchemeAnnotation):
    """An arrow annotation in the scheme.
    """

    def __init__(self, start_pos, end_pos, anchor=None, parent=None):
        BaseSchemeAnnotation.__init__(self, parent)
        self.__start_pos = start_pos
        self.__end_pos = end_pos
        self.__anchor = anchor

    def set_line(self, start_pos, end_pos):
        if self.__start_pos != start_pos or self.__end_pos != end_pos:
            self.__start_pos = start_pos
            self.__end_pos = end_pos
            self.geometry_changed.emit()

    def start_pos(self):
        return self.__start_pos

    start_pos = Property(tuple, fget=start_pos)

    def end_pos(self):
        return self.__end_pos

    end_pos = Property(tuple, fget=end_pos)

    def set_geometry(self, (start_pos, end_pos)):
        self.set_line(start_pos, end_pos)

    def geometry(self):
        return (self.start_pos, self.end_pos)

    geometry = Property(tuple, fget=geometry, fset=set_geometry)


class SchemeTextAnnotation(BaseSchemeAnnotation):
    """Text annotation in the scheme.
    """
    text_changed = Signal(unicode)

    def __init__(self, rect, text="", anchor=None, parent=None):
        BaseSchemeAnnotation.__init__(self, parent)
        self.__rect = rect
        self.__text = text
        self.__anchor = anchor

    def set_rect(self, (x, y, w, h)):
        rect = (x, y, w, h)
        if self.__rect != rect:
            self.__rect = rect
            self.geometry_changed.emit()

    def rect(self):
        return self.__rect

    rect = Property(tuple, fget=rect, fset=set_rect)

    def set_geometry(self, rect):
        self.set_rect(rect)

    def geometry(self):
        return self.rect

    geometry = Property(tuple, fget=geometry, fset=set_geometry)

    def set_text(self, text):
        if self.__text != text:
            self.__text = text
            self.text_changed.emit(text)

    def text(self):
        return self.__text

    text = Property(tuple, fget=text, fset=set_text)
