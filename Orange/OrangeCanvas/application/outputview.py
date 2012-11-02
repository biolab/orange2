"""
"""

from PyQt4.QtGui import (
    QWidget, QPlainTextEdit, QVBoxLayout, QTextCursor
)

from PyQt4.QtCore import Qt


class OutputText(QWidget):
    def __init__(self, parent=None, **kwargs):
        QWidget.__init__(self, parent, **kwargs)

        self.__lines = 5000

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.__text = QPlainTextEdit()
        self.__text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.__text.setMaximumBlockCount(self.__lines)
        font = self.__text.font()
        font.setFamily("Monaco")
        self.__text.setFont(font)

        self.layout().addWidget(self.__text)

    def setMaximumLines(self, lines):
        """Set the maximum number of lines to keep displayed.
        """
        if self.__lines != lines:
            self.__lines = lines
            self.__text.setMaximumBlockCount(lines)

    def maximumLines(self):
        """Return the maximum number of lines in the display.
        """
        return self.__lines

    def clear(self):
        """Clear the displayed text.
        """
        self.__text.clear()

    def toPlainText(self):
        """Return the full contents of the output view.
        """
        return self.__text.toPlainText()

    # A file like interface.
    def write(self, string):
        self.__text.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        self.__text.insertPlainText(string)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass
