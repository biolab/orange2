"""
Preview Browser Widget.

"""

from PyQt4.QtGui import (
    QWidget, QLabel, QListView, QAction, QVBoxLayout, QHBoxLayout, QSizePolicy
)

from PyQt4.QtSvg import QSvgWidget

from PyQt4.QtCore import (
    Qt, QSize, QByteArray, QModelIndex
)

from PyQt4.QtCore import pyqtSignal as Signal

from ..gui.dropshadow import DropShadowFrame
from . import previewmodel


NO_PREVIEW_SVG = """

"""


# Default description template
DESCRIPTION_TEMPLATE = """
<h3 class=item-heading>{name}</h3>
<p class=item-description>
{description}
</p>

"""

PREVIEW_SIZE = (440, 295)


class LinearIconView(QListView):
    def __init__(self, *args, **kwargs):
        QListView.__init__(self, *args, **kwargs)

        self.setViewMode(QListView.IconMode)
        self.setWrapping(False)
        self.setSelectionMode(QListView.SingleSelection)
        self.setEditTriggers(QListView.NoEditTriggers)
        self.setMovement(QListView.Static)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding,
                           QSizePolicy.Fixed)

        self.setIconSize(QSize(120, 80))

    def sizeHint(self):
        if not self.model().rowCount():
            return QSize(200, 140)
        else:
            height = self.sizeHintForRow(0)
            _, top, _, bottom = self.getContentsMargins()
            return QSize(200, height + top + bottom + self.verticalOffset())


class PreviewBrowser(QWidget):
    """A Preview Browser for recent/premade scheme selection.
    """
    # Emitted when the current previewed item changes
    currentIndexChanged = Signal(int)

    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.__model = None
        self.__currentIndex = -1
        self.__template = DESCRIPTION_TEMPLATE
        self.__setupUi()

    def __setupUi(self):
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(12, 12, 12, 12)

        # Top row with full text description and a large preview
        # image.
        self.__label = QLabel(self, objectName="description-label",
                              wordWrap=True,
                              alignment=Qt.AlignTop | Qt.AlignLeft)

        self.__label.setWordWrap(True)
        self.__label.setFixedSize(220, PREVIEW_SIZE[1])

        self.__image = QSvgWidget(self, objectName="preview-image")
        self.__image.setFixedSize(*PREVIEW_SIZE)

        self.__imageFrame = DropShadowFrame(self)
        self.__imageFrame.setWidget(self.__image)

        self.__path = QLabel(self, objectName="path-label")
        self.__path.setWordWrap(False)
        self.__path.setContentsMargins(12, 0, 12, 0)

        self.__selectAction = \
            QAction(self.tr("Select"), self,
                    objectName="select-action",
                    )

        top_layout.addWidget(self.__label, 1,
                             alignment=Qt.AlignTop | Qt.AlignLeft)
        top_layout.addWidget(self.__image, 1,
                             alignment=Qt.AlignTop | Qt.AlignRight)

        vlayout.addLayout(top_layout)
        vlayout.addWidget(self.__path)

        # An list view with small preview icons.
        self.__previewList = LinearIconView(objectName="preview-list-view")

        vlayout.addWidget(self.__previewList)
        self.setLayout(vlayout)

    def setModel(self, model):
        """Set the item model for preview.
        """
        if self.__model != model:
            if self.__model:
                s_model = self.__previewList.selectionModel()
                s_model.selectionChanged.disconnect(self.__onSelectionChanged)
                self.__model.dataChanged.disconnect(self.__onDataChanged)

            self.__model = model
            self.__previewList.setModel(model)

            if model:
                s_model = self.__previewList.selectionModel()
                s_model.selectionChanged.connect(self.__onSelectionChanged)
                self.__model.dataChanged.connect(self.__onDataChanged)

            if model and model.rowCount():
                self.setCurrentIndex(0)

    def model(self):
        """Return the item model.
        """
        return self.__model

    def setPreviewDelegate(self, delegate):
        """Set the delegate to render the preview images.
        """
        raise NotImplementedError

    def setDescriptionTemplate(self, template):
        self.__template = template
        self.__update()

    def setCurrentIndex(self, index):
        """Set the selected preview item index.
        """
        if self.__model is not None and self.__model.rowCount():
            index = min(index, self.__model.rowCount() - 1)
            index = self.__model.index(index, 0)
            sel_model = self.__previewList.selectionModel()
            # This emits selectionChanged signal and triggers
            # __onSelectionChanged, currentIndex is updated there.
            sel_model.select(index, sel_model.ClearAndSelect)

        elif self.__currentIndex != -1:
            self.__currentIndex = -1
            self.__update()
            self.currentIndexChanged.emit(-1)

    def currentIndex(self):
        """Return the current selected index.
        """
        return self.__currentIndex

    def __onSelectionChanged(self, *args):
        """Selected item in the preview list has changed.
        Set the new description and large preview image.

        """
        rows = self.__previewList.selectedIndexes()
        if rows:
            index = rows[0]
            self.__currentIndex = index.row()
        else:
            index = QModelIndex()
            self.__currentIndex = -1

        self.__update()
        self.currentIndexChanged.emit(self.__currentIndex)

    def __onDataChanged(self, topleft, bottomRight):
        """Data changed, update the preview if current index in the changed
        range.

        """
        if self.__currentIndex <= topleft.row() and \
                self.__currentIndex >= bottomRight.row():
            self.__update()

    def __update(self):
        """Update the description.
        """
        if self.__currentIndex != -1:
            index = self.model().index(self.__currentIndex, 0)
        else:
            index = QModelIndex()

        if not index.isValid():
            description = ""
            name = ""
            path = ""
            svg = NO_PREVIEW_SVG
        else:
            description = unicode(index.data(Qt.WhatsThisRole).toString())
            if not description:
                description = "No description."

            name = unicode(index.data(Qt.DisplayRole).toString())
            if not name:
                name = "Untitled"

            path = unicode(index.data(Qt.StatusTipRole).toString())

            svg = unicode(index.data(previewmodel.ThumbnailSVGRole).toString())

        desc_text = self.__template.format(description=description, name=name)

        self.__label.setText(desc_text)

        path_text = "<b>Path:</b><div class=item-path>{0}</div>".format(path)
        self.__path.setText(path_text)

        if not svg:
            svg = NO_PREVIEW_SVG

        if svg:
            self.__image.load(QByteArray(svg))
