# encoding: utf-8
"""
File Widget
-----------

"""
import os
import csv
import warnings
import itertools
from collections import namedtuple
from StringIO import StringIO

from PyQt4.QtGui import (
    QWidget, QGroupBox, QCheckBox, QComboBox, QLineEdit, QPushButton, QLabel,
    QFrame, QDialog, QDialogButtonBox, QTableView, QRegExpValidator,
    QFormLayout, QVBoxLayout, QHBoxLayout, QStackedLayout, QSizePolicy,
    QStandardItem, QStyle, QApplication, QFileIconProvider, QDesktopServices,
    QFileDialog
)
from PyQt4.QtCore import Qt, QEvent, QRegExp, QFileInfo, QTimer
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange.data
import Orange.feature

from OWWidget import OWWidget, ContextHandler
from OWBaseWidget import _SETTINGS_VERSION_KEY
import OWGUI

from Orange.OrangeWidgets.Data.OWDataTable import ExampleTableModel

NAME = "File"
DESCRIPTION = "Reads data from an input file."
PRIORITY = 15
ICON = "icons/File.svg"
CATEGORY = "Data"
KEYWORDS = ["data", "import", "file", "load", "read"]

OUTPUTS = [
    {"name": "Data",
     "type": Orange.data.Table,
     "doc": "Loaded data set."}
]

MakeStatus = Orange.feature.Descriptor.MakeStatus


# Data format loader options
LoadOptions = namedtuple("LoadOptions", [])
OrangeTab = namedtuple("OrangeTab", ["DK", "DC"])  # Orange tab
OrangeTxt = namedtuple("OrangeTxt", ["DK", "DC"])  # Orange tab simplified
Basket = namedtuple("Basket", [])
C45 = namedtuple("C45", [])
CSV = namedtuple("CSV", ["dialect", "header_format", "missing_values"])

DEFAULT_DK = ""
DEFAULT_DC = ""


class Dialect(csv.Dialect):
    def __init__(self, delimiter, quotechar, escapechar, doublequote,
                 skipinitialspace, quoting=csv.QUOTE_MINIMAL):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace
        self.quoting = quoting
        self.lineterminator = "\r\n"
        csv.Dialect.__init__(self)

NoHeader, PlainHeader, OrangeHeader, SimplifiedOrangeHeader = 0, 1, 2, 3

DEFAULT_OPTIONS = (
    OrangeTab(DEFAULT_DK, DEFAULT_DC),
    OrangeTxt(DEFAULT_DK, DEFAULT_DC),
    Basket(),
    C45(),
    CSV(csv.excel, PlainHeader, None),
    CSV(csv.excel_tab, PlainHeader, None),
)


def _call(f, *args, **kwargs):
    return f(*args, **kwargs)

# Make any KernelWarning raise an error if called through the '_call' function
# defined above.
warnings.filterwarnings(
    "error", ".*", Orange.core.KernelWarning,
    __name__, _call.func_code.co_firstlineno + 1
)


def load_tab(filename, create_new_on=MakeStatus.NoRecognizedValues,
             DK=None, DC=None):
    argdict = {"createNewOn": create_new_on}
    if DK is not None:
        argdict["DK"] = DK
    if DC is not None:
        argdict["DC"] = DC
    try:
        return _call(Orange.data.Table, filename, **argdict)
    except Exception as ex:
        if "is being loaded as" in str(ex):
            return Orange.data.Table(filename, **argdict)
        else:
            raise


def load_txt(filename, create_new_on=MakeStatus.NoRecognizedValues,
             DK=None, DC=None):
    argdict = {"createNewOn": create_new_on}
    if DK is not None:
        argdict["DK"] = DK
    if DC is not None:
        argdict["DC"] = DC
    return _call(Orange.data.Table, filename, **argdict)


def load_c45(filename, create_new_on=MakeStatus.NoRecognizedValues):
    argdict = {"createNewOn": create_new_on}
    return Orange.data.Table(filename, **argdict)


def load_basket(filename, create_new_on=MakeStatus.NoRecognizedValues):
    argdict = {"createNewOn": create_new_on}
    return Orange.data.Table(filename, **argdict)


def load_csv(filename, create_new_on=MakeStatus.NoRecognizedValues,
             dialect=None, header_format=PlainHeader, missing_values=None):

    if header_format == NoHeader:
        has_header, has_orange_header = False, False
    elif header_format == PlainHeader:
        has_header, has_orange_header = True, False
    elif header_format == OrangeHeader:
        has_header, has_orange_header = True, True
    else:
        has_header, has_orange_header = True, False

    DK = ",,?,NA,~*"
    if missing_values is not None:
        DK = ",".join([DK, missing_values])

    data = Orange.data.io.load_csv(
        filename,
        delimiter=dialect.delimiter,
        quotechar=dialect.quotechar,
        has_header=has_header,
        has_types=has_orange_header,
        has_annotations=has_orange_header,
        skipinitialspace=True,
        create_new_on=create_new_on,
        DK=DK
    )
    return data


#: Format description
#: :param str name: Human format name
#: :param list extensions: A list of extensions
#: :param function load:
#:     A (path: str, create_new_on=2: int, **extraparams) -> Table function.
Format = namedtuple("Format", ["name", "extensions", "load"])

FILEFORMATS = [
    Format("Tab-delimited files", [".tab"], load_tab),
    Format("Tab-delimited simplified",  [".txt"], load_txt),
    Format("Basket files", [".basket"], load_basket),
    Format("C4.5 files", [".names"], load_c45),
    Format("Comma-separated values", [".csv"], load_csv),
    Format("Tab-separated values", [".tsv"], load_csv),
]

known_ext = [".tab", ".txt", ".basket", ".data", ".names"]

extra = Orange.core.getRegisteredFileTypes()
extra = filter(lambda ft: len(ft) > 2, map(lambda ft: ft[:3], extra))
extra = [(name, extension.lstrip("*"), loader)
         for name, extension, loader in extra
         if loader is not None and extension.lstrip("*") not in known_ext]

FILEFORMATS += [Format(name, [ext], load) for name, ext, load in extra]


RegisteredFormats = {fmt.name: fmt for fmt in FILEFORMATS}

#: Load action
#: :param str format: format name
#: :param params: a namedtuple holding format specific parameters
LoadAction = namedtuple("LoadAction", ["format", "params"])

DEFAULT_ACTIONS = [
    LoadAction(fmt.name, option)
    for fmt, option in zip(FILEFORMATS, DEFAULT_OPTIONS)
]

DEFAULT_ACTIONS += [LoadAction(fmt.name, LoadOptions())
                    for fmt in FILEFORMATS[-len(extra):]]

LoadCSV = DEFAULT_ACTIONS[4]


def load_table(filename, options, create_new_on=MakeStatus.NoRecognizedValues):
    load_func = RegisteredFormats[options.format].load
    return load_func(filename, create_new_on=create_new_on,
                     **options.params._asdict())


class CSVOptionsWidget(QWidget):
    _PresetDelimiters = [
        ("Comma", ","),
        ("Tab", "\t"),
        ("Semicolon", ";"),
        ("Space", " "),
    ]

    format_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        self._delimiter_idx = 0
        self._delimiter_custom = "|"
        self._delimiter = ","
        self._quotechar = "'"
        self._escapechar = "\\"
        self._doublequote = True
        self._skipinitialspace = False

        super(QWidget, self).__init__(parent, **kwargs)

        layout = QVBoxLayout()
        # Dialect options

        form = QFormLayout()
        self.delimiter_cb = QComboBox()
        self.delimiter_cb.addItems(
            [name for name, _ in self._PresetDelimiters]
        )
        self.delimiter_cb.insertSeparator(self.delimiter_cb.count())
        self.delimiter_cb.addItem("Other")

        self.delimiter_cb.setCurrentIndex(self._delimiter_idx)
        self.delimiter_cb.activated.connect(self._on_delimiter_idx_changed)

        validator = QRegExpValidator(QRegExp("."))
        self.delimiteredit = QLineEdit(
            self._delimiter_custom,
            enabled=False
        )
        self.delimiteredit.setValidator(validator)
        self.delimiteredit.editingFinished.connect(self._on_delimiter_changed)

        delimlayout = QHBoxLayout()
        delimlayout.setContentsMargins(0, 0, 0, 0)
        delimlayout.addWidget(self.delimiter_cb)
        delimlayout.addWidget(self.delimiteredit)

        self.quoteedit = QLineEdit(self._quotechar)
        self.quoteedit.setValidator(validator)
        self.quoteedit.editingFinished.connect(self._on_quotechar_changed)

        self.escapeedit = QLineEdit(self._escapechar)
        self.escapeedit.setValidator(validator)
        self.escapeedit.editingFinished.connect(self._on_escapechar_changed)

        self.skipinitialspace_cb = QCheckBox(
            checked=self._skipinitialspace
        )

        form.addRow("Cell delimiter", delimlayout)
        form.addRow("Quote", self.quoteedit)
        form.addRow("Escape character", self.escapeedit)

        form.addRow(QFrame(self, frameShape=QFrame.HLine))
        # File format option
        self.missingedit = QLineEdit()
        self.missingedit.editingFinished.connect(self.format_changed)

        form.addRow("Missing values", self.missingedit)
        layout.addLayout(form)

        self.has_header_cb = QCheckBox("Has header")
        self.has_header_cb.toggled.connect(self.format_changed)
        self.has_type_defs_cb = QCheckBox("Has orange type definitions")
        self.has_type_defs_cb.toggled.connect(self.format_changed)

        layout.addWidget(self.has_header_cb)
        layout.addWidget(self.has_type_defs_cb)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def dialect(self):
        """
        Return the current state as a Dialect instance.
        """
        if self._delimiter_idx >= len(self._PresetDelimiters):
            delimiter = self._delimiter_custom
        else:
            _, delimiter = self._PresetDelimiters[self._delimiter_idx]

        quotechar = str(self.quoteedit.text()) or ""
        escapechar = str(self.escapeedit.text()) or None
        skipinitialspace = True
        return Dialect(delimiter, quotechar, escapechar,
                       doublequote=True, skipinitialspace=skipinitialspace)

    def set_dialect(self, dialect):
        """
        Set the current state to match dialect instance.
        """
        delimiter = dialect.delimiter
        try:
            index = [d for _, d in self._PresetDelimiters].index(delimiter)
        except ValueError:
            index = len(self._PresetDelimiters) + 1
        self._delimiter_idx = index
        self._delimiter_custom = delimiter
        self._quotechar = dialect.quotechar
        self._escapechar = dialect.escapechar
        self._skipinitialspace = dialect.skipinitialspace

        self.delimiter_cb.setCurrentIndex(index)
        self.delimiteredit.setText(delimiter)
        self.quoteedit.setText(dialect.quotechar or '"')
        self.escapeedit.setText(dialect.escapechar or "")
        self.skipinitialspace_cb.setChecked(dialect.skipinitialspace)

    def set_header_format(self, header_format):
        self._header_format = header_format
        if header_format == NoHeader:
            self.has_header_cb.setChecked(False)
        elif header_format == PlainHeader:
            self.has_header_cb.setChecked(True)
            self.has_type_defs_cb.setChecked(False)
        elif header_format == OrangeHeader:
            self.has_header_cb.setChecked(True)
            self.has_type_defs_cb.setChecked(True)

    def header_format(self):
        if self.has_header_cb.isChecked():
            if self.has_type_defs_cb.isChecked():
                return OrangeHeader
            else:
                return PlainHeader
        else:
            return NoHeader

    def set_missing_values(self, missing):
        self.missingedit.setText(missing)

    def missing_values(self):
        return str(self.missingedit.text())

    def _on_delimiter_idx_changed(self, index):
        if index < len(self._PresetDelimiters):
            self.delimiteredit.setText(self._PresetDelimiters[index][1])
        else:
            self.delimiteredit.setText(self._delimiter_custom)

        self.delimiteredit.setEnabled(index >= len(self._PresetDelimiters))
        self._delimiter_idx = index

        self.format_changed.emit()

    def _on_delimiter_changed(self):
        self._delimiter_custom = str(self.delimiteredit.text())
        self.format_changed.emit()

    def _on_quotechar_changed(self):
        self._quotechar = str(self.quoteedit.text())
        self.format_changed.emit()

    def _on_escapechar_changed(self):
        self._escapechar = str(self.escapeedit.text())
        self.format_changed.emit()

    def _on_skipspace_changed(self, skipinitialspace):
        self._skipinitialspace = skipinitialspace
        self.format_changed.emit()


class CSVImportDialog(QDialog):
    def __init__(self, parent=None, **kwargs):
        super(CSVImportDialog, self).__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self._options = None
        self._path = None
        self.__update_pending = False

        self._optionswidget = CSVOptionsWidget()
        self._optionswidget.format_changed.connect(self._invalidate_preview)

        self._stack = QStackedLayout()
        self._stack.setContentsMargins(0, 0, 0, 0)
        prev_box = QGroupBox("Preview")
        prev_box.setLayout(self._stack)
        self._preview = QTableView(tabKeyNavigation=False)
        self._preview_error = QLabel()
        self._stack.addWidget(self._preview)
        self._stack.addWidget(self._preview_error)

        buttons = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.layout().addWidget(self._optionswidget)
        self.layout().addWidget(prev_box)
        self.layout().addWidget(buttons)

    def set_options(self, options):
        self._options = options
        self._optionswidget.set_dialect(options.dialect)
        self._optionswidget.set_header_format(options.header_format)
        self._optionswidget.set_missing_values(options.missing_values or "")
        self._invalidate_preview()

    def options(self):
        missing_values = self._optionswidget.missing_values()
        return CSV(self._optionswidget.dialect(),
                   header_format=self._optionswidget.header_format(),
                   missing_values=missing_values)

    def set_path(self, path):
        if self._path != path:
            self._path = path
            self._invalidate_preview()

    def _invalidate_preview(self):
        if not self.__update_pending:
            self.__update_pending = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if self.__update_pending:
            self.__update_pending = False
            self._update_preview()

    def _update_preview(self):
        if not self._path:
            return

        head = itertools.islice(open(self._path, "rU"), 20)
        head = StringIO("".join(head))
        try:
            data = load_csv(head, **self.options()._asdict())
        except csv.Error as err:
            self._preview_error.setText(
                "Cannot load data preview:\n {!s}".format(err)
            )
            self._stack.setCurrentWidget(self._preview_error)
        except Exception as err:
            self._preview.setModel(None)
            raise
        else:
            model = ExampleTableModel(data, None, self)
            self._preview.setModel(model)
            self._stack.setCurrentWidget(self._preview)


def cb_append_file_list(combobox, paths):
    model = combobox.model()
    count = model.rowCount()
    cb_insert_file_list(combobox, count, paths)


def cb_insert_file_list(combobox, index, paths):
    model = combobox.model()
    iconprovider = QFileIconProvider()

    for i, path in enumerate(paths):
        basename = os.path.basename(path)
        item = QStandardItem(basename)
        item.setToolTip(path)
        item.setIcon(iconprovider.icon(QFileInfo(path)))
        model.insertRow(index + i, item)


class FileNameContextHandler(ContextHandler):
    def match(self, context, imperfect, filename):
        return 2 if context.filename == filename else 0


class standard_icons(object):
    def __init__(self, qwidget=None, style=None):
        self.qwidget = qwidget
        if qwidget is None:
            self.style = QApplication.instance().style()
        else:
            self.style = qwidget.style()

    @property
    def dir_open_icon(self):
        return self.style.standardIcon(QStyle.SP_DirOpenIcon)

    @property
    def reload_icon(self):
        return self.style.standardIcon(QStyle.SP_BrowserReload)


class OWFile(OWWidget):
    settingsList = ["selected_file", "recent_files", "show_advanced",
                    "create_new_on"]

    contextHandlers = {"": FileNameContextHandler()}

    def __init__(self, parent=None, signalManager=None,
                 title="CSV File Import"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False, noReport=True)

        self.symbol_DC = ""
        self.symbol_DK = ""

        #: List of recent opened files.
        self.recent_files = []
        #: Current selected file name
        self.selected_file = None
        #: Variable reuse flag
        self.create_new_on = 2
        #: Display advanced var reuse options
        self.show_advanced = False

        self.loadSettings()

        self.recent_files = filter(os.path.exists, self.recent_files)

        self._loader = None
        self._invalidated = False
        self._datareport = None

        layout = QHBoxLayout()
        OWGUI.widgetBox(self.controlArea, "File", orientation=layout)

        icons = standard_icons(self)

        self.recent_combo = QComboBox(
            self, objectName="recent_combo",
            toolTip="Recent files.",
            activated=self.activate_recent
        )
        cb_append_file_list(self.recent_combo, self.recent_files)

        self.recent_combo.insertSeparator(self.recent_combo.count())
        self.recent_combo.addItem(u"Browse documentation data sets…")

        self.browse_button = QPushButton(
            u"…",
            icon=icons.dir_open_icon, toolTip="Browse filesystem",
            clicked=self.browse
        )

        self.reload_button = QPushButton(
            "Reload", icon=icons.reload_icon,
            toolTip="Reload the selected file", clicked=self.reload,
            default=True
        )

        layout.addWidget(self.recent_combo, 2)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.reload_button)

        ###########
        # Info text
        ###########
        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace=True)
        self.infoa = OWGUI.widgetLabel(box, "No data loaded.")
        self.infob = OWGUI.widgetLabel(box, " ")
        self.warnings = OWGUI.widgetLabel(box, " ")

        # Set word wrap so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(QSizePolicy.Ignored,
                                    QSizePolicy.MinimumExpanding)

        advanced = QGroupBox(
            "Advanced Settings", checkable=True, checked=self.show_advanced
        )
        advanced.setLayout(QVBoxLayout())

        def set_group_visible(groupbox, state):
            layout = groupbox.layout()
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(state)
            groupbox.setFlat(not state)

        def toogle_advanced(state):
            self.show_advanced = state
            set_group_visible(advanced, state)
            self.layout().activate()
            QApplication.instance().processEvents()
            QTimer.singleShot(0, self.adjustSize)

        advanced.toggled.connect(toogle_advanced)

        self.taboptions = QWidget()
        self.taboptions.setLayout(QVBoxLayout())
        box = QGroupBox("Missing Value Symbols", flat=True)
        form = QFormLayout(fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(
            "Don't care:",
            OWGUI.lineEdit(None, self, "symbol_DC",
                           tooltip="Default values: '~' or '*'"))
        form.addRow(
            "Don't know:",
            OWGUI.lineEdit(None, self, "symbol_DK",
                           tooltip="Default values: empty fields (space), "
                                   "'?' or 'NA'"))
        box.setLayout(form)
        advanced.layout().addWidget(box)

        rb = OWGUI.radioButtonsInBox(
            advanced, self, "create_new_on",
            box="New Attributes",
            callback=self._invalidate,
            label=u"Create a new attribute when existing attribute(s) …",
            btnLabels=[u"Have mismatching order of values",
                       u"Have no common values with the new (recommended)",
                       u"Miss some values of the new attribute",
                       u"… Always create a new attribute"]
        )
        rb.setFlat(True)
        self.controlArea.layout().addWidget(advanced)

        button_box = QDialogButtonBox(orientation=Qt.Horizontal)
        self.import_options_button = QPushButton(
            u"Import Options…", enabled=False
        )
        self.import_options_button.pressed.connect(self._activate_import_dialog)
        button_box.addButton(
            self.import_options_button, QDialogButtonBox.ActionRole
        )
        button_box.addButton(
            QPushButton("&Report", pressed=self.reportAndFinish),
            QDialogButtonBox.ActionRole
        )
        self.controlArea.layout().addWidget(button_box)

        OWGUI.rubber(self.controlArea)

        set_group_visible(advanced, self.show_advanced)

        if self.recent_files and self.recent_files[0] == self.selected_file:
            QTimer.singleShot(
                0, lambda: self.activate_recent(0)
            )
        else:
            self.selected_file = None
            self.recent_combo.setCurrentIndex(-1)

    @Slot(int)
    def activate_recent(self, index):
        """Activate an item from the recent list."""
        if 0 <= index < len(self.recent_files):
            recent = self.recent_files[index]
            self.set_selected_file(recent)
        elif index == len(self.recent_files) + 1:
            status = self.browse(Orange.utils.environ.dataset_install_dir)
            if status == QDialog.Rejected:
                self.recent_combo.setCurrentIndex(
                    min(0, len(self.recent_files) - 1)
                )
        else:
            self.recent_combo.setCurrentIndex(-1)

    @Slot()
    def browse(self, startdir=None):
        """
        Open a file dialog and select a user specified file.
        """
        if startdir is None:
            if self.selected_file:
                startdir = os.path.dirname(self.selected_file)
            else:
                startdir = unicode(
                    QDesktopServices.storageLocation(
                        QDesktopServices.DocumentsLocation)
                )

        def format_spec(format):
            ext_pattern = ", ".join(map("*{}".format, format.extensions))
            return "{} ({})".format(format.name, ext_pattern)

        formats = [format_spec(format) for format in FILEFORMATS]
        filters = ";;".join(formats + ["All files (*.*)"])

        path, selected_filter = QFileDialog.getOpenFileNameAndFilter(
            self, "Open Data File", startdir, filters
        )
        if path:
            path = unicode(path)
            if selected_filter in formats:
                filter_idx = formats.index(str(selected_filter))
                fformat = FILEFORMATS[filter_idx]
                loader = DEFAULT_ACTIONS[filter_idx]
                if fformat.extensions in ([".csv"], [".tsv"]):
                    loader = loader._replace(
                        params=loader_for_path(path).params)
            else:
                loader = None
            self.set_selected_file(path, loader=loader)
            return QDialog.Accepted
        else:
            return QDialog.Rejected

    @Slot()
    def reload(self):
        """Reload the current selected file."""
        if self.selected_file:
            self.send_data()

    def _activate_import_dialog(self):
        assert self.selected_file is not None
        dlg = CSVImportDialog(
            self, windowTitle="Import Options",
        )
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowFlags(Qt.Sheet)
        loader = self._loader

        dlg.set_options(loader.params)
        dlg.set_path(self.selected_file)

        def update():
            self._loader = loader._replace(params=dlg.options())
            self._invalidate()

        dlg.accepted.connect(update)
        dlg.open()

    def set_selected_file(self, filename, loader=None):
        """Set the current filename path to be loaded."""
        self.closeContext("")

        self.selected_file = filename
        self._loader = None
        self._add_recent(filename)
        self.warning(1)

        self.openContext("", filename)

        if loader is not None and self._loader is not None and \
                self._loader.format == loader.format:
            loader = self._loader

        if loader is None:
            loader = self._loader

        if loader is None:
            loader = loader_for_path(filename)

        self.set_loader(loader)

    def _add_recent(self, filename):
        """Add filename to the list of recent files."""
        index_to_remove = None
        if filename in self.recent_files:
            index_to_remove = self.recent_files.index(filename)
        elif len(self.recent_files) >= 20:
            # keep maximum of 20 files in the list.
            index_to_remove = len(self.recent_files) - 1

        cb_insert_file_list(self.recent_combo, 0, [filename])
        self.recent_files.insert(0, filename)

        if index_to_remove is not None:
            self.recent_combo.removeItem(index_to_remove + 1)
            self.recent_files.pop(index_to_remove + 1)

        self.recent_combo.setCurrentIndex(0)

    def set_loader(self, loader):
        if loader is None:
            loader = DEFAULT_ACTIONS[0]

        self._loader = loader
        self.import_options_button.setEnabled(
            isinstance(loader.params, CSV)
        )

        self._invalidate()

    def _invalidate(self):
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if self._invalidated:
            self._invalidated = False
            self.send_data()

    def send_data(self):
        self.error(0)
        loader = self._loader
        if "DK" in loader.params._fields and self.symbol_DK:
            loader = loader._replace(
                params=loader.params._replace(
                    DK=str(self.symbol_DK),
                    DC=str(self.symbol_DC))
            )
        try:
            data = load_table(self.selected_file, loader,
                              create_new_on=3 - self.create_new_on)
        except Exception as err:
            self.error(str(err))
            self.infoa.setText('Data was not loaded due to an error.')
            self.infob.setText('Error:')
            self.warnings.setText(str(err))
            data = None
        else:
            self._update_status_messages(data)

        if data is not None:
            add_origin(data, self.selected_file)
            basename = os.path.basename(self.selected_file)
            data.name, _ = os.path.splitext(basename)
            self._datareport = self.prepareDataReport(data)
        else:
            self._datareport = None

        self.send("Data", data)

    def _update_status_messages(self, data):
        if data is None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            self.warnings.setText("")
            return

        def pluralize(seq):
            return "s" if len(seq) != 1 else ""

        summary = ("{n_instances} data instance{plural_1}, "
                   "{n_features} feature{plural_2}, "
                   "{n_meta} meta{plural_3}").format(
                        n_instances=len(data), plural_1=pluralize(data),
                        n_features=len(data.domain.attributes),
                        plural_2=pluralize(data.domain.attributes),
                        n_meta=len(data.domain.getmetas()),
                        plural_3=pluralize(data.domain.getmetas()))
        self.infoa.setText(summary)

        cl = data.domain.class_var
        if cl is not None:
            if isinstance(cl, Orange.feature.Continuous):
                self.infob.setText('Regression; Numerical class.')
            elif isinstance(cl, Orange.feature.Discrete):
                self.infob.setText(
                    'Classification; Discrete class with %d value%s.' %
                    (len(cl.values), 's' if len(cl.values) > 1 else '')
                )
            else:
                self.infob.setText("Class is neither discrete nor continuous.")
        elif data.domain.class_vars:
            self.infob.setText("Multiple class variables")
        else:
            self.infob.setText("Data has no dependent variable.")

        self.warnings.setText(
            feature_load_status_report(data, self.create_new_on))

    def sendReport(self):
        if self._datareport:
            formatname = self._loader.format
            self.reportSettings(
                "File", [("File name", self.selected_file),
                         ("Format", formatname)])
            self.reportData(self._datareport)

    def settingsFromWidgetCallback(self, handler, context):
        context.filename = self.selected_file or ""
        context.loader = self._loader
        context.symbolDC, context.symbolDK = self.symbol_DC, self.symbol_DK

    def settingsToWidgetCallback(self, handler, context):
        self.symbol_DC, self.symbol_DK = context.symbolDC, context.symbolDK
        self._loader = getattr(context, "loader", None)

    def setSettings(self, settings):
        if settings.get(_SETTINGS_VERSION_KEY, None) is None:
            # import old File widget's settings
            mapping = [
                ("recentFiles", "recent_files"),
                ("createNewOn", "create_new_on"),
                ("showAdvanced", "show_advanced")
            ]

            if all(old in settings for old, _ in mapping) and \
                    not any(new in settings for _, new in mapping):
                settings = settings.copy()
                for old, new in mapping:
                    settings[new] = settings[old]
                    del settings[old]

#                 settings[_SETTINGS_VERSION_KEY] = self.settingsDataVersion
                if len(settings["recent_files"]):
                    settings["selected_file"] = settings["recent_files"][0]

        super(OWFile, self).setSettings(settings)


def loader_for_path(path):
    _, ext = os.path.splitext(path)
    if ext in (".csv", ".tsv"):
        dialect, has_header = sniff_csv(path)
        header_format = PlainHeader if has_header else NoHeader
        options = CSV(dialect=dialect, header_format=header_format,
                      missing_values=None)
        return LoadCSV._replace(params=options)

    for fmt, option in zip(FILEFORMATS, DEFAULT_ACTIONS):
        if ext in fmt.extensions:
            return option
    else:
        return None


def sniff_csv(file):
    if isinstance(file, basestring):
        file = open(file, "rU")

    snifer = csv.Sniffer()
    sample = file.read(5 * 2 ** 20)  # max 5MB sample
    dialect = snifer.sniff(sample)
    dialect = Dialect(dialect.delimiter, dialect.quotechar,
                      dialect.escapechar, dialect.doublequote,
                      dialect.skipinitialspace, dialect.quoting)
    has_header = snifer.has_header(sample)
    return dialect, has_header


def feature_load_status_report(data, create_new_on):
    warnings = ""
    metas = data.domain.getmetas()
    attr_status = []
    meta_status = {}

    if hasattr(data, "attribute_load_status"):
        attr_status = data.attribute_load_status
    elif hasattr(data, "attributeLoadStatus"):
        attr_status = data.attributeLoadStatus

    if hasattr(data, "meta_attribute_load_status"):
        meta_status = data.meta_attribute_load_status
    elif hasattr(data, "metaAttributeLoadStatus"):
        meta_status = data.metaAttributeLoadStatus

    for status, message_used, message_not_used in STATUS_MESASGES:
        if create_new_on > status:
            message = message_used
        else:
            message = message_not_used
        if not message:
            continue

        attrs = [attr.name for attr, stat in zip(data.domain, attr_status)
                 if stat == status] + \
                [attr.name for id, attr in metas.items()
                 if meta_status.get(id, -99) == status]

        if attrs:
            jattrs = ", ".join(attrs)
            if len(jattrs) > 80:
                jattrs = jattrs[:80] + "..."
            if len(jattrs) > 30:
                warnings += "<li>%s:<br/> %s</li>" % (message, jattrs)
            else:
                warnings += "<li>%s: %s</li>" % (message, jattrs)
    return warnings


STATUS_MESASGES = [
    (MakeStatus.Incompatible,
     "",
     "The following attributes already existed but had a different order " +
     "of values, so new attributes needed to be created"),
    (MakeStatus.NoRecognizedValues,
     "The following attributes were reused although they share no " +
     "common values with the existing attribute of the same names",
     "The following attributes were not reused since they share no " +
     "common values with the existing attribute of the same names"),
    (MakeStatus.MissingValues,
     "The following attribute(s) were reused although some values " +
     "needed to be added",
     "The following attribute(s) were not reused since they miss some values")
]


def add_origin(table, filename):
    vars = table.domain.variables + table.domain.getmetas().values()
    strings = [var for var in vars if isinstance(var, Orange.feature.String)]
    dirname = os.path.dirname(filename)
    for var in strings:
        if "type" in var.attributes and "origin" not in var.attributes:
            var.attributes["origin"] = dirname


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = OWFile()
    w.show()
    w.raise_()
    app.exec_()
    w.saveSettings()
