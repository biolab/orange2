import os
import sys
import warnings
import Orange
from OWWidget import *
import OWGUI

NAME = "File"
DESCRIPTION = "Reads data from an input file."
ICON = "icons/File.svg"
MAINTAINER = "Janez Demsar"
MAINTAINER_EMAIL = "janez.demsar(@at@)fri.uni-lj.si"
PRIORITY = 10
CATEGORY = "Data"
KEYWORDS = ["data", "file", "load", "read"]

OUTPUTS = (
    {"name": "Data",
     "type": Orange.data.Table,
     "doc": "Attribute-valued data set read from the input file.",
     },
)

WIDGET_CLASS = "OWFile"


def call(f, *args, **kwargs):
    return f(*args, **kwargs)

# Make any KernelWarning raise an error if called through the 'call' function
# defined above.
warnings.filterwarnings(
    "error", ".*", Orange.core.KernelWarning,
    __name__, call.func_code.co_firstlineno + 1
)


class FileNameContextHandler(ContextHandler):
    def match(self, context, imperfect, filename):
        return context.filename == filename and 2


def addOrigin(examples, filename):
    vars = examples.domain.variables + examples.domain.getmetas().values()
    strings = [var for var in vars if isinstance(var, Orange.feature.String)]
    dirname, basename = os.path.split(filename)
    for var in strings:
        if "type" in var.attributes and "origin" not in var.attributes:
            var.attributes["origin"] = dirname


class OWFile(OWWidget):
    settingsList = ["recentFiles", "createNewOn", "showAdvanced"]
    contextHandlers = {"": FileNameContextHandler()}

    registeredFileTypes = [ft[:2] for ft in Orange.core.getRegisteredFileTypes()
                           if len(ft) > 2 and ft[2]]
    dlgFormats = (
        'Tab-delimited files (*.tab *.txt)\n'
        'C4.5 files (*.data)\n'
        'Assistant files (*.dat)\n'
        'Retis files (*.rda *.rdo)\n'
        'Basket files (*.basket)\n' +
        "\n".join("%s (%s)" % (ft[:2]) for ft in registeredFileTypes) +
        "\nAll files(*.*)"
    )

    formats = {
        ".tab": "Tab-delimited file",
        ".txt": "Tab-delimited file",
        ".data": "C4.5 file",
        ".dat": "Assistant file",
        ".rda": "Retis file",
        ".rdo": "Retis file",
        ".basket": "Basket file"
    }
    formats.update(dict((ext.lstrip("*."), name)
                        for name, ext in registeredFileTypes))

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "File", wantMainArea=0)

        self.inputs = []
        self.outputs = [("Data", ExampleTable)]

        self.recentFiles = []
        self.symbolDC = "?"
        self.symbolDK = "~"
        self.createNewOn = 1
        self.domain = None
        self.loadedFile = ""
        self.showAdvanced = 0
        self.loadSettings()

        self.dataReport = None

        box = OWGUI.widgetBox(self.controlArea, "Data File", addSpace=True,
                              orientation="horizontal")
        self.filecombo = QComboBox(box)
        self.filecombo.setMinimumWidth(150)
        self.filecombo.activated[int].connect(self.selectFile)

        box.layout().addWidget(self.filecombo)
        button = OWGUI.button(box, self, '...', callback=self.browse)
        button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.reloadBtn = OWGUI.button(
            box, self, "Reload", callback=self.reload, default=True)

        self.reloadBtn.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self.reloadBtn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace=True)
        self.infoa = OWGUI.widgetLabel(box, 'No data loaded.')
        self.infob = OWGUI.widgetLabel(box, ' ')
        self.warnings = OWGUI.widgetLabel(box, ' ')

        #Set word wrap so long warnings won't expand the widget
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(QSizePolicy.Ignored,
                                    QSizePolicy.MinimumExpanding)

        smallWidget = OWGUI.collapsableWidgetBox(
            self.controlArea, "Advanced settings", self, "showAdvanced",
            callback=self.adjustSize0)

        box = QGroupBox("Missing Value Symbols")
        form = QFormLayout(fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)

        form.addRow(
            "Don't care:",
            OWGUI.lineEdit(None, self, "symbolDC",
                           tooltip="Default values: '~' or '*'")
        )
        form.addRow(
            "Don't know:",
            OWGUI.lineEdit(None, self, "symbolDK",
                           tooltip="Default values: empty fields (space), "
                                   "'?' or 'NA'")
        )
        box.setLayout(form)
        smallWidget.layout().addWidget(box)
        smallWidget.layout().addSpacing(8)

        OWGUI.radioButtonsInBox(
            smallWidget, self, "createNewOn", box="New Attributes",
            label="Create a new attribute when existing attribute(s) ...",
            btnLabels=["Have mismatching order of values",
                       "Have no common values with the new (recommended)",
                       "Miss some values of the new attribute",
                       "... Always create a new attribute"]
        )

        OWGUI.rubber(smallWidget)
        smallWidget.updateControls()

        OWGUI.rubber(self.controlArea)

        # remove missing data set names
        def exists(path):
            if not os.path.exists(path):
                dirpath, basename = os.path.split(path)
                return os.path.exists(os.path.join("./", basename))
            else:
                return True

        self.recentFiles = filter(exists, self.recentFiles)
        self.setFileList()

        if len(self.recentFiles) > 0 and exists(self.recentFiles[0]):
            self.openFile(self.recentFiles[0])

    def adjustSize0(self):
        qApp.processEvents()
        QTimer.singleShot(50, self.adjustSize)

    def setFileList(self):
        self.filecombo.clear()
        model = self.filecombo.model()
        iconprovider = QFileIconProvider()
        if not self.recentFiles:
            item = QStandardItem("(none)")
            item.setEnabled(False)
            item.setSelectable(False)
            model.appendRow([item])
        else:
            for fname in self.recentFiles:
                item = QStandardItem(os.path.basename(fname))
                item.setToolTip(fname)
                item.setIcon(iconprovider.icon(QFileInfo(fname)))
                model.appendRow(item)

        self.filecombo.insertSeparator(self.filecombo.count())
        self.filecombo.addItem("Browse documentation data sets...")
        item = model.item(self.filecombo.count() - 1)
        item.setEnabled(os.path.isdir(Orange.utils.environ.dataset_install_dir))

    def reload(self):
        if self.recentFiles:
            return self.openFile(self.recentFiles[0])

    def settingsFromWidgetCallback(self, handler, context):
        context.filename = self.loadedFile
        context.symbolDC, context.symbolDK = self.symbolDC, self.symbolDK

    def settingsToWidgetCallback(self, handler, context):
        self.symbolDC, self.symbolDK = context.symbolDC, context.symbolDK

    def selectFile(self, n):
        if n < len(self.recentFiles):
            name = self.recentFiles[n]
            self.recentFiles.remove(name)
            self.recentFiles.insert(0, name)
        elif n >= 0:
            self.browseDocDatasets()

        if len(self.recentFiles) > 0:
            self.setFileList()
            self.openFile(self.recentFiles[0])

    def browseDocDatasets(self):
        """
        Display a FileDialog with the documentation datasets folder.
        """
        self.browse(Orange.utils.environ.dataset_install_dir)

    def browse(self, startpath=None):
        """
        Display a FileDialog and select a file to open.
        """
        if startpath is None:
            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
                startpath = os.path.expanduser("~/")
            else:
                startpath = self.recentFiles[0]

        filename = QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', startpath, self.dlgFormats)

        filename = unicode(filename)

        if filename == "":
            return

        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)

        self.setFileList()

        self.openFile(filename)

    # Open a file, create data from it and send it over the data channel
    def openFile(self, fn):
        self.error()
        self.warning()
        self.information()

        if not os.path.exists(fn):
            basename = os.path.basename(fn)
            if os.path.exists(os.path.join("./", basename)):
                fn = os.path.join("./", basename)
                self.information("Loading '%s' from the current directory." %
                                 basename)

        self.closeContext()
        self.loadedFile = ""

        if fn == "(none)":
            self.send("Data", None)
            self.infoa.setText("No data loaded")
            self.infob.setText("")
            self.warnings.setText("")
            return

        self.openContext("", fn)

        self.loadedFile = ""

        argdict = {"createNewOn": 3 - self.createNewOn}
        if self.symbolDK:
            argdict["DK"] = str(self.symbolDK)
        if self.symbolDC:
            argdict["DC"] = str(self.symbolDC)

        data = None
        try:
            data = call(Orange.data.Table, fn, **argdict)
            self.loadedFile = fn
        except Exception as ex:
            if "is being loaded as" in str(ex):
                try:
                    data = Orange.data.Table(fn, **argdict)
                    self.warning(0, str(ex))
                except:
                    pass

            if data is None:
                self.error(str(ex))
                self.infoa.setText('Data was not loaded due to an error.')
                self.infob.setText('Error:')
                self.warnings.setText(str(ex))
                return

        self.infoa.setText("%d data instance%s, " %
                           (len(data), 's' if len(data) > 1 else '') +
                           "%d feature%s, " %
                           (len(data.domain.attributes),
                            's' if len(data.domain.attributes) > 1 else '') +
                           '%d meta attribute%s.' %
                           (len(data.domain.getmetas()),
                            's' if len(data.domain.getmetas()) > 1 else ''))
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
        else:
            self.infob.setText("Data has no dependent variable.")

        self.warnings.setText(
            feature_load_status_report(data, self.createNewOn))

        addOrigin(data, fn)
        # make new data and send it
        name = os.path.basename(fn)
        name, _ = os.path.splitext(name)
        data.name = name

        self.dataReport = self.prepareDataReport(data)

        self.send("Data", data)

    def sendReport(self):
        if self.dataReport:
            _, ext = os.path.splitext(self.loadedFile)
            format = self.formats.get(ext, "unknown format")
            self.reportSettings(
                "File", [("File name", self.loadedFile),
                         ("Format", format)])
            self.reportData(self.dataReport)


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
    (Orange.feature.Descriptor.MakeStatus.Incompatible,
     "",
     "The following attributes already existed but had a different order " +
     "of values, so new attributes needed to be created"),
    (Orange.feature.Descriptor.MakeStatus.NoRecognizedValues,
     "The following attributes were reused although they share no " +
     "common values with the existing attribute of the same names",
     "The following attributes were not reused since they share no " +
     "common values with the existing attribute of the same names"),
    (Orange.feature.Descriptor.MakeStatus.MissingValues,
     "The following attribute(s) were reused although some values " +
     "needed to be added",
     "The following attribute(s) were not reused since they miss some values")
]


if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWFile()
    ow.show()
    a.exec_()
    ow.saveSettings()
