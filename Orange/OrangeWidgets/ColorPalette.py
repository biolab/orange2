from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math
from OWWidget import *
import OWGUI

###########################################################################
class TestWidget(OWWidget):
    def __init__(self, parent=None, name='TestWidget'):
        OWWidget.__init__(self, parent, name, 'Microarray Heat Map', FALSE)
        #self.controlArea = self
        #self.setLayout(QVBoxLayout())

        self.colorPalette = ColorPalette(self, self.controlArea, "", additionalColors = None)
        self.controlArea.layout().addWidget(self.colorPalette)
        #a = ColorButton(self)
        #self.controlArea.layout().addWidget(a)

###########################################################################

colorButtonSize = 15
specialColorLabelWidth = 160
paletteInterpolationColors = 250

# On Mac OS X there are problems with QRgb and whether it is long or int and even whether
# it is positive or negative number (there is corelation between those)
# Color can be stored in 32 bit unsigned int but Python does not have unsigned int explicitly
# So Python on Mac sometimes uses long where it should use int (when the highest bit is set and
# it sees the number as positive - so it cannot be stored as positive number in 31 bits) and sometimes
# it needs unsigned number and so uses long and does not want a signed int

try:
    qRed(-1)
    wantsPositiveColor = False
except:
    wantsPositiveColor = True

def signedColor(long):
    if isinstance(long, int):
        return long
    
    long &= 0xFFFFFFFF
    
    if long & 0x80000000:
        return int(-((long ^ 0xFFFFFFFF) + 1))
    else:
        return int(long)

def positiveColor(color):
    if wantsPositiveColor and color < 0:
        return (-color - 1) ^ 0xFFFFFFFF
    else:
        return color

def signedPalette(palette):
    return [signedColor(color) for color in palette]

class ColorPalette(QWidget):
    def __init__(self, parent, master, value, label = "Colors", additionalColors = None, callback = None):
        QWidget.__init__(self, parent)

        self.constructing = TRUE
        self.callback = callback
        self.schema = ""
        self.passThroughBlack = 0

        self.colorSchemas = {}

        self.setMinimumHeight(300)
        self.setMinimumWidth(200)

        self.box = OWGUI.widgetBox(self, label, orientation = "vertical")

        self.schemaCombo = OWGUI.comboBox(self.box, self, "schema", callback = self.onComboBoxChange)

        self.interpolationHBox = OWGUI.widgetBox(self.box, orientation = "horizontal")
        self.colorButton1 = ColorButton(self, self.interpolationHBox)
        self.interpolationView = InterpolationView(self.interpolationHBox)
        self.colorButton2 = ColorButton(self, self.interpolationHBox)

        self.chkPassThroughBlack = OWGUI.checkBox(self.box, self, "passThroughBlack", "Pass through black", callback = self.onCheckBoxChange)
        #OWGUI.separator(self.box, 10, 10)
        self.box.layout().addSpacing(10)

        #special colors buttons

        self.NAColorButton = ColorButton(self, self.box, "N/A")
        self.underflowColorButton = ColorButton(self, self.box, "Underflow")
        self.overflowColorButton = ColorButton(self, self.box, "Overflow")
        self.backgroundColorButton = ColorButton(self, self.box, "Background (Grid)")

        #set up additional colors
        self.additionalColorButtons = {}

        if additionalColors<>None:
            for colorName in additionalColors:
                self.additionalColorButtons[colorName] = ColorButton(self, self.box, colorName)

        #set up new and delete buttons
        self.buttonHBox = OWGUI.widgetBox(self.box, orientation = "horizontal")
        self.newButton = OWGUI.button(self.buttonHBox, self, "New", self.OnNewButtonClicked)
        self.deleteButton = OWGUI.button(self.buttonHBox, self, "Delete", self.OnDeleteButtonClicked)

        self.setInitialColorPalettes()
        self.paletteSelected()
        self.constructing = FALSE

    def onComboBoxChange(self, string):
        self.paletteSelected()

    def onCheckBoxChange(self, state):
        self.colorSchemaChange()

    def OnNewButtonClicked(self):
        message = "Please enter new color schema name"
        ok = FALSE
        while (not ok):
            s = QInputDialog.getText(self, "New Schema", message)
            ok = TRUE
            if (s[1]==TRUE):
                for i in range(self.schemaCombo.count()):
                    if s[0].lower().compare(self.schemaCombo.itemText(i).lower())==0:
                        ok = FALSE
                        message = "Color schema with that name already exists, please enter another name"
                if (ok):
                    self.colorSchemas[str(s[0])] = ColorSchema(self.getCurrentColorSchema().getName(),
                                                               self.getCurrentColorSchema().getPalette(),
                                                               self.getCurrentColorSchema().getAdditionalColors(),
                                                               self.getCurrentColorSchema().getPassThroughBlack())
                    self.schemaCombo.addItem(s[0])
                    self.schemaCombo.setCurrentIndex(self.schemaCombo.count()-1)
            self.deleteButton.setEnabled(self.schemaCombo.count()>1)


    def OnDeleteButtonClicked(self):
        i = self.schemaCombo.currentIndex()
        self.schemaCombo.removeItem(i)
        self.schemaCombo.setCurrentIndex(i)
        self.deleteButton.setEnabled(self.schemaCombo.count()>1)
        self.paletteSelected()

    def getCurrentColorSchema(self):
        return self.colorSchemas[str(self.schemaCombo.currentText())]

    def setCurrentColorSchema(self, schema):
        self.colorSchemas[str(self.schemaCombo.currentText())] = schema


    def getColorSchemas(self):
        return self.colorSchemas

    def setColorSchemas(self, schemas):
        self.colorSchemas = schemas
        self.schemaCombo.clear()
        self.schemaCombo.addItems(schemas)
        self.paletteSelected()

    def createPalette(self,color1,color2, passThroughBlack):
        palette = []
        if passThroughBlack:
            for i in range(paletteInterpolationColors/2):
                palette += [qRgb(color1.red() - color1.red()*i*2./paletteInterpolationColors,
                                 color1.green() - color1.green()*i*2./paletteInterpolationColors,
                                 color1.blue() - color1.blue()*i*2./paletteInterpolationColors)]

            for i in range(paletteInterpolationColors - (paletteInterpolationColors/2)):
                palette += [qRgb(color2.red()*i*2./paletteInterpolationColors,
                                 color2.green()*i*2./paletteInterpolationColors,
                                 color2.blue()*i*2./paletteInterpolationColors)]
        else:
            for i in range(paletteInterpolationColors):
                palette += [qRgb(color1.red() + (color2.red()-color1.red())*i/paletteInterpolationColors,
                                 color1.green() + (color2.green()-color1.green())*i/paletteInterpolationColors,
                                 color1.blue() + (color2.blue()-color1.blue())*i/paletteInterpolationColors)]
        return palette

    def paletteSelected(self):
        schema = self.getCurrentColorSchema()
        self.interpolationView.setPalette1(schema.getPalette())
        self.colorButton1.setColor(self.rgbToQColor(schema.getPalette()[0]))
        self.colorButton2.setColor(self.rgbToQColor(schema.getPalette()[249]))

        self.chkPassThroughBlack.setChecked(schema.getPassThroughBlack())

        self.NAColorButton.setColor(self.rgbToQColor(schema.getPalette()[255]))
        self.overflowColorButton.setColor(self.rgbToQColor(schema.getPalette()[254]))
        self.underflowColorButton.setColor(self.rgbToQColor(schema.getPalette()[253]))
        self.backgroundColorButton.setColor(self.rgbToQColor(schema.getPalette()[252]))

        for buttonName in self.additionalColorButtons:
            self.additionalColorButtons[buttonName].setColor(self.rgbToQColor(schema.getAdditionalColors()[buttonName]))

        if not self.constructing:
            self.callback()

    def rgbToQColor(self, rgb):
        # we could also use QColor(positiveColor(rgb), 0xFFFFFFFF) but there is probably a reason
        # why this was not used before so I am leaving it as it is

        return QColor(qRed(positiveColor(rgb)), qGreen(positiveColor(rgb)), qBlue(positiveColor(rgb))) # on Mac color cannot be negative number in this case so we convert it manually

    def qRgbFromQColor(self, qcolor):
        return qRgb(qcolor.red(), qcolor.green(), qcolor.blue())

    def colorSchemaChange(self):
        white = qRgb(255,255,255)
        gray = qRgb(200,200,200)
        name = self.getCurrentColorSchema().getName()
        passThroughBlack = self.chkPassThroughBlack.isChecked()
        palette = self.createPalette(self.colorButton1.getColor(), self.colorButton2.getColor(), passThroughBlack)
        palette += [white]*2 + [self.qRgbFromQColor(self.backgroundColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.underflowColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.overflowColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.NAColorButton.getColor())]

        self.interpolationView.setPalette1(palette)

        additionalColors = {}
        for buttonName in self.additionalColorButtons:
            additionalColors[buttonName] = self.qRgbFromQColor(self.additionalColorButtons[buttonName].getColor())

        schema = ColorSchema(name, palette, additionalColors, passThroughBlack)
        self.setCurrentColorSchema(schema)

        if not self.constructing and self.callback:
            self.callback()


    def setInitialColorPalettes(self):
        white = qRgb(255,255,255)
        gray = qRgb(200,200,200)

        additionalColors = {}
        for buttonName in self.additionalColorButtons:
            additionalColors[buttonName] = gray


        self.schemaCombo.addItem("Blue - Yellow")
        palette = self.createPalette(QColor(0,0,255), QColor(255,255,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0., 255.), qRgb(255., 255., 0.), gray]
        self.colorSchemas["Blue - Yellow"] = ColorSchema("Blue - Yellow", palette, additionalColors, FALSE)

        self.schemaCombo.addItem("Black - Red")
        palette = self.createPalette(QColor(0,0,0), QColor(255,0,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0, 0), qRgb(255., 0, 0), gray]
        self.colorSchemas["Black - Red"] = ColorSchema("Black - Red", palette, additionalColors, FALSE)

        self.schemaCombo.addItem("Green - Black - Red")
        palette = self.createPalette(QColor(0,255,0), QColor(255,0,0),TRUE)
        palette += [white]*3 + [qRgb(0, 255., 0), qRgb(255., 0, 0), gray]
        self.colorSchemas["Green - Black - Red"] = ColorSchema("Green - Black - Red", palette, additionalColors, TRUE)


class ColorSchema:
    def __init__(self, name, palette, additionalColors, passThroughBlack):
        self.name = name
        self.palette = palette
        self.additionalColors = additionalColors
        self.passThroughBlack = passThroughBlack

    def getName(self):
        return self.name

    def getPalette(self):
        return self.palette

    def getAdditionalColors(self):
        return self.additionalColors

    def getPassThroughBlack(self):
        return self.passThroughBlack

class InterpolationView(QGraphicsView):
    def __init__(self, parent = None):
        self.canvas = QGraphicsScene(0,0,colorButtonSize,colorButtonSize)
        QGraphicsView.__init__(self, self.canvas, parent)

        self.setFrameStyle(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setFixedHeight(colorButtonSize)
        self.setMinimumWidth(colorButtonSize)
        self.setMaximumWidth(135)
        if parent and parent.layout():
            parent.layout().addWidget(self)

    def setPalette1(self, palette):
        dx = 140; dy = colorButtonSize
        bmp = chr(252)*dx*2 + reduce(lambda x,y:x+y, [chr(i*250/dx) for i in range(dx)] * (dy-4)) + chr(252)*dx*2
        image = QImage(bmp, dx, dy, QImage.Format_MonoLSB)
        #image.setNumColors(256)
        image.setColorTable(palette)
        pm = QPixmap()
        pm.fromImage(image)

        self.canvas.addPixmap(pm)
        self.canvas.update()

class ColorButton(QWidget):
    def __init__(self, master = None, parent = None, label = None, color = None):
        QWidget.__init__(self, master)

        self.parent = parent
        self.master = master

        if self.parent and self.parent.layout():
            self.parent.layout().addWidget(self)

        self.setLayout(QHBoxLayout())
        self.layout().setMargin(0)
        self.icon = QFrame(self)
        self.icon.setFixedSize(colorButtonSize, colorButtonSize)
        self.icon.setAutoFillBackground(1)
        self.icon.setFrameStyle (QFrame.StyledPanel+ QFrame.Sunken)
        self.layout().addWidget(self.icon)

        if label != None:
            self.label = OWGUI.widgetLabel(self, label)
            self.layout().addWidget(self.label)

        if color != None:
            self.setColor(color)


    def setColor(self, color):
        self.color = color
        palette = QPalette()
        palette.setBrush(QPalette.Background, color)
        self.icon.setPalette(palette)

    def getColor(self):
        return self.color

    def mousePressEvent(self, ev):
        color = QColorDialog.getColor(self.color, self.parent)
        if color.isValid():
            self.setColor(color)
            if self.master and hasattr(self.master, "colorSchemaChange"):
                self.master.colorSchemaChange()


if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = TestWidget()
    #a.setMainWidget(ow)

    ow.show()
    sys.exit(a.exec_())
    ow.saveSettings()

