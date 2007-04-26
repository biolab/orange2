from qt import *
from qttable import *
from qtcanvas import *
try:
    import qwt
except:
    import Qwt4 as qwt
import math
from OWWidget import *
import OWGUI

###########################################################################
class TestWidget(OWWidget):
    def __init__(self, parent=None, name='TestWidget'):
        OWWidget.__init__(self, parent, name, 'Microarray Heat Map', FALSE)

        #self.colorPalette = ColorPalette(self.space, self, "", additionalColors = ["Cell Outline", "Selected Cells"])
        self.colorPalette = ColorPalette(self.space, self, "", additionalColors = None)
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

import sys
if sys.platform == "darwin":
    def signedColor(long):
        if type(long) == int:
            return long
        elif long > 0xFFFFFFFF:
            long &= 0xFFFFFFFF
        
        if long & 0x80000000:
            return int(-((long ^ 0xFFFFFFFF) + 1))
        else:
            return int(long)

    def positiveColor(color):
        if color < 0:
            return (-color - 1) ^ 0xFFFFFFFF
        else:
            return color

    def signedPalette(palette):
        return [signedColor(color) for color in palette]

else:
    signedColor = positiveColor = signedPalette = lambda x:x
    
class ColorPalette(QWidget):
    def __init__(self, parent, master, value, label = "Colors", additionalColors = None, callback = None):
        QWidget.__init__(self, parent)
        self.constructing = TRUE

        self.callback = callback
        
        self.colorSchemas = {}

        self.setMinimumHeight(300)
        self.setMinimumWidth(200)

        #create box aroung whole widget
        self.box = OWGUI.widgetBox(self, label)
        self.box.setOrientation(Qt.Vertical)
        self.box.setColumns(20)
        self.box.setMinimumHeight(300)
        self.box.setMinimumWidth(200)
        self.box.setMaximumWidth(200)
        self.box.InsideSpacing = 2
        self.box.InsideMargin = 2

        self.schemaCombo = QComboBox(self.box)
        self.connect(self.schemaCombo, SIGNAL("activated(const QString&)"), self.onComboBoxChange)

        self.interpolationHBox = QHBox(self.box)
        self.interpolationHBox.setSpacing(5)
        
        self.colorButton1 = ColorButton(self, self.interpolationHBox)
        self.interpolationView = InterpolationView(self.interpolationHBox)
        self.colorButton2 = ColorButton(self, self.interpolationHBox)

        self.chkPassThroughBlack = QCheckBox("Pass through black", self.box)
        self.chkPassThroughBlack.setMaximumWidth(190)
        self.connect(self.chkPassThroughBlack, SIGNAL("stateChanged ( int )"), self.onCheckBoxChange)        
        self.box.addSpace(30)

        #special colors buttons
        self.NAHBox = QHBox(self.box)
        self.NAHBox.setSpacing(5)
        self.NAColorButton = ColorButton(self, self.NAHBox)
        self.NALabel = OWGUI.widgetLabel(self.NAHBox, "N/A", specialColorLabelWidth)

        self.underflowHBox = QHBox(self.box)
        self.underflowHBox.setSpacing(5)
        self.underflowColorButton = ColorButton(self, self.underflowHBox)
        self.underflowLabel = OWGUI.widgetLabel(self.underflowHBox, "Underflow", specialColorLabelWidth)

        self.overflowHBox = QHBox(self.box)
        self.overflowHBox.setSpacing(5)
        self.overflowColorButton = ColorButton(self, self.overflowHBox)
        self.overflowLabel = OWGUI.widgetLabel(self.overflowHBox, "Overflow", specialColorLabelWidth)
  
        self.backgroundHBox = QHBox(self.box)
        self.backgroundHBox.setSpacing(5)
        self.backgroundColorButton = ColorButton(self, self.backgroundHBox)
        self.backgroundLabel = OWGUI.widgetLabel(self.backgroundHBox, "Background (Grid)", specialColorLabelWidth)

        #set up additional colors
        self.additionalColorButtons = {}
        self.box.addSpace(0)

        if additionalColors<>None:
            for colorName in additionalColors:
                box = QHBox(self.box)
                box.setSpacing(5)
                button = ColorButton(self, box)
                label = OWGUI.widgetLabel(box, colorName, specialColorLabelWidth)
                self.additionalColorButtons[colorName] = button

        #set up new and delete buttons
        self.buttonHBox = QHBox(self.box)
        self.buttonHBox.setSpacing(5)        
        self.newButton = OWGUI.button(self.buttonHBox, self, "New", self.OnNewButtonClicked)
        self.deleteButton = OWGUI.button(self.buttonHBox, self, "Delete", self.OnDeleteButtonClicked)

        self.box.adjustSize()  
        self.adjustSize()

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
            s = QInputDialog.getText("New Schema", message, QLineEdit.Normal)

            ok = TRUE
            if (s[1]==TRUE):
                for i in range(self.schemaCombo.count()):
                    if s[0].lower().compare(self.schemaCombo.text(i).lower())==0:
                        ok = FALSE
                        message = "Color schema with that name already exists, please enter another name"
                if (ok):
                    self.colorSchemas[str(s[0])] = ColorSchema(self.getCurrentColorSchema().getName(),
                                                               self.getCurrentColorSchema().getPalette(),
                                                               self.getCurrentColorSchema().getAdditionalColors(),
                                                               self.getCurrentColorSchema().getPassThroughBlack())
                    self.schemaCombo.insertItem(s[0])
                    self.schemaCombo.setCurrentItem(self.schemaCombo.count()-1)
            self.deleteButton.setEnabled(self.schemaCombo.count()>1)

        
    def OnDeleteButtonClicked(self):
        i = self.schemaCombo.currentItem()
        self.schemaCombo.removeItem(i)
        self.schemaCombo.setCurrentItem(i)        
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
        for name in schemas:
            self.schemaCombo.insertItem(name)

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


        self.schemaCombo.insertItem("Blue - Yellow")
        palette = self.createPalette(QColor(0,0,255), QColor(255,255,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0., 255.), qRgb(255., 255., 0.), gray]
        self.colorSchemas["Blue - Yellow"] = ColorSchema("Blue - Yellow", palette, additionalColors, FALSE)
        
        self.schemaCombo.insertItem("Black - Red")
        palette = self.createPalette(QColor(0,0,0), QColor(255,0,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0, 0), qRgb(255., 0, 0), gray]
        self.colorSchemas["Black - Red"] = ColorSchema("Black - Red", palette, additionalColors, FALSE)
        
        self.schemaCombo.insertItem("Green - Black - Red")
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

class InterpolationView(QCanvasView):
    def __init__(self, parent = None):
        self.canvas = QCanvas(colorButtonSize,colorButtonSize)
        QCanvasView.__init__(self, self.canvas, parent)

        self.setFrameStyle(QFrame.NoFrame)
        self.setVScrollBarMode(QScrollView.AlwaysOff)
        self.setHScrollBarMode(QScrollView.AlwaysOff)        

        self.setMaximumHeight(colorButtonSize)
        self.setMinimumHeight(colorButtonSize)
        self.setMaximumWidth(135)

    def setPalette1(self, palette):
        dx = 140; dy = colorButtonSize
        bmp = chr(252)*dx*2 + reduce(lambda x,y:x+y, [chr(i*250/dx) for i in range(dx)] * (dy-4)) + chr(252)*dx*2 
        image = QImage(bmp, dx, dy, 8, signedPalette(palette), 256, QImage.LittleEndian) # palette should be 32 bit, what is not so on some platforms (Mac) so we force it
        pm = QPixmap()
        pm.convertFromImage(image, QPixmap.Color);

        # a little compatibility for QT 3.3 (on Mac at least)
        if hasattr(self.canvas, "setPaletteBackgroundPixmap"):
            self.canvas.setPaletteBackgroundPixmap(pm)
        else:
            self.canvas.setBackgroundPixmap(pm)
        self.canvas.update()

class ColorButton(QCanvasView):
    def __init__(self, master = None, parent = None):
        self.canvas = QCanvas(colorButtonSize,colorButtonSize)
        QCanvasView.__init__(self, self.canvas, parent)

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setVScrollBarMode(QScrollView.AlwaysOff)
        self.setHScrollBarMode(QScrollView.AlwaysOff)        

        self.setMaximumWidth(colorButtonSize)
        self.setMaximumHeight(colorButtonSize)
        self.setMinimumWidth(colorButtonSize)
        self.setMinimumHeight(colorButtonSize)        

        self.parent = parent
        self.color = Qt.white
        self.master = master
       
        self.viewport().setMouseTracking(True)

    def setColor(self, color):
        self.color = color
        self.canvas.setBackgroundColor(color)
        self.canvas.update()        
        self.update()

    def getColor(self):
        return self.color

    def contentsMousePressEvent(self, event):
        self.clicked = TRUE
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.update()
    
    def contentsMouseReleaseEvent(self, event):
        color = QColorDialog.getColor(self.color, self.parent)        
        if color.isValid():
            self.color = color
            self.canvas.setBackgroundColor(color)            
            self.canvas.update()
            if self.master<>None:
                self.master.colorSchemaChange()
            
        self.clicked = FALSE
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.update()



if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = TestWidget()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
