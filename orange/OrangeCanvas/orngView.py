# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    handling the mouse events inside documents
#
from qt import *
from qtcanvas import *
import orngCanvasItems
TRUE  = 1
FALSE = 0


# ########################################
# ######## SCHEMA VIEW class
# ########################################
class SchemaView(QCanvasView):
    def __init__(self, doc, *args):
        apply(QCanvasView.__init__,(self,) + args)
        self.doc = doc
        self.bMouseDown = FALSE
        self.bWidgetDragging = FALSE
        self.bLineDragging = FALSE
        self.bMultipleSelection = FALSE
        self.movingWidget = None
        self.moving_start = QPoint(0,0)
        self.moving_ex_pos = QPoint(0,0)
        self.tempLine = None
        self.selectedLine = None
        self.tempWidget = None
        self.selWidgets = []
        self.createPopupMenus()
        self.connect(self, SIGNAL("contentsMoving(int,int)"), self.contentsMoving)


    def createPopupMenus(self):
        self.widgetPopup = QPopupMenu(self, "Widget")
        self.widgetPopup.insertItem( "Open",  self.openActiveWidget)
        self.widgetPopup.insertSeparator()
        rename = self.widgetPopup.insertItem( "&Rename", self.renameActiveWidget, Qt.Key_F2)
        delete = self.widgetPopup.insertItem("Remove", self.removeActiveWidget, Qt.Key_Delete )
        self.widgetPopup.setAccel(Qt.Key_Delete, delete)
        self.widgetPopup.setAccel(Qt.Key_F2, rename)
        #self.widgetPopup.insertSeparator()
        #self.menupopupWidgetEnabledID = self.widgetPopup.insertItem("Enabled", self.enabledWidget)

        self.linePopup = QPopupMenu(self, "Link")
        self.menupopupLinkEnabledID = self.linePopup.insertItem( "Enabled",  self.toggleEnabledLink)
        self.linePopup.insertSeparator()
        self.linePopup.insertItem( "Remove",  self.deleteSelectedLine)
        self.linePopup.insertSeparator() 

   
    # ###########################################
    # POPUP MENU - WIDGET actions
    # ###########################################

    # popMenuAction - user selected to show active widget
    def openActiveWidget(self):
        if self.tempWidget.instance != None:
            self.tempWidget.instance.reshow()
            if self.tempWidget.instance.isMinimized():  # if widget is minimized, show its normal size
                self.tempWidget.instance.showNormal()

    # popMenuAction - user selected to rename active widget            
    def renameActiveWidget(self):
        exName = str(self.tempWidget.caption)
        (newName ,ok) = QInputDialog.getText("Rename", "Enter new name for the widget \"" + exName + "\":", exName)
        newName = str(newName)
        if ok and self.tempWidget != None and newName != exName:
            for widget in self.doc.widgets:
                if widget.caption.lower() == newName.lower():
                    QMessageBox.critical(self,'Orange Canvas','Unable to rename widget. An instance with that name already exists.',  QMessageBox.Ok + QMessageBox.Default)
                    return
            self.tempWidget.updateText(newName)
            self.tempWidget.updateTooltip()
            self.tempWidget.updateLinePosition()
            if len(newName) < 3 or newName[:2].lower() != "qt":
                newName = "Qt " + newName
            self.tempWidget.instance.setCaption(newName)
            self.doc.enableSave(TRUE)

    # popMenuAction - user selected to delete active widget
    def removeActiveWidget(self):
        if self.doc.signalManager.signalProcessingInProgress:
             QMessageBox.information( None, "Orange Canvas", "Unable to remove widgets while signal processing is in progress. Please wait.", QMessageBox.Ok + QMessageBox.Default )
             return
        if not self.bMultipleSelection:
            self.selWidgets = [self.tempWidget]

        for item in self.selWidgets:
            self.doc.removeWidget(item)

        self.canvas().update()
        self.selWidgets = []
        self.tempWidget = None

    # ###########################################
    # POPUP MENU - LINKS actions
    # ###########################################

    # popMenuAction - enable/disable link between two widgets
    def toggleEnabledLink(self):
        if self.selectedLine != None:
            oldEnabled = self.doc.signalManager.getLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance)
            self.doc.signalManager.setLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance, not oldEnabled)
            self.selectedLine.setEnabled(not oldEnabled)
            self.selectedLine.repaintLine(self)
            self.selectedLine.inWidget.updateTooltip()
            self.selectedLine.outWidget.updateTooltip()
        self.doc.enableSave(TRUE)

    # popMenuAction - delete selected link
    def deleteSelectedLine(self):
        if self.doc.signalManager.signalProcessingInProgress:
             QMessageBox.information( None, "Orange Canvas", "Unable to remove connection while signal processing is in progress. Please wait.", QMessageBox.Ok + QMessageBox.Default )
             return
        self.deleteLine(self.selectedLine)
        self.selectedLine = None
        self.canvas().update()
    
    def deleteLine(self, line):
        if line != None:
            self.doc.removeLine1(line)
      

    # ###########################################
    # ###########################################
    
    # return number of items in "items" of type "type"
    def findItemTypeCount(self, items, type):
        count = 0
        for item in items:
            try:
                type.rtti(item)
                count = count+1
            except TypeError:
                pass
        return count

    # find and return first item of type "type"
    def findFirstItemType(self, items, type):
        for item in items:
            try:
                type.rtti(item)
                return item
            except TypeError:
                pass
        return None

    # find and return all items of type "type"
    def findAllItemType(self, items, type):
        ret = []
        for item in items:
            try:
                type.rtti(item)
                ret.append(item)
            except TypeError:
                pass
        return ret
        

    # ###########################################
    # MOUSE events
    # ###########################################

    # mouse button was pressed
    def contentsMousePressEvent(self, ev):
        for item in self.doc.widgets:
            if item not in self.selWidgets:
                item.selected = FALSE
            item.hide()
            item.show()

        rect = QRect(ev.pos().x()-5, ev.pos().y()-5,10,10)        
        activeItems = self.canvas().collisions(rect)
        if activeItems == []:
            self.tempWidget = None
            self.tempRect = None
            self.bMultipleSelection = TRUE
            for item in self.selWidgets:
                item.selected = FALSE
            self.selWidgets = []
            self.moving_start = QPoint(ev.pos().x(), ev.pos().y())
            self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())

        # we clicked on a widget or on a line            
        elif activeItems != []:
            widget = self.findFirstItemType(activeItems, orngCanvasItems.CanvasWidget)
            line   = self.findFirstItemType(activeItems, orngCanvasItems.CanvasLine)
            # if we clicked on a widget
            if widget != None:
                self.tempWidget = widget
                self.tempWidget.selected = TRUE

                # did we click inside the boxes to draw connections
                if ev.button() == QMouseEvent.LeftButton and (widget.mouseInsideLeftChannel(rect) or widget.mouseInsideRightChannel(rect)):
                    if not self.doc.signalManager.signalProcessingInProgress:  # if we are processing some signals, don't allow to add lines
                        self.bLineDragging = TRUE
                        pos = widget.getEdgePoint(rect)
                        self.tempLine = orngCanvasItems.TempCanvasLine(self.doc.canvas)
                        self.tempLine.setPoints(pos.x(), pos.y(), pos.x(), pos.y())
                        self.tempLine.show()
                        self.canvas().update()
                    
                # we clicked inside the widget and we start dragging it
                elif ev.button() == QMouseEvent.LeftButton:
                    self.bWidgetDragging = TRUE
                    self.moving_start = QPoint(ev.pos().x(), ev.pos().y())
                    self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())
                    self.moving_mouseOffset = QPoint(ev.pos().x() - widget.x(), ev.pos().y() - widget.y())

                    if widget not in self.selWidgets and self.doc.canvasDlg.ctrlPressed == 0:
                        for item in self.selWidgets:
                            item.selected = FALSE
                        self.selWidgets = [widget]
                        self.bMultipleSelection = FALSE
                    elif self.doc.canvasDlg.ctrlPressed == 1:
                        if widget not in self.selWidgets:
                            self.selWidgets.append(widget)
                        else:
                            self.selWidgets.remove(widget)
                            widget.selected = FALSE
                        self.doc.canvas.update()

                    for widget in self.selWidgets:
                        widget.setCoords(widget.x(), widget.y())
                        widget.savePosition()
                        widget.removeTooltip()
                        widget.setAllLinesFinished(FALSE)
                        widget.repaintAllLines()
                    
                # is we clicked the right mouse button we show the popup menu for widgets
                elif ev.button() == QMouseEvent.RightButton:
                    self.widgetPopup.popup(ev.globalPos())

            # if we right clicked on a line we show a popup menu
            elif line != None and ev.button() == QMouseEvent.RightButton:
                self.bMultipleSelection = FALSE
                for item in self.selWidgets:
                    item.selected = FALSE
                self.selWidgets = []
                self.selectedLine = line
                self.linePopup.setItemChecked(self.menupopupLinkEnabledID, self.selectedLine.getEnabled()) 
                self.linePopup.popup(ev.globalPos())

        self.doc.canvas.update()
        self.bMouseDown = TRUE


    # ###################################################################
    # mouse button was pressed and mouse is moving ######################
    def contentsMouseMoveEvent(self, ev):
        #if not self.bLineDragging and (ev.x() < 0 or ev.x() > self.contentsX() + self.visibleWidth() or ev.y() < 0 or ev.y() > self.contentsY() + self.visibleHeight()):
        #    self.contentsMouseReleaseEvent(ev)
        #    return

        if self.bWidgetDragging:
            for item in self.selWidgets:
                ex_pos = QPoint(item.x(), item.y())
                item.setCoordsBy(ev.pos().x() - self.moving_ex_pos.x(), ev.pos().y() - self.moving_ex_pos.y())
                if self.doc.canvasDlg.snapToGrid:
                    item.moveToGrid()
                else:
                    item.setCoords(item.xPos, item.yPos)                    

                items = self.canvas().collisions(item.rect())
                count = self.findItemTypeCount(items, orngCanvasItems.CanvasWidget)
                if count > 1: item.invalidPosition = TRUE
                else:         item.invalidPosition = FALSE
                item.updateLineCoords()
            self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())
            
        elif self.bLineDragging:
            if self.tempLine: self.tempLine.setPoints(self.tempLine.startPoint().x(), self.tempLine.startPoint().y(), ev.pos().x(), ev.pos().y())

        elif self.bMultipleSelection:
            rect = QRect(min (self.moving_start.x(), ev.pos().x()), min (self.moving_start.y(), ev.pos().y()), abs(self.moving_start.x() - ev.pos().x()), abs(self.moving_start.y() - ev.pos().y()))
            if self.tempRect: self.tempRect.hide()

            self.tempRect = QCanvasRectangle(rect, self.doc.canvas)
            self.tempRect.show()
            self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())

            # select widgets in rectangle
            for item in self.selWidgets: item.selected = FALSE
            self.selWidgets = []
            items = self.canvas().collisions(rect)
            widgets = self.findAllItemType(items, orngCanvasItems.CanvasWidget)
            for widget in widgets:
                widget.selected = TRUE
                widget.savePosition()
                widget.repaintWidget()
                self.selWidgets.append(widget)
            for widget in self.doc.widgets:
                if widget not in widgets:
                    widget.selected = FALSE
                    widget.repaintWidget()

        self.canvas().update()

    # ###################################################################
    # mouse button was released #########################################
    def contentsMouseReleaseEvent(self, ev):
        # if we are moving a widget
        if self.bWidgetDragging:
            validPos = TRUE
            for item in self.selWidgets:
                items = self.canvas().collisions(item.rect())
                count = self.findItemTypeCount(items, orngCanvasItems.CanvasWidget)
                if count > 1:
                    validPos = FALSE

            for item in self.selWidgets:
                item.invalidPosition = FALSE
                if not validPos:
                    #item.setCoordsBy(self.moving_start.x() - ev.pos().x(), self.moving_start.y() - ev.pos().y())
                    item.restorePosition()
                item.updateTooltip()
                item.updateLineCoords()
                item.setAllLinesFinished(TRUE)
                item.repaintWidget()
                item.repaintAllLines()
                
            self.doc.enableSave(TRUE)

        # if we are drawing line
        elif self.bLineDragging:
            items = self.canvas().collisions(ev.pos())
            item = self.findFirstItemType(items, orngCanvasItems.CanvasWidget)

            # we must check if we have really conected some output to input
            if self.tempWidget and self.tempLine and item and item != self.tempWidget:
                if self.tempWidget.mouseInsideLeftChannel(self.tempLine.startPoint()):
                    outWidget = item
                    inWidget  = self.tempWidget
                else:
                    outWidget = self.tempWidget
                    inWidget  = item

                # hide the temp line
                self.tempLine.hide()
                self.tempLine.setCanvas(None)
                self.tempLine = None

                if self.doc.signalManager.signalProcessingInProgress:
                     QMessageBox.information( None, "Orange Canvas", "Unable to connect widgets while signal processing is in progress. Please wait.", QMessageBox.Ok + QMessageBox.Default )
                else:
                    line = self.doc.addLine(outWidget, inWidget)
                    if line: line.repaintLine(self)
                
            if self.tempLine != None:
                self.tempLine.setPoints(0,0,0,0)
                self.tempLine.hide() 
                self.tempLine = None

        elif self.bMultipleSelection:
            if self.tempRect: self.tempRect.hide()
                
        self.canvas().update()
        self.bMouseDown = FALSE
        self.bWidgetDragging = FALSE
        self.bLineDragging = FALSE

    def contentsMouseDoubleClickEvent(self, ev):
        rect = QRect(ev.pos().x()-3, ev.pos().y()-3,6,6)        
        activeItems = self.canvas().collisions(rect)    
        widget = self.findFirstItemType(activeItems, orngCanvasItems.CanvasWidget)
        line   = self.findFirstItemType(activeItems, orngCanvasItems.CanvasLine)
        if widget:
            self.tempWidget = widget
            self.openActiveWidget()
        elif line:
            if self.doc.signalManager.signalProcessingInProgress:
                QMessageBox.information( None, "Orange Canvas", "Unable to modify signals while signal processing is in progress. Please wait.", QMessageBox.Ok + QMessageBox.Default )
                return
            self.doc.resetActiveSignals(line.outWidget, line.inWidget, enabled = self.doc.signalManager.getLinkEnabled(line.outWidget.instance, line.inWidget.instance))


    # if we scroll the view, we have to update tooltips for widgets
    def contentsMoving(self, x,y):
        for widget in self.doc.widgets:
            #QToolTip.remove(self, QRect(0,0, self.canvas().width(), self.canvas().height()))
            widget.removeTooltip()
            widget.setViewPos(x,y)
            widget.updateTooltip()

    def progressBarHandler(self, widgetInstance, value):
        qApp.processEvents()        # allow processing of other events
        for widget in self.doc.widgets:
            if widget.instance == widgetInstance:
                if value < 0: widget.showProgressBar()
                elif value > 100: widget.hideProgressBar()
                else: widget.setProgressBarValue(value)
                return

    def processingHandler(self, widgetInstance, value):
        for widget in self.doc.widgets:
            if widget.instance == widgetInstance:
                widget.setProcessing(value)
                return
            

#    def drawContents(self, painter):
#        for widget in self.doc.widgets:
#            widget.drawShape(painter)
#
#        for line in self.doc.lines:
#            line.drawShape(painter)
#
#    def drawContents(self, painter, x, y, w, h):
#        rect = QRect(x,y,w,h)        
#        activeItems = self.canvas().collisions(rect)
#        for item in activeItems:
#            item.drawShape(painter)