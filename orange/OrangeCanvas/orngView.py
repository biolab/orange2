# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	handling the mouse events inside documents
#
from qt import *
from qtcanvas import *
import orngTabs
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
        self.linePopup.insertItem( "Remove",  self.deleteLink)
        self.linePopup.insertSeparator() 

   
    # ###########################################
    # POPUP MENU - WIDGET actions
    # ###########################################

    # popMenuAction - user selected to show active widget
    def openActiveWidget(self):
        if self.tempWidget.instance != None:
            self.tempWidget.instance.hide()
            self.tempWidget.instance.show()
        pass

    # popMenuAction - user selected to rename active widget            
    def renameActiveWidget(self):
        (string,ok) = QInputDialog.getText("Rename", "Enter new name for the widget \"" + str(self.tempWidget.caption) + "\":")
        if ok and self.tempWidget != None:
            for widget in self.doc.widgets:
                if widget.caption.lower() == str(string).lower():
                    QMessageBox.critical(self,'Qrange Canvas','Unable to rename widget. An instance with that name already exists.',  QMessageBox.Ok + QMessageBox.Default)
                    return
            self.tempWidget.updateText(string)
            self.tempWidget.updateTooltip()
            self.doc.hasChanged = TRUE

    # popMenuAction - user selected to delete active widget
    def removeActiveWidget(self):
        if not self.bMultipleSelection:
            self.selWidgets = [self.tempWidget]

        for item in self.selWidgets:
            self.doc.widgets.remove(item)
            for line in item.inLines:
                self.selectedLine = line
                self.deleteLink()
            for line in item.outLines:
                self.selectedLine = line
                self.deleteLink()
            item.hideWidget()
            list = self.canvas().allItems()
            list.remove(item)

        self.selWidgets = []
        self.tempWidget = None
        self.canvas().update()
        self.doc.hasChanged = TRUE

    # ###########################################
    # POPUP MENU - LINKS actions
    # ###########################################

    # popMenuAction - enable/disable link between two widgets
    def toggleEnabledLink(self):
        if self.selectedLine != None:
            self.selectedLine.setEnabled(not self.selectedLine.getEnabled())
            self.selectedLine.repaintLine(self)
            self.selectedLine.inWidget.updateTooltip()
            self.selectedLine.outWidget.updateTooltip()
        self.doc.hasChanged = TRUE

    # popMenuAction - delete selected link
    def deleteLink(self):
        if self.selectedLine != None:
            for widget in self.doc.widgets:
                widget.removeLine(self.selectedLine)
            self.removeLine(self.selectedLine)
            self.selectedLine.repaintLine(self)
            self.selectedLine = None
            self.doc.hasChanged = TRUE
            self.selectedLine = None

    # hide and remove the line "line"
    def removeLine(self, line):
        self.doc.lines.remove(line)
        line.hide()
        line.setEnabled(FALSE)
        line.inWidget.updateTooltip()
        line.outWidget.updateTooltip()
        line = None

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

        rect = QRect(ev.pos().x()-1, ev.pos().y()-1,3,3)        
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
                if ev.button() == QMouseEvent.LeftButton and (widget.mouseInsideLeftChannel(ev.pos()) or widget.mouseInsideRightChannel(ev.pos())):
                    self.tempLineStartPos = QPoint(ev.pos().x(), ev.pos().y())
                    self.bLineDragging = TRUE
                    self.tempLine = None
                    self.moving_start = widget.getEdgePoint(ev.pos())
                    self.moving_ex_pos= self.moving_start

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
        #print "drag move event\n"
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
                if count > 1:
                    item.invalidPosition = TRUE
                else:
                    item.invalidPosition = FALSE
                item.updateLineCoords()
            self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())
            self.doc.canvas.update()

        elif self.bLineDragging:
            if self.tempLine == None:
                self.tempLine = orngCanvasItems.CanvasLine(self.doc.canvas)
            #self.repaintContents(self.moving_start.x(), self.moving_start.y(), ev.pos().x(), ev.pos().y())
            self.tempLine.setPoints(self.moving_start.x(), self.moving_start.y(), ev.pos().x(), ev.pos().y())
            self.tempLine.show()
            self.canvas().update()

        elif self.bMultipleSelection:
            rect = QRect(min (self.moving_start.x(), ev.pos().x()), min (self.moving_start.y(), ev.pos().y()), abs(self.moving_start.x() - ev.pos().x()), abs(self.moving_start.y() - ev.pos().y()))
            if self.tempRect != None:
                self.tempRect.hide()

            self.tempRect = QCanvasRectangle(rect, self.doc.canvas)
            self.tempRect.show()
            self.canvas().update()
            self.moving_ex_pos = QPoint(ev.pos().x(), ev.pos().y())

            # select widgets in rectangle
            for item in self.selWidgets:
                item.selected = FALSE
            self.selWidgets = []
            items = self.canvas().collisions(rect)
            widgets = self.findAllItemType(items, orngCanvasItems.CanvasWidget)
            for widget in widgets:
                widget.selected = TRUE
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
                    item.setCoordsBy(self.moving_start.x() - ev.pos().x(), self.moving_start.y() - ev.pos().y())
                item.updateTooltip()
                item.updateLineCoords()
                item.setAllLinesFinished(TRUE)
                item.repaintWidget()
                item.repaintAllLines()
                
            self.doc.hasChanged = TRUE

        # if we are drawing line
        elif self.bLineDragging:
            items = self.canvas().collisions(ev.pos())
            item = self.findFirstItemType(items, orngCanvasItems.CanvasWidget)

            # we must check if we have really coonected some output to input
            if item!= None and item != self.tempWidget:
                if self.tempWidget.mouseInsideLeftChannel(self.tempLineStartPos):
                    outWidget = item
                    inWidget  = self.tempWidget
                else:
                    outWidget = self.tempWidget
                    inWidget  = item
                    
                #search for common channels...
                count = 0
                for outChannel in outWidget.widget.outList:
                    if inWidget.widget.inList.count(outChannel) > 0:
                        count = count+1

                if count == 0:
                    self.tempLine.hide()
                    QMessageBox.information( None, "Orange Canvas", "Selected widgets don't share a common signal type. Unable to connect.", QMessageBox.Ok + QMessageBox.Default )
                else:
                    line = self.doc.addLine(outWidget, inWidget)
                    if line == None: self.repaintContents(QRect(min(self.tempLineStartPos.x(), ev.pos().x())-5, min(self.tempLineStartPos.y(), ev.pos().y())-5, abs(self.tempLineStartPos.x() - ev.pos().x())+10, abs(self.tempLineStartPos.y() - ev.pos().y())+10))
                    else:            
                        line.show()

                        # we add the line to the input and output list of connected widgets                    
                        if self.tempWidget.mouseInsideLeftChannel(self.moving_start):
                        	self.tempWidget.addInLine(line)	
                        	item.addOutLine(line)
                        else:
                        	self.tempWidget.addOutLine(line)	
                        	item.addInLine(line)
                        inWidget.updateTooltip()
                        outWidget.updateTooltip()
                        line.updateLinePos()
                        line.repaintLine(self)
                
            if self.tempLine != None:
                self.tempLine.setPoints(0,0,0,0)
                self.tempLine.hide() 
                self.tempLine = None

        elif self.bMultipleSelection:
            if self.tempRect != None:
                self.tempRect.hide()
                
        self.canvas().update()
        self.bMouseDown = FALSE
        self.bWidgetDragging = FALSE
        self.bLineDragging = FALSE

    def contentsMouseDoubleClickEvent(self, ev):
        rect = QRect(ev.pos().x()-3, ev.pos().y()-3,6,6)        
        activeItems = self.canvas().collisions(rect)    
        widget = self.findFirstItemType(activeItems, orngCanvasItems.CanvasWidget)
        line   = self.findFirstItemType(activeItems, orngCanvasItems.CanvasLine)
        if widget != None:
            self.tempWidget = widget
            self.openActiveWidget()
        elif line != None:
            ok = line.resetActiveSignals(self.doc.canvasDlg)
            if not ok:
                self.selectedLine = line
                self.deleteLink()
            else:
                line.setEnabled(line.getEnabled())


    # if we scroll the view, we have to update tooltips for widgets
    def contentsMoving(self, x,y):
        for widget in self.doc.widgets:
            #QToolTip.remove(self, QRect(0,0, self.canvas().width(), self.canvas().height()))
            widget.removeTooltip()
            widget.setViewPos(x,y)
            widget.updateTooltip()

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