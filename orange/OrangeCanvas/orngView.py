# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    handling the mouse events inside documents
#
import orngCanvasItems
from PyQt4.QtCore import *
from PyQt4.QtGui import *

# ########################################
# ######## SCHEMA VIEW class
# ########################################
class SchemaView(QGraphicsView):
    def __init__(self, doc, *args):
        apply(QGraphicsView.__init__,(self,) + args)
        #self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.doc = doc
        self.bMouseDown = False
        self.bWidgetDragging = False
        self.bLineDragging = False
        self.bMultipleSelection = False
        self.movingWidget = None
        self.mouseDownPosition = QPoint(0,0)
        self.lastMousePosition = QPoint(0,0)
        self.tempLine = None
        self.tempRect = None
        self.selectedLine = None
        self.tempWidget = None
        self.selWidgets = []
        self.setRenderHint(QPainter.Antialiasing)
        self.createPopupMenus()
        self.ensureVisible(0,0,1,1)

    def createPopupMenus(self):
#        self.widgetPopup = QMenu("Widget", self)
#        self.widgetPopup.addAction( "Open",  self.openActiveWidget)
#        self.widgetPopup.addSeparator()
#        rename = self.widgetPopup.addAction( "&Rename", self.renameActiveWidget, Qt.Key_F2)
#        delete = self.widgetPopup.addAction("Remove", self.removeActiveWidget, Qt.Key_Delete)
#        delete.setShortcut(Qt.Key_Delete)
#        rename.setShortcut(Qt.Key_F2)
#        #self.widgetPopup.insertSeparator()
#        #self.menupopupWidgetEnabledID = self.widgetPopup.addAction("Enabled", self.enabledWidget)

        self.linePopup = QMenu("Link", self)
        self.lineEnabledAction = self.menupopupLinkEnabledID = self.linePopup.addAction( "Enabled",  self.toggleEnabledLink)
        self.lineEnabledAction.setCheckable(1)
        self.linePopup.addSeparator()
        self.linePopup.addAction("Reset Signals", self.resetLineSignals)
        self.linePopup.addAction("Remove", self.deleteSelectedLine)
        self.linePopup.addSeparator()


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
        if not self.tempWidget:
            return
        exName = str(self.tempWidget.caption)
        (newName ,ok) = QInputDialog.getText(self, "Rename Widget", "Enter new name for the \"" + exName + "\" widget:", QLineEdit.Normal, exName)
        newName = str(newName)
        if ok and self.tempWidget != None and newName != exName:
            for widget in self.doc.widgets:
                if widget.caption.lower() == newName.lower():
                    QMessageBox.information(self, 'Orange Canvas', 'Unable to rename widget. An instance with that name already exists.')
                    return
            self.tempWidget.updateText(newName)
            self.tempWidget.updateTooltip()
            self.tempWidget.updateLinePosition()
            if len(newName) < 3 or newName[:2].lower() != "qt":
                newName = "Qt " + newName
            self.tempWidget.instance.setCaption(newName)
            self.doc.enableSave(True)

    # popMenuAction - user selected to delete active widget
    def removeActiveWidget(self):
        if self.doc.signalManager.signalProcessingInProgress:
             QMessageBox.information( self, "Orange Canvas", "Unable to remove widgets while signal processing is in progress. Please wait.")
             return
        if self.selWidgets == []:
            self.selWidgets = [self.tempWidget]

        for item in self.selWidgets:
            self.doc.removeWidget(item)

        self.scene().update()
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
##            self.selectedLine.repaintLine(self)
            self.selectedLine.inWidget.updateTooltip()
            self.selectedLine.outWidget.updateTooltip()
        self.doc.enableSave(True)

    # popMenuAction - delete selected link
    def deleteSelectedLine(self):
        if not self.selectedLine: return
        if self.doc.signalManager.signalProcessingInProgress:
             QMessageBox.information( self, "Orange Canvas", "Unable to remove connection while signal processing is in progress. Please wait.")
             return
        self.deleteLine(self.selectedLine)
        self.selectedLine = None
        self.scene().update()

    def deleteLine(self, line):
        if line != None:
            self.doc.removeLine1(line)

    # resend signals between two widgets. receiving widget will process the received data
    def resendSignals(self):
        if self.selectedLine != None:
            self.doc.signalManager.setLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance, 1, justSend = 1)

    def resetLineSignals(self):
        if self.selectedLine:
            self.doc.resetActiveSignals(self.selectedLine.outWidget, self.selectedLine.inWidget, enabled = self.doc.signalManager.getLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance))
            self.selectedLine.inWidget.updateTooltip()
            self.selectedLine.outWidget.updateTooltip()
            self.selectedLine.updateTooltip()

    def unselectAllWidgets(self):
        for item in self.selWidgets: item.setSelected(0)
        self.selWidgets = []

    # ###########################################
    # ###########################################

    # return number of items in "items" of type "type"
    def findItemTypeCount(self, items, Type):
        return sum([type(item) == Type for item in items])

    # find and return first item of type Type
    def findFirstItemType(self, items, Type):
        for item in items:
            if type(item) == Type:
                return item
        return None

    # find and return all items of type "type"
    def findAllItemType(self, items, Type):
        ret = []
        for item in items:
            if type(item) == Type:
                ret.append(item)
        return ret


    # ###########################################
    # MOUSE events
    # ###########################################

    # mouse button was pressed
    def mousePressEvent(self, ev):
        self.mouseDownPosition = self.mapToScene(ev.pos())

        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        for item in self.doc.widgets:
            if item not in self.selWidgets: item.setSelected(0)

        activeItem = self.scene().itemAt(self.mouseDownPosition)
        if not activeItem:
            self.tempWidget = None
            self.tempRect = None
            self.bMultipleSelection = True
            self.unselectAllWidgets()

        # we clicked on a widget or on a line
        else:
            if type(activeItem) == orngCanvasItems.CanvasWidget:        # if we clicked on a widget
                self.tempWidget = activeItem
                self.tempWidget.setSelected(1)

                # did we click inside the boxes to draw connections
                if ev.button() == Qt.LeftButton:
                    if activeItem.mouseInsideLeftChannel(self.mouseDownPosition) or activeItem.mouseInsideRightChannel(self.mouseDownPosition):
                        if not self.doc.signalManager.signalProcessingInProgress:   # if we are processing some signals, don't allow to add lines
                            self.unselectAllWidgets()
                            self.bLineDragging = True
                            pos = activeItem.getEdgePoint(self.mouseDownPosition)
                            self.tempLine = orngCanvasItems.TempCanvasLine(self.doc.canvasDlg, self.scene())
                            self.tempLine.setLine(pos.x(), pos.y(), pos.x(), pos.y())
                            self.tempLine.setPen(QPen(self.doc.canvasDlg.lineColor, 1))
                            #self.scene().update()

                    else:   # we clicked inside the widget and we start dragging it
                        self.bWidgetDragging = True
                        if self.doc.ctrlPressed:
                            if activeItem not in self.selWidgets:
                                self.selWidgets.append(activeItem)
                            else:
                                self.selWidgets.remove(activeItem)
                                activeItem.setSelected(0)
                        elif activeItem not in self.selWidgets:
                            self.unselectAllWidgets()
                            self.selWidgets = [activeItem]
                            self.bMultipleSelection = False

                        for w in self.selWidgets:
                            w.setCoords(w.x(), w.y())
                            w.savePosition()
                            w.setAllLinesFinished(False)

                # is we clicked the right mouse button we show the popup menu for widgets
                elif ev.button() == Qt.RightButton:
                    #self.widgetPopup.popup(ev.globalPos())
                    self.doc.canvasDlg.widgetPopup.popup(ev.globalPos())
                else:
                    self.unselectAllWidgets()

            # if we right clicked on a line we show a popup menu
            elif type(activeItem) == orngCanvasItems.CanvasLine and ev.button() == Qt.RightButton:
                self.bMultipleSelection = False
                self.unselectAllWidgets()
                self.selectedLine = activeItem
                self.lineEnabledAction.setChecked(self.selectedLine.getEnabled())
                self.linePopup.popup(ev.globalPos())
            else:
                self.unselectAllWidgets()

        self.scene().update()
        self.bMouseDown = True
        self.lastMousePosition = self.mapToScene(ev.pos())


    # ###################################################################
    # mouse button was pressed and mouse is moving ######################
    def mouseMoveEvent(self, ev):
        #if not self.bLineDragging and (ev.x() < 0 or ev.x() > self.contentsX() + self.visibleWidth() or ev.y() < 0 or ev.y() > self.contentsY() + self.visibleHeight()):
        #    self.contentsMouseReleaseEvent(ev)
        #    return
        point = self.mapToScene(ev.pos())
        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        #print self.bWidgetDragging, self.selWidgets
        if self.bWidgetDragging:
            for item in self.selWidgets:
                ex_pos = QPoint(item.x(), item.y())
                item.setCoordsBy(point.x() - self.lastMousePosition.x(), point.y() - self.lastMousePosition.y())
                if self.doc.canvasDlg.snapToGrid:
                    item.moveToGrid()
                else:
                    item.setCoords(item.xPos, item.yPos)

                items = self.scene().collidingItems(item)
                item.invalidPosition = (self.findItemTypeCount(items, orngCanvasItems.CanvasWidget) > 0)
                item.updateLineCoords()

        elif self.bLineDragging:
            if self.tempLine:
                self.tempLine.setLine(self.tempLine.line().x1(), self.tempLine.line().y1(), point.x(), point.y())

        elif self.bMultipleSelection:
            rect = QRectF(min (self.mouseDownPosition.x(), point.x()), min (self.mouseDownPosition.y(), point.y()), abs(self.mouseDownPosition.x() - point.x()), abs(self.mouseDownPosition.y() - point.y()))
            self.tempRect = QGraphicsRectItem(rect, None, self.scene())
            self.tempRect.show()

            # select widgets in rectangle
            items = self.scene().collidingItems(self.tempRect)
            widgets = self.findAllItemType(items, orngCanvasItems.CanvasWidget)

            for widget in self.doc.widgets:
                if widget in widgets and widget not in self.selWidgets:
                    widget.setSelected(1); widget.savePosition()
                elif widget not in widgets and widget in self.selWidgets:
                    widget.setSelected(0)

            self.selWidgets = widgets

        self.scene().update()
        self.lastMousePosition = point


    # ###################################################################
    # mouse button was released #########################################
    def mouseReleaseEvent(self, ev):
        point = self.mapToScene(ev.pos())
        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        # if we are moving a widget
        if self.bWidgetDragging:
            validPos = True
            for item in self.selWidgets:
                items = self.scene().collidingItems(item)
                validPos = validPos and (self.findItemTypeCount(items, orngCanvasItems.CanvasWidget) == 0)


            for item in self.selWidgets:
                item.invalidPosition = False
                if not validPos:
                    #item.setCoordsBy(self.mouseDownPosition.x() - ev.pos().x(), self.mouseDownPosition.y() - ev.pos().y())
                    item.restorePosition()
                item.updateTooltip()
                item.updateLineCoords()
                item.setAllLinesFinished(True)
##                item.repaintWidget()
##                item.repaintAllLines()

            self.doc.enableSave(True)

        # if we are drawing line
        elif self.bLineDragging:
            item = self.scene().itemAt(QPointF(ev.pos()))

            # we must check if we have really conected some output to input
            if type(item) == orngCanvasItems.CanvasWidget and self.tempWidget and self.tempLine and item != self.tempWidget:
                if self.tempWidget.mouseInsideLeftChannel(self.tempLine.line().p1()):
                    outWidget = item
                    inWidget  = self.tempWidget
                else:
                    outWidget = self.tempWidget
                    inWidget  = item

                # hide the temp line
                self.tempLine.hide()
                self.tempLine = None

                if self.doc.signalManager.signalProcessingInProgress:
                     QMessageBox.information( self, "Orange Canvas", "Unable to connect widgets while signal processing is in progress. Please wait.")
                else:
                    line = self.doc.addLine(outWidget, inWidget)
##                    if line: line.repaintLine(self)

            if self.tempLine != None:
                self.tempLine.setLine(0,0,0,0)
                self.tempLine.hide()
                self.tempLine = None

        self.scene().update()
        self.bMouseDown = False
        self.bWidgetDragging = False
        self.bLineDragging = False
        self.bMultipleSelection = False

    def mouseDoubleClickEvent(self, ev):
        point = self.mapToScene(ev.pos())
        activeItem = self.scene().itemAt(point)
        if type(activeItem) == orngCanvasItems.CanvasWidget:        # if we clicked on a widget
            self.tempWidget = activeItem
            self.openActiveWidget()
        elif type(activeItem) == orngCanvasItems.CanvasLine:
            if self.doc.signalManager.signalProcessingInProgress:
                QMessageBox.information( self, "Orange Canvas", "Please wait until Orange finishes processing signals.")
                return
            self.doc.resetActiveSignals(activeItem.outWidget, activeItem.inWidget, enabled = self.doc.signalManager.getLinkEnabled(activeItem.outWidget.instance, activeItem.inWidget.instance))
            activeItem.inWidget.updateTooltip()
            activeItem.outWidget.updateTooltip()
            activeItem.updateTooltip()


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
                self.repaint()
                widget.update()
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
#        activeItems = self.scene().collisions(rect)
#        for item in activeItems:
#            item.drawShape(painter)