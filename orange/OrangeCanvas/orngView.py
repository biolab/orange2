# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    handling the mouse events inside documents
#
import orngCanvasItems
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import orngHistory, orngTabs

        
class SchemaView(QGraphicsView):
    def __init__(self, doc, *args):
        apply(QGraphicsView.__init__,(self,) + args)
        #self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.doc = doc
        self.bWidgetDragging = False               # are we currently dragging a widget
        self.bLineDragging = False                 # are we currently drawing a line
        self.bMultipleSelection = False            # are we currently in the process of drawing a rectangle containing widgets that we wish to select
        self.movingWidget = None
        self.mouseDownPosition = QPoint(0,0)
        self.tempLine = None
        self.tempRect = None
        self.selectedLine = None
        self.tempWidget = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.ensureVisible(0,0,1,1)

        # create popup menus
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
        #if not self.tempWidget or self.tempWidget.instance == None: return
        widgets = self.getSelectedWidgets()
        if len(widgets) != 1: return
        widget = widgets[0]
        widget.instance.reshow()
        if widget.instance.isMinimized():  # if widget is minimized, show its normal size
            widget.instance.showNormal()

    # popMenuAction - user selected to rename active widget
    def renameActiveWidget(self):
        widgets = self.getSelectedWidgets()
        if len(widgets) != 1: return
        widget = widgets[0]

        exName = str(widget.caption)
        (newName ,ok) = QInputDialog.getText(self, "Rename Widget", "Enter new name for the \"" + exName + "\" widget:", QLineEdit.Normal, exName)
        newName = str(newName)
        if ok and newName != exName:
            for w in self.doc.widgets:
                if w != widget and w.caption == newName:
                    QMessageBox.information(self, 'Orange Canvas', 'Unable to rename widget. An instance with that name already exists.')
                    return
            widget.updateText(newName)
            widget.updateTooltip()
            widget.updateLinePosition()
            widget.instance.setCaption(newName)

    # popMenuAction - user selected to delete active widget
    def removeActiveWidget(self):
        if self.doc.signalManager.signalProcessingInProgress:
            QMessageBox.information( self, "Orange Canvas", "Unable to remove widgets while signal processing is in progress. Please wait.")
            return

        selectedWidgets = self.getSelectedWidgets()
        if selectedWidgets == []:
            selectedWidgets = [self.tempWidget]

        for item in selectedWidgets:
            self.doc.removeWidget(item)

        self.scene().update()
        self.tempWidget = None
        self.doc.canvasDlg.widgetPopup.setEnabled(len(self.getSelectedWidgets()) == 1)

    # ###########################################
    # POPUP MENU - LINKS actions
    # ###########################################

    # popMenuAction - enable/disable link between two widgets
    def toggleEnabledLink(self):
        if self.selectedLine != None:
            oldEnabled = self.doc.signalManager.getLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance)
            self.doc.signalManager.setLinkEnabled(self.selectedLine.outWidget.instance, self.selectedLine.inWidget.instance, not oldEnabled)
            self.selectedLine.setEnabled(not oldEnabled)
            self.selectedLine.inWidget.updateTooltip()
            self.selectedLine.outWidget.updateTooltip()

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
        for item in self.doc.widgets:
            item.setSelected(0)

    # get a tuple (outWidget, inWidget) that contains the widgets whose 
    # outputs and inputs are near position pos
    def widgetChannelsAtPos(self, pos):
        right = [item.mouseInsideRightChannel(pos) for item in self.doc.widgets]
        left = [item.mouseInsideLeftChannel(pos) for item in self.doc.widgets]
        outWidget, inWidget = None, None
        if True in left:
            inWidget = self.doc.widgets[left.index(1)]
        if True in right:
            outWidget = self.doc.widgets[right.index(1)]
        return (outWidget, inWidget)

    def getItemsAtPos(self, pos, itemType = None):
        if type(pos) == QPointF:
            pos = QGraphicsRectItem(QRectF(pos, QSizeF(1,1)))
        items = self.scene().collidingItems(pos)
        if itemType != None:
            items = [item for item in items if type(item) == itemType]
        return items

    # ###########################################
    # MOUSE events
    # ###########################################

    # mouse button was pressed
    def mousePressEvent(self, ev):
        self.mouseDownPosition = self.mapToScene(ev.pos())

        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        # do we start drawing a connection line
        if ev.button() == Qt.LeftButton:
            outWidget, inWidget = self.widgetChannelsAtPos(self.mouseDownPosition)
            activeItem = outWidget or inWidget
            self.tempWidget = activeItem
            self.inWidget, self.outWidget = inWidget, outWidget
            if activeItem:
                activeItem.setSelected(1)
                if not self.doc.signalManager.signalProcessingInProgress:   # if we are processing some signals, don't allow to add lines
                    self.unselectAllWidgets()
                    self.bLineDragging = True
                    pos = activeItem.getEdgePoint(self.mouseDownPosition)
                    self.tempLine = orngCanvasItems.TempCanvasLine(self.doc.canvasDlg, self.scene())
                    self.tempLine.setLine(pos.x(), pos.y(), pos.x(), pos.y())
                    self.tempLine.setPen(QPen(self.doc.canvasDlg.lineColor, 1))
                    if activeItem.mouseInsideLeftChannel(self.mouseDownPosition):
                        for widget in self.doc.widgets:
                            widget.canConnect(widget, activeItem)
                    else:
                        for widget in self.doc.widgets:
                            widget.canConnect(activeItem, widget)
                                                        
                self.scene().update()
                self.doc.canvasDlg.widgetPopup.setEnabled(len(self.getSelectedWidgets()) == 1)
                return
            
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

                # did we click inside the boxes to draw connections
                if ev.button() == Qt.LeftButton:
                    self.bWidgetDragging = True
                    if self.doc.ctrlPressed:
                        activeItem.setSelected(not activeItem.isSelected())
                    elif activeItem.isSelected() == 0:
                        self.unselectAllWidgets()
                        activeItem.setSelected(1)
                        self.bMultipleSelection = False

                    for w in self.getSelectedWidgets():
                        w.savePosition()
                        w.setAllLinesFinished(False)

                # is we clicked the right mouse button we show the popup menu for widgets
                elif ev.button() == Qt.RightButton:
                    self.unselectAllWidgets()
                    activeItem.setSelected(1)
                    self.bMultipleSelection = False
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

        self.doc.canvasDlg.widgetPopup.setEnabled(len(self.getSelectedWidgets()) == 1)
        self.scene().update()


    # mouse button was pressed and mouse is moving ######################
    def mouseMoveEvent(self, ev):
        point = self.mapToScene(ev.pos())
        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        if self.bWidgetDragging:
            for item in self.getSelectedWidgets():
                newPos = item.oldPos + (point-self.mouseDownPosition)
                item.setCoords(newPos.x(), newPos.y())
                item.invalidPosition = (self.findItemTypeCount(self.scene().collidingItems(item), orngCanvasItems.CanvasWidget) > 0)
                item.updateLineCoords()
            self.scene().update()

        elif self.bLineDragging:
            if self.tempLine:
                self.tempLine.setLine(self.tempLine.line().x1(), self.tempLine.line().y1(), point.x(), point.y())
            self.scene().update()

        elif self.bMultipleSelection:
            rect = QRectF(min (self.mouseDownPosition.x(), point.x()), min (self.mouseDownPosition.y(), point.y()), abs(self.mouseDownPosition.x() - point.x()), abs(self.mouseDownPosition.y() - point.y()))
            self.tempRect = QGraphicsRectItem(rect, None, self.scene())
            self.tempRect.show()

            # select widgets in rectangle
            widgets = self.getItemsAtPos(self.tempRect, orngCanvasItems.CanvasWidget)
            for widget in self.doc.widgets:
                widget.setSelected(widget in widgets)
            self.scene().update()


    # mouse button was released #########################################
    def mouseReleaseEvent(self, ev):
        point = self.mapToScene(ev.pos())
        if self.tempRect:
            self.tempRect.hide()
            self.tempRect = None

        # if we are moving a widget
        if self.bWidgetDragging:
            validPos = True
            for item in self.getSelectedWidgets():
                items = self.scene().collidingItems(item)
                validPos = validPos and (self.findItemTypeCount(items, orngCanvasItems.CanvasWidget) == 0)

            for item in self.getSelectedWidgets():
                if not validPos:
                    item.restorePosition()
                item.invalidPosition = False
                item.updateTooltip()
                item.updateLineCoords()
                item.setAllLinesFinished(True)
                orngHistory.logChangeWidgetPosition(self.doc.schemaID, (item.widgetInfo.category, item.widgetInfo.name), item.x(), item.y())


        # if we are drawing line
        elif self.bLineDragging:
            # show again the empty input/output boxes
            for widget in self.doc.widgets:
              widget.resetLeftRightEdges()      
            widgets = self.getItemsAtPos(point, orngCanvasItems.CanvasWidget)
            item = len(widgets) > 0 and widgets[0] or None
            if not item:
                outWidget, inWidget = self.widgetChannelsAtPos(point)
                item = inWidget or outWidget

            p1 = self.tempLine.line().p1()
            if self.tempLine != None:
                self.tempLine.hide()
                self.tempLine = None

            # we must check if we have really connected some output to input
            if type(item) == orngCanvasItems.CanvasWidget and self.tempWidget and item != self.tempWidget:
                if self.doc.signalManager.signalProcessingInProgress:
                     QMessageBox.information( self, "Orange Canvas", "Unable to connect widgets while signal processing is in progress. Please wait.")
                else:
                    if self.tempWidget.mouseInsideLeftChannel(p1):
                        line = self.doc.addLine(item, self.tempWidget)
                    else:
                        line = self.doc.addLine(self.tempWidget, item)
            else:
                if self.outWidget:
                    orngTabs.categoriesPopup.selectByInputs(self.outWidget.widgetInfo)
                else:
                    orngTabs.categoriesPopup.selectByOutputs(self.inWidget.widgetInfo)
                newCoords = QPoint(ev.globalPos())
                action = orngTabs.categoriesPopup.exec_(newCoords)
                if action:
                    xOff = -48 * bool(self.inWidget)
                    newWidget = self.doc.addWidget(action.widgetInfo, point.x()+xOff, point.y()-24)
                    if self.doc.signalManager.signalProcessingInProgress:
                        QMessageBox.information( self, "Orange Canvas", "Unable to connect widgets while signal processing is in progress. Please wait.")
                    else:
                        line = self.doc.addLine(self.outWidget or newWidget, self.inWidget or newWidget)

        else:
            activeItem = self.scene().itemAt(point)
            if not activeItem:
                if (self.mouseDownPosition.x() - point.x())**2 + (self.mouseDownPosition.y() - point.y())**2 < 25:
                    newCoords = QPoint(ev.globalPos())
                    orngTabs.categoriesPopup.enableAll()
                    action = orngTabs.categoriesPopup.exec_(newCoords)
                    if action:
                        newWidget = self.doc.addWidget(action.widgetInfo, point.x(), point.y())
                    

        self.scene().update()
        self.bWidgetDragging = False
        self.bLineDragging = False
        self.bMultipleSelection = False
        self.doc.canvasDlg.widgetPopup.setEnabled(len(self.getSelectedWidgets()) == 1)

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

    # ###########################################
    # Functions for processing events
    # ###########################################

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

    # ###########################################
    # misc functions regarding item selection
    # ###########################################

    # return a list of widgets that are currently selected
    def getSelectedWidgets(self):
        return [widget for widget in self.doc.widgets if widget.isSelected()]

    # return number of items in "items" of type "type"
    def findItemTypeCount(self, items, Type):
        return sum([type(item) == Type for item in items])


