import OWGUI
from OWWidget import *

class OWVisWidget(OWWidget):
    def hasDiscreteClass(self, data = -1):
        if data == -1:
            data = getattr(self, "data", None)
        return data != None and data.domain.classVar != None and data.domain.classVar.varType == orange.VarTypes.Discrete

    def createShowHiddenLists(self, placementTab, callback = None):
        maxWidth = 180
        self.updateCallbackFunction = callback
        self.shownAttributes = []
        self.selectedShown = []
        self.hiddenAttributes = []
        self.selectedHidden = []

        self.shownAttribsGroup = OWGUI.widgetBox(placementTab, " Shown attributes " )
        self.addRemoveGroup = OWGUI.widgetBox(placementTab, 1, orientation = "horizontal" )
        self.hiddenAttribsGroup = OWGUI.widgetBox(placementTab, " Hidden attributes ")

        hbox = OWGUI.widgetBox(self.shownAttribsGroup, orientation = 'horizontal')
        self.shownAttribsLB = OWGUI.listBox(hbox, self, "selectedShown", "shownAttributes", callback = self.resetAttrManipulation, selectionMode = QListWidget.ExtendedSelection)
        #self.shownAttribsLB.setMaximumWidth(maxWidth)
        vbox = OWGUI.widgetBox(hbox, orientation = 'vertical')
        self.buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attributes up")
        self.buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attributes down")
        self.buttonUPAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonUPAttr.setMaximumWidth(20)
        self.buttonDOWNAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonDOWNAttr.setMaximumWidth(20)
        self.buttonUPAttr.setMaximumWidth(20)

        self.attrAddButton =    OWGUI.button(self.addRemoveGroup, self, "", callback = self.addAttribute, tooltip="Add (show) selected attributes")
        self.attrAddButton.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_up2.png")))
        self.attrRemoveButton = OWGUI.button(self.addRemoveGroup, self, "", callback = self.removeAttribute, tooltip="Remove (hide) selected attributes")
        self.attrRemoveButton.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_down2.png")))
        self.showAllCB = OWGUI.checkBox(self.addRemoveGroup, self, "showAllAttributes", "Show all", callback = self.cbShowAllAttributes)

        self.hiddenAttribsLB = OWGUI.listBox(self.hiddenAttribsGroup, self, "selectedHidden", "hiddenAttributes", callback = self.resetAttrManipulation, selectionMode = QListWidget.ExtendedSelection)
        #self.hiddenAttribsLB.setMaximumWidth(maxWidth + 27)


    def resetAttrManipulation(self):
        if self.selectedShown:
            mini, maxi = min(self.selectedShown), max(self.selectedShown)
            tightSelection = maxi - mini == len(self.selectedShown) - 1
        self.buttonUPAttr.setEnabled(self.selectedShown != [] and tightSelection and mini)
        self.buttonDOWNAttr.setEnabled(self.selectedShown != [] and tightSelection and maxi < len(self.shownAttributes)-1)
        self.attrAddButton.setDisabled(not self.selectedHidden or self.showAllAttributes)
        self.attrRemoveButton.setDisabled(not self.selectedShown or self.showAllAttributes)
        if self.data and self.hiddenAttributes and self.data.domain.classVar and self.hiddenAttributes[0][0] != self.data.domain.classVar.name:
            self.showAllCB.setChecked(0)


    def moveAttrSelection(self, labels, selection, dir):
        self.graph.insideColors = None
        self.graph.clusterClosure = None

        labs = getattr(self, labels)
        sel = getattr(self, selection)
        mini, maxi = min(sel), max(sel)+1
        if dir == -1:
            setattr(self, labels, labs[:mini-1] + labs[mini:maxi] + [labs[mini-1]] + labs[maxi:])
        else:
            setattr(self, labels, labs[:mini] + [labs[maxi]] + labs[mini:maxi] + labs[maxi+1:])
        setattr(self, selection, map(lambda x:x+dir, sel))

        self.resetAttrManipulation()
        self.sendShownAttributes()
        self.graph.potentialsBmp = None
        if self.updateCallbackFunction:
            self.updateCallbackFunction()
        self.graph.removeAllSelections()

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        self.moveAttrSelection("shownAttributes", "selectedShown", -1)

    # move selected attribute in "Attribute Order" list one place down
    def moveAttrDOWN(self):
        self.moveAttrSelection("shownAttributes", "selectedShown", 1)


    def cbShowAllAttributes(self):
        if self.showAllAttributes:
            self.addAttribute(True)
        self.resetAttrManipulation()

    def addAttribute(self, addAll = False):
        self.graph.insideColors = None
        self.graph.clusterClosure = None

        if addAll:
            if self.data:
                self.setShownAttributeList(self.data, [attr.name for attr in self.data.domain.attributes])
        else:
            self.setShownAttributeList(self.data, self.shownAttributes + [self.hiddenAttributes[i] for i in self.selectedHidden])
        self.selectedHidden = []
        self.selectedShown = []
        self.resetAttrManipulation()

        if self.graph.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.getShownAttributeList())

        self.sendShownAttributes()
        if self.updateCallbackFunction: self.updateCallbackFunction()
        self.graph.replot()
        self.graph.removeAllSelections()

    def removeAttribute(self):
        self.graph.insideColors = None
        self.graph.clusterClosure = None

        newShown = self.shownAttributes[:]
        self.selectedShown.sort(lambda x,y:-cmp(x, y))
        for i in self.selectedShown:
            del newShown[i]
        self.setShownAttributeList(self.data, newShown)

        if self.graph.globalValueScaling == 1:
            self.graph.rescaleAttributesGlobaly(self.getShownAttributeList())
        self.sendShownAttributes()
        if self.updateCallbackFunction: self.updateCallbackFunction()
        self.graph.replot()
        self.graph.removeAllSelections()

    def getShownAttributeList(self):
        return [a[0] for a in self.shownAttributes]


    def setShownAttributeList(self, data, shownAttributes = None):
        shown = []
        hidden = []

        if data:
            if shownAttributes:
                if type(shownAttributes[0]) == tuple:
                    shown = shownAttributes
                else:
                    domain = self.data.domain
                    shown = [(domain[a].name, domain[a].varType) for a in shownAttributes]
                hidden = filter(lambda x:x not in shown, [(a.name, a.varType) for a in data.domain.attributes])
            else:
                shown = [(a.name, a.varType) for a in data.domain.attributes]
                if not self.showAllAttributes:
                    hidden = shown[10:]
                    shown = shown[:10]

            if data.domain.classVar and (data.domain.classVar.name, data.domain.classVar.varType) not in shown:
                hidden += [(data.domain.classVar.name, data.domain.classVar.varType)]

        self.shownAttributes = shown
        self.hiddenAttributes = hidden
        self.selectedHidden = []
        self.selectedShown = []
        self.resetAttrManipulation()

        self.sendShownAttributes()

