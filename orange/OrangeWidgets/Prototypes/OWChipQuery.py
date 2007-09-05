"""
<name>Chip Query</name>
<description>Queries a microarray database.</description>
<icon>icons/ChipQuery.png</icon>
<priority>11</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import urllib
import OWGUI

class OWChipQuery(OWWidget):
    settingsList=["serverURL", "biologicalSample", "timePoint", "treatment", "groupBy", "hideRefChannel"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "ChipQuery")

        self.inputs = []
        self.outputs = [("Chip data", ExampleTable)]

        self.biologicalSample = self.timePoint = self.treatment = 0
        
        self.allGroups = ["Biological sample", "Time point", "Chip set id", "Biological sample id", "Extraction id"]
        self.groupAttrs = ["biologcal.sample", "extraction.developmental_time_point", "chip.chip_set_id", "biologcal.biological_sample_id", "extraction.extraction_id"]
        self.groupBy = []
        
        self.allNormalizations = []
        self.normalization = []
        self.hideRefChannel = 1
        
        self.serverURL = "212.235.189.61/index.php"
        self.loadSettings()

        self.hierarchy = None
        self.groups = None

        box = OWGUI.widgetBox(self.controlArea, "Server URL", orientation="horizontal", addSpace = True)
        OWGUI.lineEdit(box, self, "serverURL", "", callback = self.getPossibleAnnotations)
#        OWGUI.button(box, self, "Set", self.getAnnotations)

        box = OWGUI.widgetBox(self.controlArea, "Filter", addSpace = True)
        self.lbfBiologicalSample = OWGUI.comboBox(box, self, "biologicalSample", label="Biological sample", labelWidth = 90, orientation = 0, callback = self.queryChanged)
        self.lbfTreatment = OWGUI.comboBox(box, self, "treatment", label="Treatment", labelWidth = 90, orientation = 0, callback = self.queryChanged)
        self.lbfTimePoint = OWGUI.comboBox(box, self, "timePoint", label="Time point", labelWidth = 90, orientation = 0, callback = self.queryChanged)
        OWGUI.separator(box)

#        box = OWGUI.widgetBox(self.controlArea, "Preview", orientation = 0)
#        OWGUI.label(box, self, "Selected chips: %(nChips)s")
        self.btRetrieveAnnotations = OWGUI.button(box, self, "Retrieve Annotations", callback = self.retrieveAnnotations)
                
        box = OWGUI.widgetBox(self.controlArea, "Normalization and grouping", addSpace = True)
        self.cbNormalizations = OWGUI.comboBox(box, self, "normalization", label="Normalization")
        OWGUI.separator(box)

        OWGUI.widgetLabel(box, "Grouping")
        OWGUI.listBox(box, self, "groupBy", "allGroups",  selectionMode = QListBox.Multi, callback=[self.constructHierarchy, self.updateChipAnnotations])
        
        self.getPossibleAnnotations()
        self.queryChanged()
        
        self.layout=QVBoxLayout(self.mainArea)

        box = OWGUI.radioButtonsInBox(self.mainArea, self, "hideRefChannel", [], "Annotations", callback = self.updateAnnotationColumns)
        self.layout.addWidget(box)
        hbox = OWGUI.widgetBox(box, orientation =0)
        for rb in ["Show both channels", "Hide reference channel"]:
            OWGUI.appendRadioButton(box, self, "hideRefChannel", rb, insertInto = hbox)
        
        self.lvAnnotations = QListView(box)
        self.connect(self.lvAnnotations, SIGNAL("selectionChanged()"), self.updateDetails)
        self.updateAnnotationColumns()
        
        self.tvDetails = QTextView(box)
        

        hb = OWGUI.widgetBox(self.controlArea, "Query", "horizontal")
        OWGUI.button(hb, self, "Query", callback = self.getData)

        self.adjustSize()

    
    def sendQuery(self, action, args):
        url = "http://%s?action=%s%s" % (self.serverURL, action, args and "&"+args or "")
        print url
        for i in range(5):
            try:
                r = urllib.urlopen(url)
                return r
            except:
                pass

    def getPossibleAnnotations(self):
        for annotations, listbox, id in (("allBiologicalSamples", "lbfBiologicalSample", 6),
                                         ("allTreatments", "lbfTreatment", 8),
                                         ("allTimePoints", "lbfTimePoint", 30)
                                        ):
            r = self.sendQuery("get_possible_annotation_options", "id=%i" % id)
            if not r:
                self.error("Cannot retrieve the annotations from the server")
                return
            r.readline()
            l = [x.strip() for x in r.readlines()]
            setattr(self, annotations, [None]+l)
            lb = getattr(self, listbox)
            lb.clear()
            lb.insertItem("<any>")
            for i in l:
                lb.insertItem(i)
        
    def queryChanged(self):
        queryParts = []
        for annotations, attribute, queryS in (("allBiologicalSamples", "biologicalSample", "biological_samples,sample,"),
                                               ("allTreatments", "treatment", "biological_samples,treatment,"),
                                               ("allTimePoints", "timePoint", "extractions,developmental%20time%20point,")):
            value = getattr(self, annotations)[getattr(self, attribute)]
            if not value is None:
                queryParts.append(queryS+value)

        r = self.sendQuery("search", queryParts and "&query="+"|".join(queryParts))
        if not r:
            self.error("Cannot retrieve the data from the server")
            return
                    
        r.readline()
        self.filteredIDs = [int(x.strip()) for x in r if x.strip()]
        self.btRetrieveAnnotations.setText("Retrieve Annotations for %i Selected Chips" % len(self.filteredIDs))


    def appendAnnotation(self, annotation):
        try:
            expChannel = annotation["ch1_biologcal.sample"] != "AX4" and 1 or 2
            for key, value in annotation.items():
                if key.startswith("ch%i" % expChannel):
                    annotation[key[4:]] = value
            self.chipAnnotations.append(annotation)
    
            nnpref = "normalizations_data_sets."
            lennpref = len(nnpref)
    
            normalizations = [k[lennpref:] for k in annotation.keys() if k.startswith(nnpref)]
            if not normalizations: # the line must be missing, but we always have loess
                normalizations = ["loess"]
            if self.allNormalizations is None:
                self.allNormalizations = normalizations
            else:
                self.allNormalizations = [n for n in self.allNormalizations if n in normalizations]
        except:
            pass
        
    def retrieveAnnotations(self):
        print "retrieving"
        self.chipAnnotations = []
        self.allNormalizations = None

        r = self.sendQuery("get_chip_annotation", "ids=" + ",".join([str(i) for i in self.filteredIDs]))
        if not r:
            self.error("Cannot retrieve the annotations from the server")
            return

        annotation = {}
        for line in r:
            line = line.strip()
            if not line and annotation:
                self.appendAnnotation(annotation)
                annotation = {}
                
            t = line.split("\t")[1:]
            if len(t) == 2:
                annotation[t[0]] = t[1]
                
        if annotation:
            self.appendAnnotation(annotation)

        print "retrieved"

        self.hierarchy = None
        self.updateChipAnnotations()
        
        self.cbNormalizations.clear()
        return
        for n in self.allNormalizations:
            self.cbNormalizations.insertItem(n)
        self.normalization = 0
            
        
    def updateAnnotationColumns(self):
        if self.hideRefChannel:
            colnames = ("Sample", " Dye ", " Time ", "Chip set id")
        else:
            colnames = ("Channel 1", "...dye ", "...time ", "Channel 2 ", "...dye ", "...time ", "Chip set id")
            
        self.lvAnnotations.clear()
        while self.lvAnnotations.columns():
            self.lvAnnotations.removeColumn(0)
            
        for i, colname in enumerate(colnames):
            self.lvAnnotations.addColumn(colname)
            self.lvAnnotations.setColumnAlignment(i, Qt.AlignCenter)
            
        if hasattr(self, "chipAnnotations"):
            self.updateChipAnnotations()
        
    def updateChipAnnotations0(self, parent, node, depth):
        if type(node) == dict:
            for key, child in node.items():
                item = QListViewItem(parent, key)
                self.updateChipAnnotations0(item, child, depth+1)
                item.setOpen(True)
        else:
            for chipId in node:
                chip = self.chipAnnotations[chipId]
                if self.hideRefChannel:
                    item = QListViewItem(parent, *tuple([chip.get(a, "") for a in ("biologcal.sample", "dye", "extraction.developmental_time_point",
                                                                                               "chip.chip_set_id")]))
                else:
                    item = QListViewItem(parent, *tuple([chip.get(a, "") for a in ("ch1_biologcal.sample", "ch1.dye", "ch1_extraction.developmental_time_point",
                                                                                               "ch2_biologcal.sample", "ch2.dye", "ch2_extraction.developmental_time_point",
                                                                                               "chip.chip_set_id")]))
                item.chip = chip
                
    def updateChipAnnotations(self):
        self.lvAnnotations.clear()
        if not self.hierarchy:
            self.constructHierarchy()
        item = QListViewItem(self.lvAnnotations, "Selected chips")
        self.updateChipAnnotations0(item, self.hierarchy, 0)
        item.setOpen(True)
        
    def updateDetails(self):
        item = self.lvAnnotations.currentItem()
        
        if not hasattr(item, "chip"):
            self.tvDetails.setText("")
            return
        
        chip = item.chip
        tooltip = ""
        for ch in range(1, 3):
            tooltip += "Channel %i<br>" % ch
            tooltip += "<br>".join(["%s: %s" % (desc, chip.get(attr % ch, "")) for desc, attr in (("biological sample", "ch%i_biologcal.sample"),
                                                                                                 ("growth condition", "ch%i_biologcal.growth_condition"),
                                                                                                 ("sample type", "ch%i_biologcal.sample_type"),
                                                                                                 ("user", "ch%i_biologcal.user")
                                                                                                 )])
            tooltip += "<br><br>"
        tooltip += "<br>".join(["%s: %s" % (desc, chip.get(attr, "")) for desc, attr in (("Chip set id", "chip.chip_set_id"),
                                                                                             ("User", "chip.user"))])
        self.tvDetails.setText(tooltip)
    
    def constructHierarchy0(self, node, criteria):
        res = {}
        crit = criteria[0]
        for chipId in node:
            ann = self.chipAnnotations[chipId][crit]
            if res.has_key(ann):
                res[ann].append(chipId)
            else:
                res[ann] = [chipId]
        if len(criteria) == 1:
            self.groups.extend(res.values())
            return res
        else:
            return dict([(key, self.constructHierarchy0(chip, criteria[1:])) for key, chip in res.items()])
        
    def constructHierarchy(self):
        criteria = [self.groupAttrs[g] for g in self.groupBy]
        chipIdxs = range(len(self.chipAnnotations))
        if not criteria:
            self.hierarchy = chipIdxs
            self.groups = [chipIdxs]
        else:
            self.groups = []
            self.hierarchy = self.constructHierarchy0(chipIdxs, criteria)
            
        
    def getData(self):
        r = self.sendQuery("get_raw_chip", "id=%s" % self.chipId.strip())
        if not r:
            self.error("Cannot retrieve the data from the server")
            self.send("Chip data", None)
            return
        outf = file("c:\\d\\cq.txt", "wt").write(r.read().replace("\tspot_id\t", "\tS#spot_id\t"))
        try:
            data = orange.ExampleTable("c:\\d\\cq.txt", noClass=True)
        except:
            self.error("Cannot read the data, possibly due to invalid format")
            self.send("Chip data", None)
            return
        
        self.send("Chip data", data)
        self.error()
