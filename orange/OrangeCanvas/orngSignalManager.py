# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#    manager, that handles correct processing of widget signals
#

import sys, time
import orange
import orngDebugging

Single = 2
Multiple = 4

Default = 8
NonDefault = 16

class InputSignal:
    def __init__(self, name, signalType, handler, parameters = Single + NonDefault, oldParam = 0):
        self.name = name
        self.type = signalType
        self.handler = handler

        if type(parameters) == str: parameters = eval(parameters)   # parameters are stored as strings
        # if we have the old definition of parameters then transform them
        if parameters in [0,1]:
            self.single = parameters
            self.default = not oldParam
            return

        if not (parameters & Single or parameters & Multiple): parameters += Single
        if not (parameters & Default or parameters & NonDefault): parameters += NonDefault
        self.single = parameters & Single
        self.default = parameters & Default

class OutputSignal:
    def __init__(self, name, signalType, parameters = NonDefault):
        self.name = name
        self.type = signalType

        if type(parameters) == str: parameters = eval(parameters)
        if parameters in [0,1]: # old definition of parameters
            self.default = not parameters
            return

        if not (parameters & Default or parameters & NonDefault): parameters += NonDefault
        self.default = parameters & Default


# class that allows to process only one signal at a time
class SignalWrapper:
    def __init__(self, widget, method):
        self.widget = widget
        self.method = method

    def __call__(self, *k):
        manager = self.widget.signalManager
        if not manager:
            manager = signalManager

        manager.signalProcessingInProgress += 1
        try:
            self.method(*k)
        finally:
            manager.signalProcessingInProgress -= 1
            if not manager.signalProcessingInProgress:
                manager.processNewSignals(self.widget)



class SignalManager:
    widgets = []    # topologically sorted list of widgets
    links = {}      # dicionary. keys: widgetFrom, values: (widgetTo1, signalNameFrom1, signalNameTo1, enabled1), (widgetTo2, signalNameFrom2, signalNameTo2, enabled2)
    freezing = 0            # do we want to process new signal immediately
    signalProcessingInProgress = 0 # this is set to 1 when manager is propagating new signal values

    def __init__(self, *args):
        self.debugFile = None
        self.verbosity = orngDebugging.orngVerbosity
        self.stderr = sys.stderr
        self._seenExceptions = {}
        #self.stdout = sys.stdout
        if orngDebugging.orngDebuggingEnabled:
            self.debugFile = open(orngDebugging.orngDebuggingFileName, "wt")
            sys.excepthook = self.exceptionHandler
            sys.stderr = self.debugFile
            #sys.stdout = self.debugFile

    def setDebugMode(self, debugMode = 0, debugFileName = "signalManagerOutput.txt", verbosity = 1):
        self.verbosity = verbosity
        if self.debugFile:
            sys.stderr = self.stderr
            #sys.stdout = self.stdout
            self.debugFile.close()
            self.debugFile = None
        if debugMode:
            self.debugFile = open(debugFileName, "wt", 0)
            sys.excepthook = self.exceptionHandler
            sys.stderr = self.debugFile
            #sys.stdout = self.debugFile

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # DEBUGGING FUNCTION

    def closeDebugFile(self):
        if self.debugFile:
            self.debugFile.close()
            self.debugFile = None
        sys.stderr = self.stderr
        #sys.stdout = self.stdout

#    def __del__(self):
#        if not self or type(self) == type(None):
#            return
#        if self.debugFile:
#            self.debugFile.close()
#        sys.stderr = self.stderr
#        #sys.stdout = self.stdout

    #
    def addEvent(self, strValue, object = None, eventVerbosity = 1):
        if not self.debugFile: return
        if self.verbosity < eventVerbosity: return

        self.debugFile.write(str(strValue))
        if isinstance(object, orange.ExampleTable):
            name = " " + getattr(object, "name", "")
            self.debugFile.write(". Token type = ExampleTable" + name + ". len = " + str(len(object)))
        elif type(object) == list:
            self.debugFile.write(". Token type = %s. Value = %s" % (str(type(object)), str(object[:10])))
        elif object != None:
            self.debugFile.write(". Token type = %s. Value = %s" % (str(type(object)), str(object)[:100]))
        self.debugFile.write("\n")
        self.debugFile.flush()


    def exceptionSeen(self, type, value, tracebackInfo):
        import traceback, os
        shortEStr = "".join(traceback.format_exception(type, value, tracebackInfo))[-2:]
        return self._seenExceptions.has_key(shortEStr)

    def exceptionHandler(self, type, value, tracebackInfo):
        import traceback, os
        if not self.debugFile: return

        # every exception show only once
        shortEStr = "".join(traceback.format_exception(type, value, tracebackInfo))[-2:]
        if self._seenExceptions.has_key(shortEStr): return
        self._seenExceptions[shortEStr] = 1
        
        list = traceback.extract_tb(tracebackInfo, 10)
        space = "\t"
        totalSpace = space
        self.debugFile.write("Unhandled exception of type %s\n" % ( str(type)))
        self.debugFile.write("Traceback:\n")

        for i, (file, line, funct, code) in enumerate(list):
            if not code: continue
            self.debugFile.write(totalSpace + "File: " + os.path.split(file)[1] + " in line %4d\n" %(line))
            self.debugFile.write(totalSpace + "Function name: %s\n" % (funct))
            self.debugFile.write(totalSpace + "Code: " + code + "\n")
            totalSpace += space

        self.debugFile.write(totalSpace[:-1] + "Exception type: " + str(type) + "\n")
        self.debugFile.write(totalSpace[:-1] + "Exception value: " + str(value)+ "\n")
        self.debugFile.flush()

    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # freeze/unfreeze signal processing. If freeze=1 no signal will be processed until freeze is set back to 0
    def setFreeze(self, freeze, startWidget = None):
        self.freezing = freeze
        if not freeze and self.widgets != []:
            if startWidget: self.processNewSignals(startWidget)
            else: self.processNewSignals(self.widgets[0])

    # add widget to list
    def addWidget(self, widget):
        if self.verbosity >= 2:
            self.addEvent("added widget " + widget.captionTitle, eventVerbosity = 2)

        if widget not in self.widgets:
            #self.widgets.insert(0, widget)
            self.widgets.append(widget)

    # remove widget from list
    def removeWidget(self, widget):
        if self.verbosity >= 2:
            self.addEvent("remove widget " + widget.captionTitle, eventVerbosity = 2)
        self.widgets.remove(widget)


    # send list of widgets, that send their signal to widget's signalName
    def getLinkWidgetsIn(self, widget, signalName):
        widgets = []
        for key in self.links.keys():
            links = self.links[key]
            for (widgetTo, signalFrom, signalTo, enabled) in links:
                if widget == widgetTo and signalName == signalTo: widgets.append(key)
        return widgets

    # send list of widgets, that widget "widget" sends his signal "signalName"
    def getLinkWidgetsOut(self, widget, signalName):
        widgets = []
        if not self.links.has_key(widget): return widgets
        links = self.links[widget]
        for (widgetTo, signalFrom, signalTo, enabled) in links:
            if signalName == signalFrom: widgets.append(widgetTo)
        return widgets

    # can we connect widgetFrom with widgetTo, so that there is no cycle?
    def canConnect(self, widgetFrom, widgetTo):
        return not self.existsPath(widgetTo, widgetFrom)

    def addLink(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo, enabled):
        if self.verbosity >= 2:
            self.addEvent("add link from " + widgetFrom.captionTitle + " to " + widgetTo.captionTitle, eventVerbosity = 2)

        if not self.canConnect(widgetFrom, widgetTo): return 0
        # check if signal names still exist
        found = 0
        for o in widgetFrom.outputs:
            output = OutputSignal(*o)
            if output.name == signalNameFrom: found=1
        if not found:
            print "Error. Widget %s changed its output signals. It does not have signal %s anymore." % (str(getattr(widgetFrom, "captionTitle", "")), signalNameFrom)
            return 0

        found = 0
        for i in widgetTo.inputs:
            input = InputSignal(*i)
            if input.name == signalNameTo: found=1
        if not found:
            print "Error. Widget %s changed its input signals. It does not have signal %s anymore." % (str(getattr(widgetTo, "captionTitle", "")), signalNameTo)
            return 0


        if self.links.has_key(widgetFrom):
            for (widget, signalFrom, signalTo, Enabled) in self.links[widgetFrom]:
                if widget == widgetTo and signalNameFrom == signalFrom and signalNameTo == signalTo:
                    print "connection ", widgetFrom, " to ", widgetTo, " alread exists. Error!!"
                    return 0

        self.links[widgetFrom] = self.links.get(widgetFrom, []) + [(widgetTo, signalNameFrom, signalNameTo, enabled)]

        widgetTo.addInputConnection(widgetFrom, signalNameTo)

        # if there is no key for the signalNameFrom, create it and set its id=None and data = None
        if not widgetFrom.linksOut.has_key(signalNameFrom):
            widgetFrom.linksOut[signalNameFrom] = {None:None}

        # if channel is enabled, send data through it
        if enabled:
            for key in widgetFrom.linksOut[signalNameFrom].keys():
                widgetTo.updateNewSignalData(widgetFrom, signalNameTo, widgetFrom.linksOut[signalNameFrom][key], key, signalNameFrom)

        # reorder widgets if necessary
        if self.widgets.index(widgetFrom) > self.widgets.index(widgetTo):
            self.widgets.remove(widgetTo)
            self.widgets.append(widgetTo)   # appent the widget at the end of the list
            self.fixPositionOfDescendants(widgetTo)
##            print "--------"
##            for widget in self.widgets: print widget.captionTitle
##            print "--------"
        return 1

    # fix position of descendants of widget so that the order of widgets in self.widgets is consistent with the schema
    def fixPositionOfDescendants(self, widget):
        for link in self.links.get(widget, []):
            widgetTo = link[0]
            self.widgets.remove(widgetTo)
            self.widgets.append(widgetTo)
            self.fixPositionOfDescendants(widgetTo)


    # return list of signals that are connected from widgetFrom to widgetTo
    def findSignals(self, widgetFrom, widgetTo):
        signals = []
        for (widget, signalNameFrom, signalNameTo, enabled) in self.links.get(widgetFrom, []):
            if widget == widgetTo:
                signals.append((signalNameFrom, signalNameTo))
        return signals

    # is signal from widgetFrom to widgetTo with name signalName enabled?
    def isSignalEnabled(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo):
        for (widget, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
            if widget == widgetTo and signalFrom == signalNameFrom and signalTo == signalNameTo:
                return enabled
        return 0

    def removeLink(self, widgetFrom, widgetTo, signalNameFrom, signalNameTo):
        if self.verbosity >= 2:
            self.addEvent("remove link from " + widgetFrom.captionTitle + " to " + widgetTo.captionTitle, eventVerbosity = 2)

        # no need to update topology, just remove the link
        if self.links.has_key(widgetFrom):
            for (widget, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
                if widget == widgetTo and signalFrom == signalNameFrom and signalTo == signalNameTo:
                    for key in widgetFrom.linksOut[signalFrom].keys():
                        widgetTo.updateNewSignalData(widgetFrom, signalNameTo, None, key, signalNameFrom)
                    self.links[widgetFrom].remove((widget, signalFrom, signalTo, enabled))
                    if not self.freezing and not self.signalProcessingInProgress: self.processNewSignals(widgetFrom)
        widgetTo.removeInputConnection(widgetFrom, signalNameTo)


    # ############################################
    # ENABLE OR DISABLE LINK CONNECTION

    def setLinkEnabled(self, widgetFrom, widgetTo, enabled, justSend = False):
        links = self.links[widgetFrom]
        for i in range(len(links)):
            (widget, nameFrom, nameTo, e) = links[i]
            if widget == widgetTo:
                if not justSend:
                    links[i] = (widget, nameFrom, nameTo, enabled)
                if enabled:
                    for key in widgetFrom.linksOut[nameFrom].keys():
                        widgetTo.updateNewSignalData(widgetFrom, nameTo, widgetFrom.linksOut[nameFrom][key], key, nameFrom)

        if enabled: self.processNewSignals(widgetTo)


    def getLinkEnabled(self, widgetFrom, widgetTo):
        for (widget, nameFrom, nameTo, enabled) in self.links[widgetFrom]:      # it is enough that we find one signal connected from widgetFrom to widgetTo
            if widget == widgetTo:                                  # that we know wheather the whole link (all signals) is enabled or not
                return enabled


    # widget widgetFrom sends signal with name signalName and value value
    def send(self, widgetFrom, signalNameFrom, value, id):
        # add all target widgets new value and mark them as dirty
        # if not freezed -> process dirty widgets
        if self.verbosity >= 2:
            self.addEvent("send data from " + widgetFrom.captionTitle + ". Signal = " + signalNameFrom, value, eventVerbosity = 2)

        if not self.links.has_key(widgetFrom): return
        for (widgetTo, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
            if signalFrom == signalNameFrom and enabled == 1:
                #print "signal from ", widgetFrom, " to ", widgetTo, " signal: ", signalNameFrom, " value: ", value, " id: ", id
                widgetTo.updateNewSignalData(widgetFrom, signalTo, value, id, signalNameFrom)


        if not self.freezing and not self.signalProcessingInProgress:
            #print "processing new signals"
            self.processNewSignals(widgetFrom)

    # when a new link is created, we have to
    def sendOnNewLink(self, widgetFrom, widgetTo, signals):
        for (outName, inName) in signals:
            for key in widgetFrom.linksOut[outName].keys():
                widgetTo.updateNewSignalData(widgetFrom, inName, widgetFrom.linksOut[outName][key], key, outName)


    def processNewSignals(self, firstWidget):
        if len(self.widgets) == 0: return
        if self.signalProcessingInProgress: return

        if self.verbosity >= 2:
            self.addEvent("process new signals from " + firstWidget.captionTitle, eventVerbosity = 2)

        if firstWidget not in self.widgets:
            firstWidget = self.widgets[0]   # if some window that is not a widget started some processing we have to process new signals from the first widget

        # start propagating
        self.signalProcessingInProgress = 1

        index = self.widgets.index(firstWidget)
        for i in range(index, len(self.widgets)):
            if self.widgets[i].needProcessing:
                try:
                    self.widgets[i].processSignals()
                except:
                    type, val, traceback = sys.exc_info()
                    sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that it doesn't crash canvas

        # we finished propagating
        self.signalProcessingInProgress = 0


    def existsPath(self, widgetFrom, widgetTo):
        # is there a direct link
        if not self.links.has_key(widgetFrom): return 0

        for (widget, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
            if widget == widgetTo: return 1

        # is there a nondirect link
        for (widget, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
            if self.existsPath(widget, widgetTo): return 1

        # there is no link...
        return 0


# create a global instance of signal manager
globalSignalManager = SignalManager()

