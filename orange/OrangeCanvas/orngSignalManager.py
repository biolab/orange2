# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#	manager, that handles correct processing of widget signals
#

import sys

class SignalManager:
	widgets = []	# topologically sorted list of widgets
	links = {}	  # dicionary. keys: widgetFrom, values: (widgetTo1, signalNameFrom1, signalNameTo1, enabled1), (widgetTo2, signalNameFrom2, signalNameTo2, enabled2)
	freezing = 0			# do we want to process new signal immediately
	signalProcessingInProgress = 0 # this is set to 1 when manager is propagating new signal values

	# freeze/unfreeze signal processing. If freeze=1 no signal will be processed until freeze is set back to 0
	def setFreeze(self, freeze, startWidget = None):
		self.freezing = freeze
		if not freeze and self.widgets != []:
			if startWidget: self.processNewSignals(startWidget)
			else: self.processNewSignals(self.widgets[0])

	# add widget to list
	def addWidget(self, widget):
		if widget not in self.widgets:
			#self.widgets.insert(0, widget)
			self.widgets.append(widget)

	# remove widget from list
	def removeWidget(self, widget):
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
		if not self.canConnect(widgetFrom, widgetTo): return 0
		# check if signal names still exist
		found = 0
		for (name, type) in widgetFrom.outputs:
			if name == signalNameFrom: found=1
		if not found:
			print "Error. Widget %s changed its output signals. It does not have signal %s anymore." % (str(widgetFrom.caption()), signalNameFrom)
			return 0

		found = 0
		for (name, type, handler, single) in widgetTo.inputs:
			if name == signalNameTo: found=1
		if not found:
			print "Error. Widget %s changed its input signals. It does not have signal %s anymore." % (str(widgetTo.caption()), signalNameTo)
			return 0


		if self.links.has_key(widgetFrom):
			for (widget, signalFrom, signalTo, Enabled) in self.links[widgetFrom]:
				if widget == widgetTo and signalNameFrom == signalFrom and signalNameTo == signalTo:
					print "connection ", widgetFrom, " to ", widgetTo, " alread exists. Error!!"
					return 0

		existingLinks = []
		if self.links.has_key(widgetFrom): existingLinks = self.links[widgetFrom]
		self.links[widgetFrom] = existingLinks + [(widgetTo, signalNameFrom, signalNameTo, enabled)]

		widgetTo.addInputConnection(widgetFrom, signalNameTo)
		if widgetFrom.linksOut.has_key(signalNameFrom) and enabled:
			widgetTo.updateNewSignalData(widgetFrom, signalNameTo, widgetFrom.linksOut[signalNameFrom][0], widgetFrom.linksOut[signalNameFrom][1])

		# update topology
		#currentIndex = self.widgets.index(widgetTo)+1
		#for i in range(self.widgets.index(widgetTo)+1, self.widgets.index(widgetFrom)+1):
		#	if self.existsPath(self.widgets[i], widgetTo):
		#		widget = self.widgets[i]
		#		self.widgets.remove(widget)
		#		self.widgets.insert(self.widgets.index(widgetTo), widget)

		if self.widgets.index(widgetTo) < self.widgets.index(widgetFrom):
			self.widgets.remove(widgetTo)
			self.widgets.insert(self.widgets.index(widgetFrom)+1, widgetTo)

		return 1


	# return list of signals that are connected from widgetFrom to widgetTo
	def findSignals(self, widgetFrom, widgetTo):
		signals = []
		if not self.links.has_key(widgetFrom): return []
		for (widget, signalNameFrom, signalNameTo, enabled) in self.links[widgetFrom]:
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
		# no need to update topology, just remove the link
		for (widget, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
			if widget == widgetTo and signalFrom == signalNameFrom and signalTo == signalNameTo:
				#print "signal Manager - remove link. removing ", widgetFrom, widgetTo, signalFrom, signalTo
				widgetTo.updateNewSignalData(widgetFrom, signalNameTo, None, None)
				self.links[widgetFrom].remove((widget, signalFrom, signalTo, enabled))
				if not self.freezing and not self.signalProcessingInProgress: self.processNewSignals(widgetFrom)


	# ############################################
	# ENABLE OR DISABLE LINK CONNECTION

	def setLinkEnabled(self, widgetFrom, widgetTo, enabled):
		for key in self.links.keys():
			links = self.links[key]
			for i in range(len(links)):
				(widget, nameFrom, nameTo, e) = links[i]
				if widget == widgetTo:
					links[i] = (widget, nameFrom, nameTo, enabled)
					if enabled: widgetTo.updateNewSignalData(widgetFrom, nameTo, widgetFrom.linksOut[nameFrom][0], widgetFrom.linksOut[nameFrom][1])
		if enabled:
			self.processNewSignals(widgetTo)

	
	def getLinkEnabled(self, widgetFrom, widgetTo):
		for (widget, nameFrom, nameTo, enabled) in self.links[widgetFrom]:	  # it is enough that we find one signal connected from widgetFrom to widgetTo
			if widget == widgetTo:								  # that we know wheather the whole link (all signals) is enabled or not
				return enabled


	# widget widgetFrom sends signal with name signalName and value value
	def send(self, widgetFrom, signalNameFrom, value, id):
		# add all target widgets new value and mark them as dirty
		# if not freezed -> process dirty widgets

		if not self.links.has_key(widgetFrom): return
		for (widgetTo, signalFrom, signalTo, enabled) in self.links[widgetFrom]:
			if signalFrom == signalNameFrom and enabled == 1:
				#print "signal from ", widgetFrom, " to ", widgetTo, " signal: ", signalNameFrom, " value: ", value, " id: ", id
				widgetTo.updateNewSignalData(widgetFrom, signalTo, value, id)
				

		if not self.freezing and not self.signalProcessingInProgress:
			#print "processing new signals"
			self.processNewSignals(widgetFrom)

	# when a new link is created, we have to 
	def sendOnNewLink(self, widgetFrom, widgetTo, signals):
		for (outName, inName) in signals:
			widgetTo.updateNewSignalData(widgetFrom, inName, widgetFrom.linksOut[outName][0], widgetFrom.linksOut[signalNameFrom][1])


	def processNewSignals(self, firstWidget):
		if self.signalProcessingInProgress: return
		if firstWidget not in self.widgets: return # we may have windows that are not widgets

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
signalManager = SignalManager()

