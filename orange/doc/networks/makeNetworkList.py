import os
import os.path
import glob

import orange
import orngNetwork

atts = []
atts.append(orange.StringVariable("Network Name"))
atts.append(orange.StringVariable("Network File"))
atts.append(orange.StringVariable("dir"))
atts.append(orange.StringVariable("Item Set"))
atts.append(orange.StringVariable("Edge Set"))
atts.append(orange.FloatVariable("Vertices"))
atts[-1].numberOfDecimals = 0
atts.append(orange.FloatVariable("Edges"))
atts[-1].numberOfDecimals = 0
atts.append(orange.StringVariable("Date"))
atts.append(orange.StringVariable("Description"))

netlist = orange.ExampleTable(orange.Domain(atts, False))

for netFile in glob.glob(os.path.join(os.getcwd(), '*.net')):
	net = orngNetwork.Network.read(netFile)
	name, ext = os.path.splitext(netFile)
	
	itemFile = ""
	if os.path.exists(name + '_items.tab'):
		itemFile = name + '_items.tab'
	elif os.path.exists(name + '.tab'):
		itemFile = name + '.tab'
	
	edgeFile = ""
	if os.path.exists(name + '_edges.tab'):
		edgeFile = name + '_edges.tab'
	
	netlist.append([net.name, os.path.basename(netFile), "doc/datasets/", os.path.basename(itemFile), os.path.basename(edgeFile), net.nVertices, len(net.getEdges()), "4/12/2010", net.description])
	
netlist.save("network_info.tab")	