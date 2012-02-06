import sys
from PyQt4.QtGui import QApplication

appl = QApplication(sys.argv)

import orange
import orngNetwork
import OWNetExplorer

net = orngNetwork.Network.read('musicians.net')
net.items = orange.ExampleTable('musicians_items.tab')

ow = OWNetExplorer.OWNetExplorer()
ow.setNetwork(net)

ow.show()
appl.exec_()