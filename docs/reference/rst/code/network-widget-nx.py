import sys
from PyQt4.QtGui import QApplication

appl = QApplication(sys.argv)

import Orange.data
import Orange.network
import OWNxExplorer

net = Orange.network.readwrite.read('musicians.net')
net.set_items(Orange.data.Table('musicians_items.tab'))

ow = OWNxExplorer.OWNxExplorer()
ow.set_graph(net)

ow.show()
appl.exec_()