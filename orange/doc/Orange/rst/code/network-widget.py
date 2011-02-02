import sys
from PyQt4.QtGui import QApplication

appl = QApplication(sys.argv)

import Orange.data
import Orange.network
import OWNetExplorer

net = Orange.network.Network.read('musicians.net')
net.items = Orange.data.Table('musicians_items.tab')

ow = OWNetExplorer.OWNetExplorer()
ow.setGraph(net)

ow.show()
appl.exec_()