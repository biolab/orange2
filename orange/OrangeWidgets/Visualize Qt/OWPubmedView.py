"""
<name>Pubmed Network View</name>
<description></description>
<icon>icons/Network.png</icon>
<contact></contact> 
<priority>6450</priority>
"""

import Orange
import OWGUI

from OWWidget import *

class PubmedNetworkView(Orange.network.NxView):
    """Network Inside View
    
    """
    
    def __init__(self, parent):#, nhops, edge_threshold, algorithm, n_max_neighbors, k):
        Orange.network.NxView.__init__(self)
        
        self._nhops = 2
        self._edge_threshold = 0.5
        self.algorithm = 1 # 0 without clustering, 1 with clustering
        self._n_max_neighbors = 10
        self._center_nodes = []
        self.parent = parent
        self._hidden_nodes = []
        self._k_algorithm = 0.3
        
    def init_network(self, graph):
        self._network = graph
        
        if hasattr(self.parent, 'init_network'):
            self.parent.init_network()

        if graph is None:
            return None
                      
        return graph
        
    
    def update_network(self):
        print 'update_network'
        
        if self._center_nodes == []:
            return
        #if len(self._center_nodes) == 1:  
        #    return
        
        subnet = Orange.network.Graph()
        central_nodes, to_add = self._center_nodes[:], self._center_nodes[:]
        for l in range(self._nhops):
            for i in central_nodes:
                neig = sorted([x for x in self._network.neighbors(i) if self._network.edge[i][x]['weight'] > self._edge_threshold], reverse=True)
                if len(neig) > self._n_max_neighbors:
                    neig = neig[:self._n_max_neighbors]
                to_add.extend(neig)
            central_nodes = neig
        to_add = list(set(to_add))
        subnet.add_nodes_from([(x, self._network.node[x]) for x in to_add])
        nodes = subnet.nodes()
        while nodes:
            i = nodes.pop()
            neig = [x for x in self._network.neighbors(i) if x in nodes] #if net.edge[i][x]['weight'] > 0.5]
            subnet.add_weighted_edges_from([(i,x,w) for x,w in zip(neig, [self._network.edge[i][y]['weight'] for y in neig])])
        
        subnet.remove_nodes_from(self._hidden_nodes)
        subnet = self._propagate(subnet)
        
        if self._nx_explorer is not None:
            self._nx_explorer.change_graph(subnet)
        
        
    def set_nhops(self, nhops):
        self._nhops = nhops  
		
    def set_edge_threshold(self, edge_threshold):
        self._edge_threshold = edge_threshold

    def set_algorithm(self, algorithm):
        self._algorithm = algorithm
        
    def set_n_max_neighbors(self, n_max_neighbors):
        self._n_max_neighbors = n_max_neighbors 
        
    def set_center_nodes(self, c_nodes):
        self._center_nodes = c_nodes
        
    def set_k(self, k):
        self._k_algorithm = k
    
    def node_selection_changed(self):
        # selection --> delete_node/delete_node_and_neig/expand/set_score ?
        
        selection = self._nx_explorer.networkCanvas.selected_nodes()
        # case "set_score"  - input = input_score, to be provided
                #self._center_nodes.extend(selection)  # Non servira'
        for i in selection:
            self._network.node[i]['user_score'] = 1
            self._network.node[i]['score'] = 1 # input_score
        # case "expand_node" - with the settings already set
        self._center_nodes.expand(selection)
        # case "delete_node"
        self._hidden_nodes.append(selection)
        # delete_node_and_neig --> ???? 
        
        self.update_network()  #chiama propagate
        
    def _get_neighbors(self):
    #TO DELETE?
        nodes = set([self._center_node])
        for n in range(self._nhops):
            neighbors = set()
            for node in nodes:
                neighbors.update(self._network.neighbors(node))
            nodes.update(neighbors)
        return nodes
        
    def _propagate(self, net):
        return
        
class OWPubmedView(OWWidget):
    
    settingsList = ['_nhops']
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Pubmed Network View', wantMainArea=0)
        
        self.inputs = []
        self.outputs = [("Nx View", Orange.network.NxView)]
        
        self._nhops = 2
        self._edge_threshold = 0.5
        self._n_max_neighbors = 20
        self.selected_titles = []
        self.titles = []
        self.filter = ''
        self.ids = []
        self._selected_nodes = []
        self._algorithm = 0
        self._k_algorithm = 0.3
        
        self.loadSettings()
        
        box = OWGUI.widgetBox(self.controlArea, "Paper Selection", orientation="vertical")
        OWGUI.lineEdit(box, self, "filter", callback=self.filter_list, callbackOnType=True)
        self.list_titles = OWGUI.listBox(box, self, "selected_titles", "titles", selectionMode=QListWidget.MultiSelection, callback=self.update_view)
        OWGUI.separator(self.controlArea)
        box_pref = OWGUI.widgetBox(self.controlArea, "Preferences", orientation="vertical")
        OWGUI.spin(box_pref, self, "_nhops", 1, 6, 1, label="Number of hops: ", callback=self.update_view)
        OWGUI.spin(box_pref, self, "_n_max_neighbors", 1, 100, 1, label="Max number of neighbors: ", callback=self.update_view)
        OWGUI.doubleSpin(box_pref, self, "_edge_threshold", 0, 1, step=0.01, label="Edge threshold: ", callback=self.update_view)
        OWGUI.separator(self.controlArea)
        box_alg = OWGUI.widgetBox(self.controlArea, "Interest Propagation Algorithm", orientation="vertical")
        radio_box = OWGUI.radioButtonsInBox(box_alg, self, "_algorithm", [], callback=self.update_view)
        OWGUI.appendRadioButton(radio_box, self, "_algorithm", "Without Clustering", callback=self.update_view)
        OWGUI.doubleSpin(OWGUI.indentedBox(radio_box), self, "_k_algorithm", 0, 1, step=0.01, label="Parameter k: ", callback=self.update_view)
        OWGUI.appendRadioButton(radio_box, self, "_algorithm", "With Clustering", callback=self.update_view)
        
        self.inside_view = PubmedNetworkView(self)
        self.send("Nx View", self.inside_view)
        
    
        
    def init_network(self):
        if self.inside_view._network is None:
            return
        
        self.titles = [self.inside_view._network.node[node]['title'] for node in self.inside_view._network]
        self.ids = self.inside_view._network.nodes()
        
        
    def update_view(self):
        self.inside_view.set_nhops(self._nhops)
        self.inside_view.set_edge_threshold(self._edge_threshold)
        self.inside_view.set_n_max_neighbors(self._n_max_neighbors)
        self.inside_view.set_algorithm(self._algorithm)
        self.inside_view.set_k(self._k_algorithm)
        self._selected_nodes = [self.ids[row] for row in self.selected_titles]
        self.inside_view.set_center_nodes(self._selected_nodes)
        self.inside_view.update_network()
        
   
        
    def filter_list(self):
        """Given a query for similar titles sets titles and ids"""
        
        str_input = self.filter
        #str_input = str_input.replace(' ', '').strip(' .').lower()
        #self.titles = sorted([self.inside_view._network.node[n]['title'] for n in self.inside_view._network.nodes() if str_input in self.inside_view._network.node[n]['title'].encode().replace(' ', '').strip(' .').lower()]) # [(id,title)]
        str_input = str_input.strip(' .').lower().split(' ')
        titles2 = [(n, str.lower(self.inside_view._network.node[n]['title'].encode('utf-8').strip(' .')).split(' ')) for n in self.inside_view._network.nodes()] # [(id,title)]
        titles2 = sorted(titles2, key = lambda x: sum(i in str_input for i in x[1]), reverse=True)
        self.titles = [self.inside_view._network.node[x[0]]['title'] for x in titles2]
        self.ids = [x[0] for x in titles2]
        
     