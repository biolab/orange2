"""
<name>Net Analysis</name>
<description>Statistical analysis of network data.</description>
<icon>icons/NetworkFromDistances.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>6425</priority>
"""
import numpy
import networkx as nx

import Orange
import OWGUI

from OWWidget import *

NODELEVEL = 0
GRAPHLEVEL = 1

class WorkerThread(QThread):
    def __init__(self, receiver, name, label, type, algorithm):
        QThread.__init__(self)
        self.receiver = receiver
        self.name = name
        self.label = label
        self.type = type
        self.algorithm = algorithm
        
        self.stopped = 0
        self.result = None
        self.error = None
        self.is_terminated = False
         
    def run(self):
        try:
            self.result = self.algorithm(self.receiver.graph)
        except Exception as ex:
            self.result = None
            self.error = ex
        
class OWNxAnalysis(OWWidget):
    settingsList=[
        "auto_commit", "tab_index", "degree", "in_degree", "out_degree", "average_neighbor_degree",
        "clustering", "triangles", "square_clustering", "number_of_cliques",
        "degree_centrality", "in_degree_centrality", "out_degree_centrality", 
        "closeness_centrality", "betweenness_centrality", 
        "current_flow_closeness_centrality", "current_flow_betweenness_centrality",
        "approximate_current_flow_betweenness_centrality", 
        "eigenvector_centrality", "eigenvector_centrality_numpy", "load_centrality", 
        "core_number", "eccentricity", "closeness_vitality", 
        
        "number_of_nodes", "number_of_edges", "average_degree", "density",
        "degree_assortativity_coefficient", "degree_pearson_correlation_coefficient", 
        "degree_pearson_correlation_coefficient",
        "estrada_index", "graph_clique_number", "graph_number_of_cliques", 
        "transitivity", "average_clustering", "number_connected_components", 
        "number_strongly_connected_components", "number_weakly_connected_components", 
        "number_attracting_components", "diameter", "radius", "average_shortest_path_length"
    ]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Nx Analysis", noReport=True, wantMainArea=False)
        
        self.inputs = [("Network", Orange.network.Graph, self.set_graph), 
                        ("Items", Orange.data.Table, self.set_items)]
        
        self.outputs = [("Network", Orange.network.Graph), 
                        ("Items", Orange.data.Table)]
    
        self.methods = [
            ("number_of_nodes", True, "Number of nodes", GRAPHLEVEL, lambda G: G.number_of_nodes()),
            ("number_of_edges", True, "Number of edges", GRAPHLEVEL, lambda G: G.number_of_edges()),
            ("average_degree", True, "Average degree", GRAPHLEVEL, lambda G: numpy.average(G.degree().values())),
            ("diameter", False, "Diameter", GRAPHLEVEL, nx.diameter),
            ("radius", False, "Radius", GRAPHLEVEL, nx.radius),
            ("average_shortest_path_length", False, "Average shortest path length", GRAPHLEVEL, nx.average_shortest_path_length),
            ("density", True, "Density", GRAPHLEVEL, nx.density),
            ("degree_assortativity_coefficient", False, "Degree assortativity coefficient", GRAPHLEVEL, nx.degree_assortativity_coefficient),
            # additional attr needed
            #("attribute_assortativity_coefficient", False, "Attribute assortativity coefficient", GRAPHLEVEL, nx.attribute_assortativity_coefficient),
            #("numeric_assortativity_coefficient", False, "Numeric assortativity coefficient", GRAPHLEVEL, nx.numeric_assortativity_coefficient),
            ("degree_pearson_correlation_coefficient", False, "Degree pearson correlation coefficient", GRAPHLEVEL, nx.degree_pearson_correlation_coefficient),
            ("estrada_index", False, "Estrada index", GRAPHLEVEL, nx.estrada_index),
            ("graph_clique_number", False, "Graph clique number", GRAPHLEVEL, nx.graph_clique_number),
            ("graph_number_of_cliques", False, "Graph number of cliques", GRAPHLEVEL, nx.graph_number_of_cliques),
            ("transitivity", False, "Graph transitivity", GRAPHLEVEL, nx.transitivity),
            ("average_clustering", False, "Average clustering coefficient", GRAPHLEVEL, nx.average_clustering),
            ("number_connected_components", False, "Number of connected components", GRAPHLEVEL, nx.number_connected_components),
            ("number_strongly_connected_components", False, "Number of strongly connected components", GRAPHLEVEL, nx.number_strongly_connected_components),
            ("number_weakly_connected_components", False, "Number of weakly connected components", GRAPHLEVEL, nx.number_weakly_connected_components),
            ("number_attracting_components", False, "Number of attracting components", GRAPHLEVEL, nx.number_attracting_components),
            # TODO: input parameters
            #("max_flow", False, "Maximum flow", GRAPHLEVEL, nx.max_flow),
            #("min_cut", False, "Minimum cut", GRAPHLEVEL, nx.min_cut),
            #("ford_fulkerson", False, "Maximum single-commodity flow (Ford-Fulkerson)", GRAPHLEVEL, nx.ford_fulkerson),
            #("min_cost_flow_cost", False, "min_cost_flow_cost", GRAPHLEVEL, nx.min_cost_flow_cost),
            # returns dict of dict
            #("shortest_path_length", False, "Shortest path length", GRAPHLEVEL, nx.shortest_path_length),
            
            ("degree", False, "Degree", NODELEVEL, nx.degree),
            ("in_degree", False, "In-degree", NODELEVEL, lambda G: G.in_degree()),
            ("out_degree", False, "Out-degree", NODELEVEL, lambda G: G.out_degree()),
            ("average_neighbor_degree", False, "Average neighbor degree", NODELEVEL, nx.average_neighbor_degree),
            ("clustering", False, "Clustering coefficient", NODELEVEL, nx.clustering),
            ("triangles", False, "Number of triangles", NODELEVEL, nx.triangles),
            ("square_clustering", False, "Squares clustering coefficient", NODELEVEL, nx.square_clustering),
            ("number_of_cliques", False, "Number of cliques", NODELEVEL, nx.number_of_cliques),
            ("degree_centrality", False, "Degree centrality", NODELEVEL, nx.degree_centrality),
            ("in_degree_centrality", False, "In-egree centrality", NODELEVEL, nx.in_degree_centrality),
            ("out_degree_centrality", False, "Out-degree centrality", NODELEVEL, nx.out_degree_centrality),
            ("closeness_centrality", False, "Closeness centrality", NODELEVEL, nx.closeness_centrality),
            ("betweenness_centrality", False, "Betweenness centrality", NODELEVEL, nx.betweenness_centrality),
            ("current_flow_closeness_centrality", False, "Information centrality", NODELEVEL, nx.current_flow_closeness_centrality),
            ("current_flow_betweenness_centrality", False, "Random-walk betweenness centrality", NODELEVEL, nx.current_flow_betweenness_centrality),
            ("approximate_current_flow_betweenness_centrality", False, "Approx. random-walk betweenness centrality", NODELEVEL, nx.approximate_current_flow_betweenness_centrality),
            ("eigenvector_centrality", False, "Eigenvector centrality", NODELEVEL, nx.eigenvector_centrality),
            ("eigenvector_centrality_numpy", False, "Eigenvector centrality (NumPy)", NODELEVEL, nx.eigenvector_centrality_numpy),
            ("load_centrality", False, "Load centrality", NODELEVEL, nx.load_centrality),
            ("core_number", False, "Core number", NODELEVEL, nx.core_number),
            ("eccentricity", False, "Eccentricity", NODELEVEL, nx.eccentricity),
            ("closeness_vitality", False, "Closeness vitality", NODELEVEL, nx.closeness_vitality),                    
        ]
        
        self.auto_commit = False
        self.tab_index = 0
        self.mutex = QMutex()
        
        self.graph = None
        self.items = None          # items set by Items signal
        self.items_graph = None    # items set by graph.items by Network signal
        self.items_analysis = None # items to output and merge with analysis result
        
        self.job_queue = []
        self.job_working = []
        self.analfeatures = [] 
        self.analdata = {}
        
        for method in self.methods:
            setattr(self, method[0], method[1])
            setattr(self, "lbl_" + method[0], "")
    
        self.loadSettings()
        
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.tabs.setMinimumWidth(450)
        self.graphIndices = OWGUI.createTabPage(self.tabs, "Graph-level indices")    
        self.nodeIndices = OWGUI.createTabPage(self.tabs, "Node-level indices")
        self.tabs.setCurrentIndex(self.tab_index)
        self.connect(self.tabs, SIGNAL("currentChanged(int)"), lambda index: setattr(self, 'tab_index', index))
        
        for name, default, label, type, algorithm in self.methods:
            if type == NODELEVEL:
                box = OWGUI.widgetBox(self.nodeIndices, orientation="horizontal")
            elif type == GRAPHLEVEL:
                box = OWGUI.widgetBox(self.graphIndices, orientation="horizontal")
            
            OWGUI.checkBox(box, self, name, label=label, callback=lambda n=name: self.method_clicked(n))
            box.layout().addStretch(1)
            lbl = OWGUI.label(box, self, "%(lbl_" + name + ")s")
            setattr(self, "tool_" + name, lbl)
            
        self.graphIndices.layout().addStretch(1)
        self.nodeIndices.layout().addStretch(1)
        
        OWGUI.checkBox(self.controlArea, self, "auto_commit", label="Commit automatically")
        
        hb = OWGUI.widgetBox(self.controlArea, None, orientation='horizontal')
        self.btnCommit = OWGUI.button(hb, self, "Commit", callback=self.analyze, toggleButton=1)
        self.btnStopC = OWGUI.button(hb, self, "Stop current", callback=lambda current=True: self.stop_job(current))
        self.btnStopA = OWGUI.button(hb, self, "Stop all", callback=lambda current=False: self.stop_job(current))
        self.btnStopC.setEnabled(False)
        self.btnStopA.setEnabled(False)
        
        self.reportButton = OWGUI.button(hb, self, "&Report", self.reportAndFinish, debuggingEnabled=0)
        self.reportButton.setAutoDefault(0)
        
    def set_graph(self, graph):
        if graph is None:
            return
        
        self.stop_job(current=False)
        
        self.mutex.lock()
        
        self.graph = graph
        self.items_graph = graph.items()
        self.items_analysis = graph.items()
        
        if self.items is not None:
            self.items_analysis =  self.items
        
        self.clear_results()
        self.clear_labels()
        # only clear computed statistics on new graph
        self.analdata.clear()
        
        self.mutex.unlock()
        
        if self.auto_commit:
            self.btnCommit.setChecked(True)
            self.analyze()
        
        
    def set_items(self, items):
        self.mutex.lock()
        
        if items is None and self.items_graph is not None:
            self.items_analysis = self.items_graph
            
        elif items is not None:
            self.items_analysis = items
        
        self.items = items
        
        self.mutex.unlock()
        
    def analyze(self):
        if self.graph is None or not self.btnCommit.isChecked():
            return
        
        if len(self.job_queue) > 0 or len(self.job_working) > 0:
            return
        
        self.btnStopC.setEnabled(True)
        self.btnStopA.setEnabled(True)
        self.clear_labels()
        qApp.processEvents()
        
        self.clear_results()
        
        for method in self.methods:
            self.add_job(method)
            
        if len(self.job_queue) > 0:         
            self.start_job()
        else:
            self.send_data()
    
    def add_job(self, method):
        name, default, label, type, algorithm = method
        
        is_method_enabled = getattr(self, name)
            
        if not is_method_enabled:
            return
        
        #if type == NODELEVEL:
        job = WorkerThread(self, name, label, type, algorithm)
        self.connect(job, SIGNAL("terminated()"), lambda j=job: self.job_terminated(j))
        self.connect(job, SIGNAL("finished()"), lambda j=job: self.job_finished(j))
        self.job_queue.insert(0, job)
        setattr(self, "lbl_" + job.name, "   waiting")
    
    def start_job(self):
        max_jobs = QThread.idealThreadCount() - 1
        
        self.mutex.lock()
        if len(self.job_queue) > 0 and len(self.job_working) < max_jobs:
            job = self.job_queue.pop()
            setattr(self, "lbl_" + job.name, "   started")
            
            # if data for this job already computed
            if job.name in self.analdata:
                if job.type == NODELEVEL:
                    self.analfeatures.append((job.name, \
                                Orange.data.variable.Continuous(job.label)))
                    setattr(self, "lbl_" + job.name, "  finished")
                    
                elif job.type == GRAPHLEVEL:
                    setattr(self, "lbl_" + job.name,("%.4f" % \
                            self.analdata[job.name]).rstrip('0').rstrip('.'))                     
                
                job.quit()
                self.send_data()
            else:
                self.job_working.append(job)
                job.start()
        self.mutex.unlock()
        
        if len(self.job_queue) > 0 and len(self.job_working) < max_jobs:
            self.start_job()
            
    def job_terminated(self, job):
        self.mutex.lock()
        job.is_terminated = True
        self.mutex.unlock()
        
    def job_finished(self, job):
        self.mutex.lock()
        if job.is_terminated:
            setattr(self, "lbl_" + job.name, "terminated")
        else:
            setattr(self, "lbl_" + job.name, "  finished")
        
            if job.error is not None:
                setattr(self, "lbl_" + job.name, "     error")
                tooltop = getattr(self, "tool_" + job.name)
                tooltop.setToolTip(QString(job.error.message))
                
            elif job.result is not None:
                if job.type == NODELEVEL:
                    self.analfeatures.append((job.name, Orange.data.variable.Continuous(job.label)))
                    self.analdata[job.name] = [job.result[node] for node in sorted(job.result.iterkeys())]
                    
                elif job.type == GRAPHLEVEL:
                    self.analdata[job.name] = job.result
                    setattr(self, "lbl_" + job.name, ("%.4f" % job.result).rstrip('0').rstrip('.'))
        
        if job in self.job_working:
            self.job_working.remove(job)

        self.send_data()
        self.mutex.unlock()
        
        if len(self.job_queue) > 0:
            self.start_job()
            
    def stop_job(self, current=True, name=None):
        self.mutex.lock()
        
        if name is not None:
            for i in range(len(self.job_queue) - 1, -1, -1 ):
                job = self.job_queue[i]
                if name == job.name:
                    
                    job.is_terminated = True
                    job.quit()
                    job.wait()
                    self.job_queue.remove(job)
                    setattr(self, "lbl_" + name, "terminated")                            
                    
            for job in self.job_working:
                if name == job.name:
                    job.is_terminated = True
                    job.terminate()
        else:
            if not current:
                while len(self.job_queue) > 0:
                    job = self.job_queue.pop()
                    job.is_terminated = True
                    job.quit()
                    job.wait()
                    setattr(self, "lbl_" + job.name, "terminated")
    
            for job in self.job_working:
                job.is_terminated = True
                job.terminate()

        self.mutex.unlock()
        
    def send_data(self):
        if len(self.job_queue) <= 0 and len(self.job_working) <= 0:
            self.btnCommit.setChecked(False)
            self.btnStopC.setEnabled(False)
            self.btnStopA.setEnabled(False)
            
            if self.analdata is not None and len(self.analdata) > 0 and \
                                                    len(self.analfeatures) > 0:
                vars = [] 
                analdata = []
                for name, var in self.analfeatures:
                    analdata.append(self.analdata[name])
                    vars.append(var)
                    
                analitems  = Orange.data.Table(Orange.data.Domain(vars, False), [list(t) for t in zip(*analdata)])
                self.graph.set_items(Orange.data.Table([analitems, self.items_analysis]))
            
            self.send("Network", self.graph)
            self.send("Items", self.graph.items())
            
            self.clear_results()
            
    def method_clicked(self, name):
        if self.auto_commit:
            self.mutex.lock()
            if len(self.job_queue) <= 0 and len(self.job_working) <= 0:
                self.btnCommit.setChecked(True)
                self.mutex.unlock()
                self.analyze()
            else:
                is_method_enabled = getattr(self, name)
                if is_method_enabled:
                    for method in self.methods:
                        if name == method[0]:
                            self.add_job(method)
                    self.mutex.unlock()
                else:
                    self.mutex.unlock()
                    self.stop_job(name=name)
            
    def clear_results(self):
        del self.job_queue[:]
        del self.job_working[:]
        del self.analfeatures[:]
        
    def clear_labels(self):
        for method in self.methods:
            setattr(self, "lbl_" + method[0], "")
            
    def sendReport(self):
        report = []
        
        for name, default, label, type, algorithm in self.methods:
            if type == GRAPHLEVEL:
                value = getattr(self, "lbl_" + name)
                value = str(value).strip().lower()
                if value != "" and value != "error"  and value != "waiting" \
                            and value != "terminated" and value != "finished":
                    report.append((label, value))
        
        self.reportSettings("Graph statistics", report)
            
        
if __name__ == "__main__":    
    a=QApplication(sys.argv)
    ow=OWNxAnalysis()
    ow.show()
    def setNetwork(signal, data, id=None):
        if signal == 'Network':
            ow.set_graph(data)
        #if signal == 'Items':
        #    ow.set_items(data)
        
    import OWNxFile
    owFile = OWNxFile.OWNxFile()
    owFile.send = setNetwork
    owFile.show()
    owFile.selectNetFile(0)
    
    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()