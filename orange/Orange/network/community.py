import random
import itertools

import Orange

def label_propagation(G, results2items=0, resultHistory2items=0):
    """Label propagation method from Raghavan et al., 2007
    
    :param results2items: append a new feature result to items 
        (Orange.data.Table)
    :type results2items: bool
    :param resultHistory2items: append new features result to items 
        (Orange.data.Table) after each iteration of the algorithm
    :type resultHistory2items: bool
    """
    
    vertices = range(G.number_of_nodes())
    labels = range(G.number_of_nodes())
    lblhistory = []
    #consecutiveStop = 0
    for i in range(1000):
        random.shuffle(vertices)
        stop = 1
        for v in vertices:
            nbh = G.neighbors(v)
            if len(nbh) == 0:
                continue
            
            lbls = [labels[u] for u in nbh]
            lbls = [(len(list(c)), l) for l, c in itertools.groupby(lbls)]
            m = max(lbls)[0]
            mlbls = [l for c, l in lbls if c >= m]
            lbl = random.choice(mlbls)
            
            if labels[v] not in mlbls: stop = 0
            labels[v] = lbl
            
        lblhistory.append([str(l) for l in labels])
        # if stopping condition might be satisfied, check it
        if stop:
            for v in vertices:
                nbh = G.neighbors(v)
                if len(nbh) == 0: continue
                lbls = [labels[u] for u in nbh]
                lbls = [(len(list(c)), l) for l, c \
                        in itertools.groupby(lbls)]
                m = max(lbls)[0]
                mlbls = [l for c, l in lbls if c >= m]
                if labels[v] not in mlbls: 
                    stop = 0
                    break
            if stop: break
                
    if results2items and not resultHistory2items:
        attrs = [Orange.data.variable.Discrete(
                                    'clustering label propagation',
                                    values=list(set([l for l \
                                                    in lblhistory[-1]])))]
        dom = Orange.data.Domain(attrs, 0)
        data = Orange.data.Table(dom, [[l] for l in lblhistory[-1]])
        if G.items() is None:
            G.set_items(data)  
        else: 
            G.set_items(Orange.data.Table([G.items(), data]))
    if resultHistory2items:
        attrs = [Orange.data.variable.Discrete('c'+ str(i),
            values=list(set([l for l in lblhistory[0]]))) for i,labels \
            in enumerate(lblhistory)]
        dom = Orange.data.Domain(attrs, 0)
        # transpose history
        data = map(list, zip(*lblhistory))
        data = Orange.data.Table(dom, data)
        if G.items() is None:
            G.set_items(data)  
        else: 
            G.set_items(Orange.data.Table([G.items(), data]))

    return labels