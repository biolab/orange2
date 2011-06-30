import Orange
import networkx as nx
import networkx.readwrite as rw
import warnings
import itertools
import Orange.network

from networkx.utils import _get_fh

__all__ = ['generate_pajek', 'write_pajek', 'read_pajek', 'parse_pajek']

def _wrap(g):
    for base, new in [(nx.DiGraph, Orange.network.DiGraph),
                      (nx.MultiGraph, Orange.network.MultiGraph),
                      (nx.MultiDiGraph, Orange.network.MultiDiGraph),
                      (nx.Graph, Orange.network.Graph)]:
        if isinstance(g, base):
            return g if isinstance(g, new) else new(g, name=g.name)
    return g

def read_pajek(path,encoding='UTF-8'):
    """
    A copy&paste of networkx's function. Calls the local parse_pajek().
    """
    fh=_get_fh(path, 'rb')
    lines = (line.decode(encoding) for line in fh)
    return parse_pajek(lines)

def parse_pajek(lines):
    """
    A copy&paste of networkx's function with some bugs fixed:
      - make it a Graph or DiGraph if there is no reason to have a Multi*,
      - do not lose graph's name during its conversion.
    """
    import shlex
    from networkx.utils import is_string_like
    multigraph=False
    if is_string_like(lines): lines=iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines])
    G=nx.MultiDiGraph() # are multiedges allowed in Pajek? assume yes
    directed=True # assume this is a directed network for now
    while lines:
        try:
            l=next(lines)
        except: #EOF
            break
        if l.lower().startswith("*network"):
            label,name=l.split(None, 1)
            G.name=name
        if l.lower().startswith("*vertices"):
            nodelabels={}
            l,nnodes=l.split()
            for i in range(int(nnodes)):
                splitline=shlex.split(str(next(lines)))
                id,label=splitline[0:2]
                G.add_node(label)
                nodelabels[id]=label
                G.node[label]={'id':id}
                try: 
                    x,y,shape=splitline[2:5]
                    G.node[label].update({'x':float(x),
                                          'y':float(y),
                                          'shape':shape})
                except:
                    pass
                extra_attr=zip(splitline[5::2],splitline[6::2])
                G.node[label].update(extra_attr)
        if l.lower().startswith("*edges") or l.lower().startswith("*arcs"):
            if l.lower().startswith("*edge"):
               # switch from multi digraph to multi graph
                G=nx.MultiGraph(G, name=G.name)
            for l in lines:
                splitline=shlex.split(str(l))
                ui,vi=splitline[0:2]
                u=nodelabels.get(ui,ui)
                v=nodelabels.get(vi,vi)
                # parse the data attached to this edge and put in a dictionary 
                edge_data={}
                try:
                    # there should always be a single value on the edge?
                    w=splitline[2:3]
                    edge_data.update({'weight':float(w[0])})
                except:
                    pass
                    # if there isn't, just assign a 1
#                    edge_data.update({'value':1})
                extra_attr=zip(splitline[3::2],splitline[4::2])
                edge_data.update(extra_attr)
                if G.has_edge(u,v):
                    multigraph=True
                G.add_edge(u,v,**edge_data)

    if not multigraph: # use Graph/DiGraph if no parallel edges
        if G.is_directed():
            G=nx.DiGraph(G, name=G.name)
        else:
            G=nx.Graph(G, name=G.name)
    return _wrap(G)

def generate_pajek(G):
    """
    A copy&paste of networkx's function with some bugs fixed:
     - generate one line per object (vertex, edge, arc); do not add one per
       entry in data dictionary.
    """
    from networkx.utils import make_str, is_string_like

    if G.name=='': 
        name='NetworkX'
    else:
        name=G.name
    yield '*network %s'%name

    # write nodes with attributes
    yield '*vertices %s'%(G.order())
    nodes = G.nodes()
    # make dictionary mapping nodes to integers
    nodenumber=dict(zip(nodes,range(1,len(nodes)+1))) 
    for n in nodes:
        na=G.node.get(n,{})
        x=na.get('x',0.0)
        y=na.get('y',0.0)
        id=int(na.get('id',nodenumber[n]))
        nodenumber[n]=id
        shape=na.get('shape','ellipse')
        s = ' '.join(map(make_str,(id,n,x,y,shape)))
        for k,v in na.items():
            s += ' %s %s'%(k,v)
        yield s

    # write edges with attributes         
    if G.is_directed():
        yield '*arcs'
    else:
        yield '*edges'
    for u,v,edgedata in G.edges(data=True):
        d=edgedata.copy()
        value=d.pop('weight',1.0) # use 1 as default edge value
        s = ' '.join(map(make_str,(nodenumber[u],nodenumber[v],value)))
        for k,v in d.items():
            if not is_string_like(v):
                v = repr(v)
            # add quotes to any values with a blank space
            if " " in v: 
                v="\"%s\"" % v.replace('"', r'\"')
            s += ' %s %s'%(k,v)
        yield s
        
def write_pajek(G, path, encoding='UTF-8'):
    """
    A copy&paste of networkx's function with some bugs fixed:
     - call the new generate_pajek.
    """
    fh=_get_fh(path, 'wb')
    for line in generate_pajek(G):
        line+='\n'
        fh.write(line.encode(encoding))

def parse_pajek_project(lines):
    network_lines = []
    result = []
    for i, line in enumerate(itertools.chain(lines, ["*"])):
        line_low = line.strip().lower()
        if not line_low:
            continue
        if line_low[0] == "*" and not any(line_low.startswith(x)
                                          for x in ["*vertices", "*arcs", "*edges"]):
            if network_lines != []:
                result.append(parse_pajek(network_lines))
                network_lines = []
        if line_low.startswith("*network") or network_lines != []:
            network_lines.append(line)
    return result

def read_pajek_project(path, encoding='UTF-8'):
    fh = _get_fh(path, 'rb')
    lines = (line.decode(encoding) for line in fh)
    return parse_pajek_project(lines)

#read_pajek.__doc__ = rw.read_pajek.__doc__
#parse_pajek.__doc__ = rw.parse_pajek.__doc__
