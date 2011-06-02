import os.path
import warnings
import itertools

import networkx as nx
import networkx.readwrite as rw
from networkx.utils import _get_fh

import orangeom
import Orange
import Orange.network

__all__ = ['read', 'generate_pajek', 'write_pajek', 'read_pajek', 'parse_pajek']

def _wrap(g):
    for base, new in [(nx.DiGraph, Orange.network.DiGraph),
                      (nx.MultiGraph, Orange.network.MultiGraph),
                      (nx.MultiDiGraph, Orange.network.MultiDiGraph),
                      (nx.Graph, Orange.network.Graph)]:
        if isinstance(g, base):
            return g if isinstance(g, new) else new(g, name=g.name)
    return g

def read(path, encoding='UTF-8'):
    #supported = ['.net', '.gml', '.gpickle', '.gz', '.bz2', '.graphml']
    supported = ['.net', '.gml']
    
    if not os.path.isfile(path):
        return None
    
    root, ext = os.path.splitext(path)
    if not ext in supported:
        return None
    
    if ext == '.net':
        return read_pajek(path, encoding)
    
    if ext == '.gml':
        return read_gml(path, encoding)

def write(G, path, encoding='UTF-8'):
    #supported = ['.net', '.gml', '.gpickle', '.gz', '.bz2', '.graphml']
    supported = ['.net', '.gml']
    
    root, ext = os.path.splitext(path)
    if not ext in supported:
        raise ValueError('Extension %s is not supported. Use %s.' % (ext, ', '.join(supported)))
    
    if ext == '.net':
        write_pajek(G, path, encoding)
        
    if ext == '.gml':
        write_gml(G, path)
        
    if G.items() is not None:
        G.items().save(root + '_items.tab')
        
    if G.links() is not None:
        G.links().save(root + '_links.tab')

def read_pajek(path, encoding='UTF-8'):
    """ 
    Read Pajek file.
    """
    edges, arcs, items = orangeom.GraphLayout().readPajek(path)
    if len(arcs) > 0:
        # directed graph
        G = Orange.network.DiGraph()
        G.add_edges_from(((u,v,{'weight':d}) for u,v,d in edges))
        G.add_edges_from(((v,u,{'weight':d}) for u,v,d in edges))
        G.add_edges_from(((u,v,{'weight':d}) for u,v,d in arcs))
        G.set_items(items)
    else:
        G = Orange.network.Graph()
        G.add_edges_from(((u,v,{'weight':d}) for u,v,d in edges))
        G.set_items(items)
        
    return G
    #fh=_get_fh(path, 'rb')
    #lines = (line.decode(encoding) for line in fh)
    #return parse_pajek(lines)

def parse_pajek(lines):
    """
    Parse string in Pajek file format.
    """
    return read_pajek(lines)

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
            if k != 'x' and k != 'y':
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

def read_gml(path, encoding='latin-1', relabel=False):
    G = _wrap(rw.read_gml(path, encoding, relabel))
    return G

def write_gml(G, path):
    rw.write_gml(G, path)


#read_pajek.__doc__ = rw.read_pajek.__doc__
#parse_pajek.__doc__ = rw.parse_pajek.__doc__
