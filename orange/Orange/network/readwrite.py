""" 
.. index:: reading and writing networks

.. index::
   single: network; reading and writing networks

****************************
Reading and writing networks
****************************

When using networks in Orange data mining suite, I advise you not to use 
NetworkX reading and writing methods.  Instead, use new methods provided in
the :obj:`Orange.network.readwrite` module. If, for some reason, you have to 
use the original read / write methods, do not forget to cast the network (see 
Orange.network.readwrite._wrap method).  

"""

import os.path
import warnings
import itertools

import networkx as nx
import networkx.readwrite.pajek as rwpajek
import networkx.readwrite.gml as rwgml
import networkx.readwrite.gpickle as rwgpickle

import Orange
import Orange.network
import orangeom

__all__ = ['read', 'write', 'read_gpickle', 'write_gpickle', 'read_pajek', 
           'write_pajek', 'parse_pajek', 'generate_pajek', 'read_gml', 
           'write_gml']

def _wrap(g):
    for base, new in [(nx.DiGraph, Orange.network.DiGraph),
                      (nx.MultiGraph, Orange.network.MultiGraph),
                      (nx.MultiDiGraph, Orange.network.MultiDiGraph),
                      (nx.Graph, Orange.network.Graph)]:
        if isinstance(g, base):
            return g if isinstance(g, new) else new(g, name=g.name)
    return g

def _add_doc(myclass, nxclass):
    tmp = nxclass.__doc__.replace('nx.write', 'Orange.network.readwrite.write')
    tmp = tmp.replace('nx.read', 'Orange.network.readwrite.read')
    tmp = tmp.replace('nx', 'Orange.network.nx')
    myclass.__doc__ += tmp 

def _is_string_like(obj): # from John Hunter, types-free version
    """Check if obj is string."""
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def _get_fh(path, mode='r'):
    """ Return a file handle for given path.

    Path can be a string or a file handle.

    Attempt to uncompress/compress files ending in '.gz' and '.bz2'.

    """
    if _is_string_like(path):
        if path.endswith('.gz'):
            import gzip
            fh = gzip.open(path,mode=mode)
        elif path.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(path,mode=mode)
        else:
            fh = open(path,mode = mode)           
    elif hasattr(path, 'read'):
        fh = path
    else:
        raise ValueError('path must be a string or file handle')
    return fh

def _make_str(t):
    """Return the string representation of t."""
    if _is_string_like(t): return t
    return str(t)

def read(path, encoding='UTF-8'):
    """Read graph in any of the supported file formats (.gpickle, .net, .gml).
    The parser is chosen based on the file extension.
    
    :param path: File or filename to write.
    :type path: string

    Return the network of type :obj:`Orange.network.Graph`, 
    :obj:`Orange.network.DiGraph`, :obj:`Orange.network.Graph` or 
    :obj:`Orange.network.DiGraph`.
    
    """
    
    #supported = ['.net', '.gml', '.gpickle', '.gz', '.bz2', '.graphml']
    supported = ['.net', '.gml', '.gpickle']
    
    if not os.path.isfile(path):
        raise OSError('File %s does not exist.' % path)
    
    root, ext = os.path.splitext(path)
    if not ext in supported:
        raise ValueError('Extension %s is not supported.' % ext)
    
    if ext == '.net':
        return read_pajek(path, encoding)
    
    if ext == '.gml':
        return read_gml(path, encoding)
    
    if ext == '.gpickle':
        return read_gpickle(path)

def write(G, path, encoding='UTF-8'):
    """Write graph in any of the supported file formats (.gpickle, .net, .gml).
    The file format is chosen based on the file extension.
    
    :param G: A Orange graph.
    :type G: Orange.network.Graph
         
    :param path: File or filename to write.
    :type path: string
     
    """
    
    #supported = ['.net', '.gml', '.gpickle', '.gz', '.bz2', '.graphml']
    supported = ['.net', '.gml', '.gpickle']
    
    root, ext = os.path.splitext(path)
    if not ext in supported:
        raise ValueError('Extension %s is not supported. Use %s.' % (ext, ', '.join(supported)))
    
    if ext == '.net':
        write_pajek(G, path, encoding)
        
    if ext == '.gml':
        write_gml(G, path)
        
    if ext == '.gpickle':
        write_gpickle(G, path)
        
    if G.items() is not None:
        G.items().save(root + '_items.tab')
        
    if G.links() is not None:
        G.links().save(root + '_links.tab')

def read_gpickle(path):
    """NetworkX read_gpickle method and wrap graph to Orange network.
    
    """
    
    return _wrap(rwgpickle.read_gpickle(path))

_add_doc(read_gpickle, rwgpickle.read_gpickle)

def write_gpickle(G, path):
    """NetworkX write_gpickle method.
    
    """
    
    rwgpickle.write_gpickle(G, path)

_add_doc(write_gpickle, rwgpickle.write_gpickle)

def read_pajek(path, encoding='UTF-8', project=False):
    """A completely reimplemented method for reading Pajek files. Written in 
    C++ for maximum performance.  
    
    :param path: File or filename to write.
    :type path: string
    
    :param encoding: Encoding of input text file, default 'UTF-8'.
    :type encoding: string
    
    :param project: Determines whether the input file is a Pajek project file,
        possibly containing multiple networks and other data. If :obj:`True`,
        a list of networks is returned instead of just a network. Default is
        :obj:`False`.
    :type project: boolean.
        
    Return the network (or a list of networks if project=:obj:`True`) of type
    :obj:`Orange.network.Graph` or :obj:`Orange.network.DiGraph`.


    Examples

    >>> G=Orange.network.nx.path_graph(4)
    >>> Orange.network.readwrite.write_pajek(G, "test.net")
    >>> G=Orange.network.readwrite.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1=Orange.network.Graph(G)

    References

    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    
    """
    
    input = orangeom.GraphLayout().readPajek(path, project)
    result = []
    for g in input if project else [input]:
        graphname, vertices, edges, arcs, items = g
        if len(arcs) > 0:
            # directed graph
            G = Orange.network.DiGraph()
            G.add_nodes_from(range(len(items)))
            G.add_edges_from(((u,v,dict(d.items()+[('weight',w)])) for u,v,w,d in edges))
            G.add_edges_from(((v,u,dict(d.items()+[('weight',w)])) for u,v,w,d in edges))
            G.add_edges_from(((u,v,dict(d.items()+[('weight',w)])) for u,v,w,d in arcs))
            G.set_items(items)
        else:
            G = Orange.network.Graph()
            G.add_nodes_from(range(len(items)))
            G.add_edges_from(((u,v,dict(d.items()+[('weight',w)])) for u,v,w,d in edges))
            G.set_items(items)
        for i, vdata in zip(range(len(G.node)), vertices):
            G.node[i].update(vdata)
        G.name = graphname
        
        result.append(G)
        
    if not project:
        result = result[0]
        
    return result
    #fh=_get_fh(path, 'rb')
    #lines = (line.decode(encoding) for line in fh)
    #return parse_pajek(lines)

def write_pajek(G, path, encoding='UTF-8'):
    """A copy & paste of NetworkX's function with some bugs fixed (call the new 
    generate_pajek).
    
    """
    
    fh=_get_fh(path, 'wb')
    for line in generate_pajek(G):
        line+='\n'
        fh.write(line.encode(encoding))

_add_doc(write_pajek, rwpajek.write_pajek)

def parse_pajek(lines):
    """Parse string in Pajek file format. See read_pajek for usage examples.
    
    :param lines: a string of network data in Pajek file format.
    :type lines: string
    
    """
    
    return read_pajek(lines)


def generate_pajek(G):
    """A copy & paste of NetworkX's function with some bugs fixed (generate 
    one line per object: vertex, edge, arc. Do not add one per entry in data 
    dictionary).
    
    Generate lines in Pajek graph format.
    
    :param G: A Orange graph.
    :type G: Orange.network.Graph

    References
    
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    
    """
    
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
        s = ' '.join(map(_make_str,(id,n,x,y,shape)))
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
        s = ' '.join(map(_make_str,(nodenumber[u],nodenumber[v],value)))
        for k,v in d.items():
            if not _is_string_like(v):
                v = repr(v)
            # add quotes to any values with a blank space
            if " " in v: 
                v="\"%s\"" % v.replace('"', r'\"')
            s += ' %s %s'%(k,v)
        yield s
        

#_add_doc(generate_pajek, rwpajek.generate_pajek)
        
def read_gml(path, encoding='latin-1', relabel=False):
    """NetworkX read_gml method and wrap graph to Orange network.
    
    """
    
    return _wrap(rwgml.read_gml(path, encoding, relabel))

_add_doc(read_gml, rwgml.read_gml)

def write_gml(G, path):
    """NetworkX write_gml method.
    
    """
    
    rwgml.write_gml(G, path)

_add_doc(write_gml, rwgml.write_gml)
