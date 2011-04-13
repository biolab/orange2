import network
import networkx.readwrite as rw

__all__ = ['generate_pajek', 'write_pajek', 'read_pajek', 'parse_pajek']

generate_pajek = rw.generate_pajek
write_pajek = rw.write_pajek

def read_pajek(path,encoding='UTF-8'):
    """Read graph in Pajek format from path. 

    Parameters
    ----------
    path : file or string
       File or filename to write.  
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : NetworkX MultiGraph or MultiDiGraph.

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> nx.write_pajek(G, "test.net")
    >>> G=nx.read_pajek("test.net")

    To create a Graph instead of a MultiGraph use

    >>> G1=nx.Graph(G)

    References
    ----------
    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
    for format information.
    """
    return rw.read_pajek(path, encoding)
    
def parse_pajek(lines):
    """Parse Pajek format graph from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in Pajek format.

    Returns
    -------
    G : NetworkX graph

    See Also
    --------
    read_pajek()

    """
    G = rw.parse_pajek(lines)
    
    if G.is_directed():
        G = network.DiGraph(G)
    else:
        G = network.Graph(G)
        
    return G