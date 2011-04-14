import network
import networkx.readwrite as rw
import warnings

__all__ = ['generate_pajek', 'write_pajek', 'read_pajek', 'parse_pajek']

try:
    generate_pajek = rw.generate_pajek
except:
    warnings.warn("Warning: your version of networkx does not contain the "+
                  "generate_pajek method; you may encounter problems when "+
                  "using the Orange.network module.")
    
write_pajek = rw.write_pajek

def read_pajek(path,encoding='UTF-8'):
    return rw.read_pajek(path, encoding)

def parse_pajek(lines):
    G = rw.parse_pajek(lines)
    
    if G.is_directed():
        G = network.DiGraph(G)
    else:
        G = network.Graph(G)
        
    return G

read_pajek.__doc__ = rw.read_pajek.__doc__
parse_pajek.__doc__ = rw.parse_pajek.__doc__
