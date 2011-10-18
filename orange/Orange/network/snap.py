"""
.. index: Stanford Large Network Dataset Collection
.. index:
   single: network; Stanford Large Network Dataset Collection
   
*****************************************
Stanford Large Network Dataset Collection
*****************************************

This module includes classes to access the `Stanford Large Network Dataset 
Collection <http://snap.stanford.edu/data/>`_ by `Jure Leskovec 
<http://cs.stanford.edu/~jure/>`_.

.. autoclass:: Orange.network.snap.SNAP
   :members:

.. autoclass:: Orange.network.snap.NetworkInfo
   :members:

"""

import os
import os.path
import urllib
import httplib

import PyQt4.QtNetwork
import Orange.misc
import Orange.network

from HTMLParser import HTMLParser
from PyQt4.QtCore import QObject, QString, SIGNAL

class NetworkInfo(object):
    """The NetworkInfo class provides information about a network on the SNAP 
    web site.
    
    .. attribute:: name
    
        The name of the network.
    
    .. attribute:: link
    
        The url address of the network file.
    
    .. attribute:: type
    
        Network type (directed, undirected).
    
    .. attribute:: nodes
    
        Number of nodes in the network.
    
    .. attribute:: edges
    
        Number of edges in the network. 
    
    .. attribute:: repository
    
        The repository name (Social networks, Communication networks, ...).
    
    .. attribute:: description
    
        Detailed description of the network.
    
    """
    def __init__(self, name='', link='', type='', nodes='', edges='', repository='', description=''):
        self.name = name
        self.link = link
        self.type = type
        self.nodes = nodes
        self.edges = edges
        self.repository = repository
        self.description = description
        self._root =  Orange.misc.environ.buffer_dir + "/snap/"
        self._local_file = self._root + self.name + ".txt.gz"
        self._remote_file = "http://snap.stanford.edu/data/" + self.name + ".txt.gz"
        
    def read(self, progress_callback=None):
        """Read and return the network from file. Download the network to the 
        Orange home first if it was not jet downloaded. 
        
        :param progress_callback: a callback method to update a progress bar
        :type progress_callback: function(numblocks, blocksize, filesize)
        
        """
        
        if not self._is_downloaded():
            self._download(progress_callback)
        
        return Orange.network.readwrite.read(self._local_file)
    
    def _is_downloaded(self):
        if os.path.isfile(self._local_file):
            return True
        else:
            return False
        
    def _download(self, progress_callback=None):
        if not os.path.exists(self._root):
            os.makedirs(self._root)
            
        urllib.urlretrieve(self._remote_file, self._local_file, progress_callback)

class SNAPParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self) 
        
        self.h3 = False
        self.tr = False
        self.td = False
        self.title = ''
        self.table = False
        self.networks = []
        
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        
        if tag == 'h3':
            self.h3 = True
            
        if tag == 'table':
            self.table = True
            #self.networks = []
            
        if tag == 'tr':
            self.tr = True
            self.net_data = []
            
        if tag == 'td':
            self.td = True
            
        if tag == 'a' and self.td and self.tr:
            self.net_data.append(attrs.get('href', ''))
            
    def handle_data(self, data):
        if self.h3:
            self.title = data
            
        if self.tr and self.td:
            self.net_data.append(data)
            
    def handle_endtag(self, tag):
        if tag == 'h3':
            self.h3 = False
            
        if tag == 'table':
            self.table = False
            #self.repos[self.title] = self.networks
            
        if tag == 'tr':
            if len(self.net_data) == 6:
                self.networks.append(NetworkInfo(self.net_data[1], 
                     'http://snap.stanford.edu/data/' + self.net_data[0],
                     self.net_data[2],
                     self.net_data[3],
                     self.net_data[4],
                     self.title,
                     self.net_data[5]))
                
            self.tr = False
            
        if tag == 'td':
            self.td = False
    
class SNAP(object):
    """A collection of methods to access the information about networks in the 
    Stanford Large Network Dataset Collection.
    
    .. attribute:: network_list
    
        A list of networks on the `Stanford Large Network Dataset Collection web 
        site <http://snap.stanford.edu/data>`_. Each list item is an instance of the 
        :obj:`Orange.networks.snap.NetworkInfo` class.
    
    """
    
    def __init__(self):
        self.network_list = []
        self.http = None
        
    def parse_snap(self, error, done_callback, progress_callback=None):
        if not error:
            #self.last_total = self.http.bytesAvailable()
            #if self.last_total <= 0:
            #    self.last_total = 24763
            
            src = str(QString(self.http.readAll()))
            snap_parser = SNAPParser()
            snap_parser.feed(src)
            
            self.network_list = snap_parser.networks
            done_callback(snap_parser.networks)
        else:
            done_callback(None)
        
        if progress_callback is not None:
            progress_callback(1,1)
        
    def get_network_list(self, done_callback=None, progress_callback=None):
        """Read the networks from the SNAP web site and populate the n
        etwork_list attribute. If done_callback is set, an asynchronous HTTP 
        request is made to the SNAP web site. If the done_callback is left None, 
        the HTTP request made is synchronous and the network_list is returned.
        
        :param done_callback: a callback method called when the network info is downloaded
        :type done_callback: function(bool)
        
        :param progress_callback: a callback method to update a progress bar
        :type progress_callback: function(done, total)
        
        """
        if done_callback == None:  
            conn = httplib.HTTPConnection("snap.stanford.edu")
            conn.request("GET", "/data/index.html")
            r1 = conn.getresponse()
            src = r1.read()
            snap_parser = SNAPParser()
            snap_parser.feed(src)
            self.network_list = snap_parser.networks
            return self.network_list
        else:
            self.http = PyQt4.QtNetwork.QHttp()
            
            QObject.connect(self.http, SIGNAL('done(bool)'), lambda error,\
                    done_callback=done_callback, progress_callback=progress_callback: \
                    self.parse_snap(error, done_callback, progress_callback))
            
            if progress_callback is not None:
                QObject.connect(self.http, SIGNAL('dataReadProgress(int,int)'), progress_callback)
            
            self.http.setHost('snap.stanford.edu')
            self.http.get('/data/index.html')
            
    def get_network(self, id):
        """Find and return the network by name. If no network is found, return 
        None. Call get_network_list before calling this method to populate the 
        network_list attribute.
        
        :param id: a name of the network in SNAP collection
        :type id: string
        
        """
        
        for network in self.network_list:
            if str(network.name) == str(id):
                return network
            
        return None