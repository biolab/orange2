import sys, pprint, time, re
from itertools import *
import urllib
import urllib2
import orange
import socket
import os
from collections import defaultdict
import orngServerFiles
import pickle

defaddress = "http://bcm.fri.uni-lj.si/microarray/api/index.php?"
#defaddresspipa = "https://pipa.fri.uni-lj.si/pipa/script/api/orange.py?action="
defaddresspipa = "https://pipa.fri.uni-lj.si/PIPA/PIPAapi/PIPAorange.py"

pipaparuser = "pipa_username"
pipaparpass = "pipa_password"

#utility functions - from Marko's mMisc.py

def splitN(origsize, maxchunk):
    """
    Splits an integer into chunks of given size. Each created chunk
    except possibly the last one is of maximum allowed size.
    Chunks are returned in list.
    """
    l = [maxchunk]*(origsize/maxchunk)
    a = origsize % maxchunk
    if a > 0: 
        l.append(a)
    return l

def split(l, maxchunk):
    """
    Splits list l into chunks of size maxchunk. Each created chunk
    except possibly the last one is of maximum allowed size.
    """
    sizes = splitN(len(l), maxchunk)
    out = []
    tillNow = 0
    for s in sizes:
        out.append(l[tillNow:tillNow+s])
        tillNow += s
    return out       

def lloc(l,n):
    """
    List location in list of list structure.
    Enable the use of negative locations:
    -1 is the last element, -2 second last...
    """
    if n < 0:
        return len(l[0])+n
    else:
        return n

def loc(l,n):
    """
    List location.
    Enable the use of negative locations:
    -1 is the last element, -2 second last...
    """
    if n < 0:
        return len(l)+n
    else:
        return n

def nth(l,n):
    """
    Returns only nth elemnt in a list.
    """
    n = lloc(l,n)
    return [ a[n] for a in l ]

def imnth(l, ns):
    """
    Return only columns as specified in ns. Returns an generator.
    """
    ns = [ lloc(l,n) for n in ns ]
    for a in l:
        yield [ a[n] for n in ns ]

def flatten(l,r=0):
    """
    Flatten a python structure into a list. Leave strings alone.
    """
    if type(l) == type("a"):
        return [ l ]
    try: #if enumerable then flatten it's elements
        rec = [ flatten(a,r=r+1) for a in l ]
        ret = []
        for a in rec:
            ret = ret + a
        return ret
    except:
        return [ l ]

def mxrange(lr):
    """
    Multiple xranges. Can be used to traverse matrices.
    This function is very slow due to unknown number of
    parameters.

    >>> mxrange([3,5]) 
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    >>> mxrange([[3,5,1],[9,0,-3]])
    [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
    """
    if len(lr) == 0:
        yield ()
    else:
        #it can work with single numbers
        index = lr[0]
        if type(1) == type(index):
            index = [ index ]
        for a in range(*index):
            for b in mxrange(lr[1:]):
                yield tuple([a] + list(b))


def issequencens(x):
    """
    Is x a sequence and not string ? We say it is if it has a __getitem__ 
    method and it is not an instance of basestring.
    """
    return hasattr(x, '__getitem__') and not isinstance(x, basestring)

#end utility functions

socket.setdefaulttimeout(60)

verbose = 0

def median(l):
    if len(l) == 0:
        return None
    l = sorted(l)
    if len(l) % 2 == 1: #odd
        return l[len(l)/2]
    else: #even
        return (l[len(l)/2-1] + l[len(l)/2])/2.0

class HttpGetException(Exception): pass

def replaceChars(address):
    return address.replace(" ", "%20")

def httpGet(address, *args, **kwargs):
    if verbose: 
        print address, args, kwargs,  " "
    address = replaceChars(address)
    t1 = time.time()
    f = urllib2.urlopen(address, *args, **kwargs)
    read = f.read()
    if verbose:
        print "bytes", len(read),
    if verbose:
        print time.time() - t1
    return read

def txt2ll(s, separ=' ', lineSepar='\n'):
    return [ a.split(separ) for a in s.split(lineSepar) ]

class AuthenticationError(Exception):
    pass

class DBInterface(object):
 
    def __init__(self, address):
        self.address = address

    def raw(self, request, data=None, tryN=3):
        if verbose:
            print "tryN", tryN

        if tryN == 0:
            return None
        try:
            if data == None:
                return httpGet(self.address + request)
            else:
                return httpGet(self.address + request, data=urllib.urlencode(data))
        except IOError:
            return self.raw(request, data=data, tryN=tryN-1)

    def get(self, request, data=None, tryN=3):
        rawf = self.raw(request, data)
        if rawf == None:
            raise Exception("Connection error when contacting " + self.address + request)
        if rawf.startswith("error: authentication failed"):
            raise AuthenticationError()
        elif rawf[:1] == "<" or rawf[:5] == "error" or rawf.startswith("MOD_PYTHON ERROR"): #an error occurred - starting some html input
            #TODO are there any other kinds of errors?
            if tryN > 0:
                if verbose:
                    print "trying again"
                return self.get(request, data=data, tryN=tryN-1)
            else:
                if verbose:
                    print rafw[:1000]
                raise Exception("Error with the database")

        a = txt2ll(rawf, separ='\t')
        
        if a[-1][0] == "": #remove empty line on end
            a = a[:-1]
        return a


def _test():
    import doctest
    doctest.testmod()

def splitTableOnColumn(ll,n):
    omap = {}
    n = lloc(ll,n)
    for l in ll:
        cell = omap.get(l[n], [])
        cell.append(l)
        omap[l[n]] = cell
    return omap

def neededColumns(legend, want):
    return [ legend.index(a) for a in want ]

def onlyColumns(ll, legend, want):
    return list(imnth(ll, neededColumns(legend, want)))

def ll2dic(ll, key=0, value=1):
    """
    Converts LL to map. Key is key position, value value position
    """
    names = nth(ll, lloc(ll, key))
    if len(names) == len(set(names)):
        return dict(list(imnth(ll, [ lloc(ll, key),  lloc(ll, value) ] )))
    else:
        raise Exception("all keys are not unique")

def dic2ll(dic):

    columns = sorted(dic.values()[0].keys())
    ll = []
    for key, d in dic.items():
        ll.append([ key ] + [ d[a] for a in columns ])
    return columns,ll

def allUnique(els):
    return len(els) == len(set(els))

def reverseDic(d):
    """
    Create a reverse dictionary if a unique reverse is possible.
    """
    if allUnique(d.values()):
        return dict([ (b,a) for a,b in d.items() ])

def chainLookup(a, dics, force=[]):
    """
    Goes through a list of dictionaries. On each step try to
    transform current key to value. The values becomes the key
    for next search If unsuccessfull, query the next
    dictionary with current key. Force last translation.
    """
    force = [ loc(dics, i) for i in force ]
    for i,dic in enumerate(dics):
        if i in force: #force specified translations
            a = dic[a]
        else:
            if a in dic:
                a = dic[a]
    return a

class DBCommon(object):

    def fromBuffer(self, addr):
        return self.buffer.get(self.address + addr)

    def toBuffer(self, addr, cont, version, autocommit=True):
        if self.buffer:
            return self.buffer.add(self.address + addr, cont, version=version, autocommit=autocommit)

    def bufferCommit(self):
        if self.buffer:
            self.buffer.commit()

    def bufferFun(self, bufkey, bufver, reload, fn, *args, **kwargs):
        """
        If bufkey is already present in buffer, return its contents.
        If not, run function with arguments and save its result
        into the buffer.
        """
        if self.inBuffer(bufkey) == bufver and reload == False:
            res = self.fromBuffer(bufkey)
        else:
            res = fn(*args, **kwargs)
            self.toBuffer(bufkey, res, bufver)
        return res

    def sq(self, s1, data=None, buffer=True, bufadd="", bufname=None, bufver="0", reload=False, bufferkey=None):
        if buffer:
            bufkey = bufadd + (bufname if bufname != None else s1)
            if bufferkey != None:
                bufkey = bufferkey(bufkey, data)
            res = self.bufferFun(bufkey, bufver, reload, self.db.get, s1, data=data)
        else:
            res = self.db.get(s1, data=data)
        return res[1:],res[0]

    def inBuffer(self, addr):
        if self.buffer:
            return self.buffer.contains(self.address + addr)
        else:
            return False

    def dictionarize(self, ids, fn, *args, **kwargs):
        """
        Creates a dictionary from id: function result.
        Callback for each part done.
        """
        callback = kwargs.pop("callback", None)
        odic = {}
        for a,b in izip(ids, fn(*args, **kwargs)):
            odic[a] = b
            if callback: callback()
        return odic

    def downloadMulti_bufcommand_replace_multi(self, command, data=None, chunk=100, bufferkey=None, transformfn=None):
        """
        Get function which gives buffer address for an id and a function 
        which replaces $MULTI$.
        """

        def bufferkey1(command, data):
            if transformfn:
                return "TRANS " + command
            else:
                return command

        def replace_multi(command, data, repl):
            return command.replace("$MULTI$", repl),\
                dict((a,b.replace("$MULTI$", repl)) for a,b in sorted(data.items())) if data != None else None

        if bufferkey == None:
            bufferkey=bufferkey1

        bufcommand = lambda x, c=command, d=data: bufferkey(*replace_multi(c, d, x))
        return bufcommand, replace_multi

    def downloadMulti(self, command, ids, data=None, chunk=100, transformfn=None, bufferkey=None, separatefn=None, bufreload=False, bufver="0"):
        """
        Downloads multiple results at once.
        Results in the same order as in ids.

        Bufferkey transforms command and data into buffer key.
        bufver is a function returning buffer version for a given id. if
            a string is given, use it for all ids
        """

        sids = split(ids,chunk)
    
        bufverfn = None
        if isinstance(bufver, basestring):
            bufverfn = lambda x: bufver
        else:
            bufverfn = bufver

        bufcommand, replace_multi = self.downloadMulti_bufcommand_replace_multi(command, data=data, chunk=chunk, bufferkey=bufferkey, transformfn=transformfn)

        for i,sidp in enumerate(sids):

            buffered = []
            unbuffered = []
        
            for a in sidp:
                if self.inBuffer(bufcommand(a)) == bufverfn(a) and bufreload == False:
                    buffered.append(a)
                else:
                    unbuffered.append(a)

            res = []
            legend = []

            if len(unbuffered) > 0:
                com1, d1 = replace_multi(command, data, ",".join(unbuffered))
                res, legend = self.sq(com1, data=d1, buffer=False) #get unbuffered part
            else:
                # get legend from buffer also
                legend = self.fromBuffer(bufcommand(buffered[0]))[0]

            #split on different values of the first column - first attribute

            if not separatefn:
                antss = splitTableOnColumn(res, 0)
            else:
                legend, antss = separatefn([legend]+res)

            #if transform before saving is requested, do it
            if transformfn:
                nantss = {}
                nlegend = None
                for a,b in antss.items():
                    nb, nlegend = transformfn(b, legend)
                    nantss[a] = nb
                legend = nlegend
                antss = nantss
 
            #here save buffer
            for a,b in antss.items():
                self.toBuffer(bufcommand(a), [ legend ] + b, bufverfn(a), autocommit=False)
            self.bufferCommit()
            

            #get buffered from the buffer
            antssb = dict([ (b, self.fromBuffer(bufcommand(b))[1:]) for b in buffered ])
            antss.update(antssb)

            #put results in order
            tl = []
            for ci in sidp:
                yield antss[ci], legend

    def exampleTables(self, ids, chipsm=None, spotmap={}, callback=None, exclude_constant_labels=False, annots={}, chipfn=None, allowed_labels=None):
        """
        Create example tables from chip readings, spot mappings and 
        group specifications.

        group is the output from "sortAnnotations" function. 
        spotmap is a dictionary of { spotid: gene }
        chipsm is a dictionary of chip readings

        Callback: number of chipids + 2
        """

        if verbose:
            print "Creating example table"

        if callback: callback()

        amap = {}
        amapnext = 0

        togen = []

        groupnames = []
        groupvals = []
        groupannots = []

        if chipsm == None:
            chipdl = chipfn(ids)

        for chipid in ids:

            if chipsm != None:
                chipdata = chipsm[chipid]
            else:
                chipdata = chipdl.next()

            if callback: callback()

            #add to current position mapping
            repeats = {}
            for id,_ in chipdata:
                rep = repeats.get(id, 0)
                repeats[id] = rep+1
                key = (id, rep)
                if key not in amap:
                    amap[key] = amapnext
                    amapnext += 1

            vals = [ None ] * len(amap)

            repeats = {}
            for id,v in chipdata:
                rep = repeats.get(id, 0)
                repeats[id] = rep+1
                key = (id, rep)
                putind = amap[key]
                vals[putind] = v
            groupvals.append(vals)

            groupnames.append(chipid) 

            newannots = [['id', str(chipid)]] #add chipid to annotations
            if annots:
                newannots += annots[chipid]
            groupannots.append(newannots)

        togen = (groupnames, groupvals, groupannots)
        
        if callback: callback()

        ddb = [ None ]*len(amap)
        for (a,rep),pos in amap.items():
            if len(spotmap):
                ddb[pos] = spotmap.get(a, "#"+a)
            else:
                ddb[pos] = a

        #this is sorted position mapping: key -> sortedind
        posMap = dict( (k,i) for i,k in enumerate(sorted(amap.keys())) )
        revmap = dict( ( (i,k) for k,i in amap.items() ) )
        #permutation[i] holds target of current [i]
        permutation = [ posMap[revmap[i]] for i in range(len(amap)) ]

        def enlength(a, tlen):
            """ Adds Nones to the end of the list """
            if len(a) < tlen:
                return a + [ "None" ]*(tlen-len(a))
            else:
                return a

        def enlengthl(l, tlen):
            return [ enlength(a, tlen) for a in l ]
    
        groupnames, groupvals, groupannots = togen

        et = createExampleTable(groupnames, 
            enlengthl(groupvals, len(ddb)),
            groupannots, ddb, exclude_constant_labels=exclude_constant_labels, permutation=permutation, allowed_labels=allowed_labels)

        if callback: callback()

        return et

def bufferkeypipa(command, data):
    """ Do not save password to the buffer! """
    command = command + " v5" #add version
    if data != None:
        data = data.copy()
        if pipaparpass in data:
            data.pop(pipaparpass)
        return command + " " +  urllib.urlencode(sorted(data.items()))
    else:
        return command

class PIPA(DBCommon):

    def __init__(self, address=defaddresspipa, buffer=None, username=None, password=None):
        self.address = address
        self.db=DBInterface(address)
        self.buffer = buffer
        self.username = None
        if username != None:
            self.username = username
            self.password = password

    def add_auth(self, data=None):
        if self.username == None:
            return data
        authdic = { pipaparuser: self.username, pipaparpass: self.password }
        if data != None:
            authdic.update(data)
        return authdic

    def annotations(self, reload=False, bufver="0"):
        """
        Returns a dictionary of (id: dictionary of annotations).
        """
        #res, legend = self.sq("mapping_list", data=self.add_auth(), reload=reload, bufferkey=bufferkeypipa, bufver=bufver)
        res, legend = self.sq("", data=self.add_auth({"action": "expression_list"}), reload=reload, bufferkey=bufferkeypipa, bufver=bufver)
        return dict( (sa[0], dict(zip(legend[1:], sa[1:]))) for sa in res )

    def chips(self, ids, ctype, reload=False, bufver="0"):
        def separatefn(res):
            #each one is own rown
            #genes are in the first row
            #remove unknown
            genes = res[0][1:]
            cids = nth(res,0)[1:]

            antss = {}
            for i,cid in enumerate(cids):
                row = i+1
                vals = res[row][1:]
                antss[cid] = [ list(a) for a in zip(genes, vals) if a[1] != "?" ]
            return ['gene_id', 'value'], antss

        download_command = "gene_expression"
        datadict = {"action": download_command, "ids":"$MULTI$", 'transpose':'1'}
        if ctype != None:
            datadict["type"] = ctype
        antss = self.downloadMulti("", ids, data=self.add_auth(datadict), chunk=10, separatefn=separatefn, bufferkey=bufferkeypipa, bufreload=reload, bufver=bufver)
        for a,legend in antss:
            yield a

    def gene_expression_types(self, reload=False, bufver="0"):
        #res, legend = self.sq("gene_expression_type", data={self.add_auth(), reload=reload, bufferkey=bufferkeypipa, bufver=bufver)
        res, legend = self.sq("", data=self.add_auth({"action":"gene_expression_type"}), reload=reload, bufferkey=bufferkeypipa, bufver=bufver)
        return sorted(tuple(a) for a in res)

    def chips_keynaming(self, ctype):
        keynamingfn,_ = self.downloadMulti_bufcommand_replace_multi("", data=self.add_auth({"action": "gene_expression", "ids":"$MULTI$", 'transpose':'1', 'type':ctype}), chunk=100, bufferkey=bufferkeypipa, transformfn=None)
        return keynamingfn

    def get_data(self, exclude_constant_labels=False, average=median, 
        ids=None, callback=None, bufver="0", transform=None, ctype=None, allowed_labels=None):
        """
        Get data in a single example table with labels of individual attributes
        set to annotations for query and post-processing
        instructions.

        Parameters: 
            average: function used for combining multiple reading of the same spot on
                a chip. If None, no averaging is done. Fuction should take a list
                of floats and return an "averaged" float.
            ids: a list of chip ids. If absent, make a search
            exclude_constant_labels: if a label has the same value in whole 
                example table, remove it
            format: if short, use short format for chip download
            ctype: expression type, from gene_expression_types

        Defaults: Median averaging.
        """

        def optcb():
            if callback: callback()

        cbc = CallBack(1, optcb, callbacks=10)

        if not ids:
            #returns ids of elements that match the search function
            #FIXME do a search
            searchNotDone

        cbc.end()

        #downloads annotations
        cbc = CallBack(len(ids), optcb, callbacks=10)

        readall = self.annotations()

        read = {}
        for a,b in readall.items():
            read[a] = b.items()

        cbc.end()

        #till now downloads were small

        import time
        tstart = time.time()

        #here download actually happens
        chipfn = None

        chipfn = lambda x: self.chips(x, ctype, bufver=bufver)
       
        if verbose:
            print "DOWNLOAD TIME", time.time() - tstart

        cbc = CallBack(len(ids)*2+len(ids)+1, optcb, callbacks=999-30)
        et = self.exampleTables(ids, spotmap={}, callback=cbc, annots=read, exclude_constant_labels=exclude_constant_labels, chipfn=chipfn, allowed_labels=allowed_labels)
        cbc.end()

        cbc = CallBack(1, optcb, callbacks=10)

        #transformation is performed prior to averaging
        if transform != None:
            transformValues(et, fn=transform) #in place transform
            cbc()

        #if average function is given, use it to join same spotids
        if average != None:
            et = averageAttributes(et, fn=average)
            cbc()

        cbc.end()

        return et

class DictyExpress(DBCommon):
    """
    Type is object id
    """
    
    aoidPairs = txt2ll("""time extractions.developmental_time_point
sample biological_samples.sample
growthCond biological_samples.growth_condition
treatment biological_samples.treatment
replicate biological_sample_replicates.replicate
techReplicate chips.replicate
platform chips.chip_platform
isTimeSeries biological_sample_replicates.is_time_series""")

    obidPairs = txt2ll("""norms normalizations
samples biological_samples
replicates biological_sample_replicates
analysis analysis
experiments experiments
extractions extractions
chips chips""")

    def __init__(self, address=defaddress, buffer=None):
        self.address = address
        self.db = DBInterface(address)
        self.buffer = buffer
        self.preload()

    def preload(self):

        # aoids are mappings from annotation name to annotation id
        self.aoids = ll2dic(self.__annotationTypes(), 1, 0)
        self.saoids = ll2dic(self.aoidPairs, 0, 1)
        self.aoidsr = reverseDic(self.aoids)
        self.saoidsr = reverseDic(self.saoids)

        # obids are mappings from object id to annotation id
        self.obids = ll2dic(self.__objects(), 1, 0)
        self.sobids = ll2dic(self.obidPairs, 0, 1)
        self.obidsr = reverseDic(self.obids)
        self.sobidsr = reverseDic(self.sobids)

    def aoidt(self, s):
        return chainLookup(s, [self.saoids, self.aoids], force=[-1])

    def obidt(self, s):
        return chainLookup(s, [self.sobids, self.obids], force=[-1])
 
    def aoidtr(self, s, **kwargs):
        return chainLookup(s, [self.aoidsr, self.saoidsr], **kwargs)

    def obidtr(self, s):
        return chainLookup(s, [self.obidsr, self.sobidsr])

    def pq(self, q):
        """
        Prepare query. 
        ||| separator between conditions, 
        *** denotes equivalence
        """
        o =  "|||".join([ self.aoidt(a) + "***" + b for a,b in q.items()])
        return o

    def geneInfo(self):
        res,legend = self.sq("action=gene_info")
        return res, legend

    def annotationOptions(self, ao=None, onlyDiff=False, **kwargs):
        """
        Returns annotation options for given query. Returns all possible 
        annotations if the query is omitted.

        If ao is choosen, only result
        """
        params = ""
        if len(kwargs) > 0: params += "&query=" + self.pq(kwargs)
        if ao: params += "&annotation_object_id=" +  self.aoidt(ao)
        res,legend = self.sq("action=get_annotation_options%s" % (params), bufadd=self.address)
        res = onlyColumns(res, legend, ['annotation_object_id', 'value' ])

        #join columns with the annotation object id
        joined = {}
        for key,v in res:
            key = self.aoidtr(key)
            cur = joined.get(key, [])
            cur.append(v)
            joined[key] = cur

        if onlyDiff:
            joined = dict([ (a,b) for a,b in joined.items() if len(b)>1 ])

        return dict([ (a, sorted(b)) for a,b in joined.items() ])

    def annotation(self, type, id):
        return list(self.annotations(type, [ id ]))[0]

    def meaningfulAnnot(self, name):
        if name in self.saoids:
            return True
        else:
            return False

    def keepOnlyMeaningful(self, annot):
        """
        Keep only meaningful annotations
        """
        if type(annot) == type({}):
            return dict( [ (a,b) for a,b in annot.items() \
                if self.meaningfulAnnot(a) ] )
        else:
            return [ [ a,b ] for a,b in annot \
                if self.meaningfulAnnot(a) ]


    def annotations(self, type, ids=None, all=False):
        """
        Returns a generator returning annotations for specified type and ids. 
        If ids are left blank, all annotations are outputed. Annotations are in the same order
        as input ids.
        If all is True, all annotations are kept, else keep only "meaningful".
        """
        
        inputids = False
        if ids != None:
            inputids = True
            antss = self.downloadMulti(
                "action=get_annotations&ids=$MULTI$&object_id=%s" 
                % (self.obidt(type)), ids)
        else:
            res,legend = self.sq(
                "action=get_annotations&object_id=%s"
                % (self.obidt(type)))
            antss = splitTableOnColumn(res, 0)
            ids = nth(antss.items(),0)
            antss = zip(nth(antss.items(),1), [ legend ]*len(antss))

        for ants in izip(antss,ids):
            (res, legend), id = ants
            res2 = onlyColumns(res, legend, ['name', 'value'])
            res2 = [ [ self.aoidtr(a),b ] for a,b in res2 ]
            if not all:
                res2 = self.keepOnlyMeaningful(res2)
            if inputids:
                yield res2
            else:
                yield (id, res2)

    def search(self, type, **kwargs):
        """
        Break search for multiple values of one attribute to independant searches.
        Search is case insensitive.
        
        List of searchable annotation types: self.saoids.keys()

        example usage:
        search("norms", platform='minichip', sample='abcC3-') 
            finds all ids of normalized entries where platform is minchip and 
            sample is abcC3-
        search("norms", platform='minichip', sample=[ 'abcC3-', 'abcG15-']) 
            finds all ids of normalized entries where platform is minichip and 
            sample is abcC3- or those where platform is minichip and sample
            is abcG15-
        """
        
        #search for all combinations of values - this is slow!

        l = []
        for k,v in kwargs.items():
            if not issequencens(v):
                v = [ v ]
            l.append((k,v))

        ares = []

        for r in mxrange([ len(v) for k,v in l ]):
            dico = {}
            for i,a in enumerate(r):
                dico[l[i][0]] = l[i][1][a]

            res,_ = self.sq("action=search&object_id=%s&query=%s" \
                % (self.obidt(type), self.pq(dico)), bufadd=self.address)

            ares += res

        return sorted(set(nth(ares, 0)))


    def chipN(self, id):
        return list(self.chipNs([id]))[0]

    def chipR(self, id):
        return list(self.chipRs([id]))[0]
  
    def chipNs(self, ids, remove_bad=True):
        """
        If removebad = True removes those with weights 0.
        """
          
        def sel(res, legend):
            #Drop unwanted columns - for efficiency
            res = onlyColumns(res, legend, ["spot_id", 'M', 'weights'])
            legend = onlyColumns( [ legend ], legend, ["spot_id", 'M', 'weights'])[0]

            def transf(a):
                if a[2] == 0:
                    return a[0], '', a[2]
                else:
                    return a[0], a[1], a[2]

            res = [ transf(a) for a in res ]
            res = onlyColumns(res, legend, ["spot_id", 'M'])
            legend = onlyColumns( [ legend ], legend, ["spot_id", 'M'])[0]
            return res, legend

        antss = self.downloadMulti("action=get_normalized_data&ids=$MULTI$", ids, chunk=2, transformfn=sel)
        for a,legend in antss:
            yield a   

    def chipNsN(self, ids, annots):
        """
        Download chips using new shorter format.
        """
        chip_map_ids = zip(ids,[ dict(a)['chips.chip_map_id'] for a in annots ])

        def separateByChipsMaps(l):
            begin = 0
            cm = l[0][1]
            cp = 0
            for id,m in l[1:]:
                cp += 1
                if m != cm:
                    yield l[begin:cp]
                    cm = m
                    begin = cp
            yield l[begin:cp+1]
        
        sep = list(separateByChipsMaps(chip_map_ids))
      
        def sel(res, legend):
            #Drop unwanted columns - for efficiency
            res = onlyColumns(res, legend, ["spot_id", 'M'])
            legend = onlyColumns( [ legend ], legend, ["spot_id", 'M'])[0]
            return res, legend

        def separatefn(res):
            #each one is own rown
            #genes are in the first row
            genes = res[0][1:]
            cids = nth(res,0)[1:]

            antss = {}
            for i,cid in enumerate(cids):
                row = i+1
                vals = res[row][1:]
                antss[cid] = [ list(a) for a in zip(genes, vals) ]
            return ['spot_id', 'M'], antss

        for part in sep:
            pids = nth(part,0)
            antss = self.downloadMulti("action=get_normalized_data&mergeexperiments=1&ids=$MULTI$", pids, chunk=10, transformfn=sel, separatefn=separatefn)
            for a, legend in antss:
                yield a

    def chipRs(self, id):
        antss = self.downloadMulti("action=get_raw_data&ids=$MULTI$", ids, chunk=2)
        for a,legend in antss:
            yield a
  
    def spotId(self):
        res,legend = self.sq("action=spot_id_mapping")
        res2 = onlyColumns(res, legend, ["spot_id", 'ddb_g', 'genename'])
        return res2

    def annotationTypes(self):
        """
        Returns all annotation types.
        """
        return self.aoids.keys()

    def objects(self):
        """
        Returns all objects.
        """
        return self.obids.keys()

    def __annotationTypes(self):
        """
        Returns list of [ annotation_object_id, name ]
        """
        res, legend = self.sq("action=get_annotation_types")
        res2 = onlyColumns(res, legend, ["annotation_object_id", 'name'])
        return res2

    def __objects(self):
        res, legend = self.sq("action=get_objects")
        res2 = onlyColumns(res, legend, ["object_id", 'object_name'])
        return res2

    def spotMap(self):
        spotids = self.spotId()
        spotmap = [ (a[0],a[1]) for a in spotids ]

        spotmapd = {}

        for a,b in spotmap:
            if a in spotmapd:
                spotmapd[a] = spotmapd[a] + "-" + b
            else:
                spotmapd[a] = b

        return spotmapd

    def getData(self, *args, **kwargs):
        deprecatedError("Use get_single_data instead")

    def get_data(self, type="norms", exclude_constant_labels=False, average=median, 
        ids=None, callback=None, format="short", transform=None, allowed_labels=None, **kwargs):
        """
        Get data in a single example table with labels of individual attributes
        set to annotations for query and post-processing
        instructions.

        Parameters: 
            average: function used for combining multiple reading of the same spot on
                a chip. If None, no averaging is done. Fuction should take a list
                of floats and return an "averaged" float.
            ids: a list of chip ids. If absent, make a search
            exclude_constant_labels: if a label has the same value in whole 
                example table, remove it
            format: if short, use short format for chip download

        Defaults: Median averaging.
        """

        def optcb():
            if callback: callback()

        cbc = CallBack(1, optcb, callbacks=10)

        if not ids:
            #returns ids of elements that match the search function
            ids = self.search(type, **kwargs)

        cbc.end()

        #downloads annotations
        cbc = CallBack(len(ids), optcb, callbacks=10)

        readall = self.dictionarize(ids, self.annotations, type, ids, all=True, callback=cbc)

        read = {}
        for a,b in readall.items():
            read[a] = self.keepOnlyMeaningful(b)

        annotsinlist = [] #annotations in the same order
        for id in ids:
            annotsinlist.append(readall[id])

        if verbose:
            print zip(ids,[ dict(a)['chips.chip_map_id'] for a in annotsinlist ])

        cbc.end()

        #till now downloads were small

        import time
        tstart = time.time()

        #here download actually happens
        chipfn = None

        if type == "norms":
            if format == "short":
                chipfn = lambda x, al=annotsinlist: self.chipNsN(x, al)
            else:
                chipfn = self.chipNs
        else:
            chipfn = self.chipRs
        
        if verbose:
            print "DOWNLOAD TIME", time.time() - tstart

        cbc = CallBack(len(ids)*2+len(ids)+1, optcb, callbacks=999-30)
        et = self.exampleTables(ids, spotmap=self.spotMap(), callback=cbc, annots=read, exclude_constant_labels=exclude_constant_labels, chipfn=chipfn, allowed_labels=allowed_labels)
        cbc.end()

        cbc = CallBack(1, optcb, callbacks=10)

        #transformation is performed prior to averaging
        if transform != None:
            transformValues(et, fn=transform) #in place transform
            cbc()

        #if average function is given, use it to join same spotids
        if average != None:
            et = averageAttributes(et, fn=average)
            cbc()

        cbc.end()

        return et
    
    def get_single_data(self, *args, **kwargs):
        return self.get_data(*args, **kwargs)

class DatabaseConnection(DictyExpress):
    pass

def allAnnotationVals(annots):
    """
    All annotation valuess for given annotations
    in a dict of { name: set of possible values } pairs.
    """
    av = defaultdict(set)
    for a in annots:
        for name,val in a:
            av[name].add(val)
    return av

def createExampleTable(names, vals, annots, ddb, cname="DDB", \
        exclude_constant_labels=False, permutation=None, always_include=["id"], allowed_labels=None):
    """
    Create an ExampleTable for this group. Attributes are those in
    names. 
    """
    attributes = [ orange.FloatVariable(n, numberOfDecimals=3) \
        for n in names ]

    #exclusion of names with constant values
    annotsvals = allAnnotationVals(annots)
    oknames = set(annotsvals.keys())
    if exclude_constant_labels:
        oknames = set(nth(filter(lambda x: len(x[1]) > 1 or x[0] in always_include, 
            annotsvals.items()), 0))

    if allowed_labels != None:
        oknames = set(filter(lambda x: x in allowed_labels, oknames))

    #print oknames

    for a,an in zip(attributes, annots):
        a.setattr("attributes", dict([(name,val) for name,val in an if name in oknames]))

    domain = orange.Domain(attributes, False)
    ddbv = orange.StringVariable(cname)
    id = orange.newmetaid()
    domain.addmeta(id, ddbv)

    examples = [ None ]*len(ddb)
    for i,(v,d) in enumerate(izip(izip(*vals), ddb)):
        ex = orange.Example(domain, [ floatOrUnknown(a) for a in v ])
        ex[cname] = d

        if permutation:
            examples[permutation[i]] = ex
        else:
            examples[i] = ex

    return orange.ExampleTable(domain,examples)

def transformValues(data, fn):
    """
    In place transformation.
    """
    for ex in data:
        for at in data.domain.attributes:
            if not ex[at].isSpecial():
                ex[at] = fn(ex[at])

def averageAttributes(data, joinc="DDB", fn=median):
    """
    Averages attributes with the same "join" parameter using
    specified function. Builds a now dataset. Order of
    "join" parameter stays the same with regard to first
    appearance.
    """

    if verbose:
        print "Averaging attributes"

    valueso = []
    valuess = set(valueso)

    attributes = [ a for a in data.domain.attributes ]
    domain = data.domain

    #accumulate values
    valuesddb = dict( [ (at,{}) for at in attributes ])

    for ex in data:
        #join attribute - ddb
        j = str(ex[joinc])

        if j not in valuess:
            valueso.append(j)
            valuess.add(j)

        for a in attributes:
            val = ex[a]
            l = valuesddb[a].get(j, [])
            if not val.isSpecial():
                l.append(val.native())
            valuesddb[a][j] = l

    #print "len valueso", len(valueso)
    #print sorted(set([ str(ex[join]) for ex in data ]))

    #apply function fn to each attribute

    """
    for i,at in enumerate(data.domain.attributes):
        print valuesddb[at]["DDB_G0282817"], "CI" + data.annot["chipids"][i]
    """

    for a in attributes:
        for n,v in valuesddb[a].items():
            valuesddb[a][n] = floatOrUnknown(fn(v))
            #if n == "DDB_G0282817": print valuesddb[a][n]

    #create a new example table reusing the domain
    examples = []
    for v in valueso:
        example = orange.Example(domain, \
            [ valuesddb[a][v] for a in attributes ] )
        example[joinc] = v
        examples.append(example)

    return orange.ExampleTable(domain, examples)

def floatOrUnknown(a):
    """
    Converts an element to float if possible.
    If now, output "?".
    """
    try:
        return float(a)
    except:
        return "?"

class CallBack():
    """
    Converts "allparts" callbacks into by "callbacks"
    specified number of callbacks of function fn.
    """

    def __init__(self, allparts, fn, callbacks=100):
        self.allparts = allparts
        self.lastreport = 0.00001
        self.getparts = 0
        self.increase = 1.0/callbacks
        self.callbacks = callbacks
        self.fn = fn
        self.cbs = 0

    def __call__(self):
        self.getparts += 1
        done = float(self.getparts)/self.allparts
        while done > self.lastreport + self.increase:
            self.lastreport += self.increase
            self.fn()
            self.cbs += 1

    def end(self):
        while self.cbs < self.callbacks:
            self.fn()
            self.cbs += 1

class BufferSQLite(object):

    def __init__(self, filename, compress=True):
        self.compress = compress
        self.filename = filename
        self.conn = self.connect()

    def clear(self):
        """
        Removes all entries in the buffer
        """
        self.conn.close()
        os.remove(self.filename)
        self.conn = self.connect()

    def connect(self):
        import sqlite3
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        c.execute('''create table if not exists buf
        (address text primary key, time text, con blob)''')
        c.close()
        conn.commit()
        return conn

    def contains(self, addr):
        """ Returns version or False, if it does not exists """
        c = self.conn.cursor()
        c.execute('select time from buf where address=?', (addr,))
        lc = list(c)
        c.close()
        if len(lc) == 0:
            return False
        else:
            return lc[0][0]

    def list(self):
        c = self.conn.cursor()
        c.execute('select address from buf')
        return nth(list(c), 0)

    def add(self, addr, con, version="0", autocommit=True):
        import cPickle, zlib, sqlite3
        if verbose:
            print "Adding", addr
        c = self.conn.cursor()
        if self.compress:
            bin = sqlite3.Binary(zlib.compress(cPickle.dumps(con)))
        else:
            bin = sqlite3.Binary(cPickle.dumps(con))
        c.execute('insert or replace into buf values (?,?,?)', (addr, version, bin))
        c.close()
        if autocommit:
            self.commit()

    def commit(self):
        self.conn.commit()

    def get(self, addr):
        import cPickle, zlib
        if verbose:
            print "getting from buffer", addr
            t = time.time()
        c = self.conn.cursor()
        c.execute('select con from buf where address=?', (addr,))
        ls = list(c)
        first = ls[0][0]
        if verbose:
            print time.time() - t
        if self.compress:
            rc = cPickle.loads(zlib.decompress(first))
        else:
            rc = cPickle.loads(str(first))
        c.close()

        if verbose:
            print time.time() - t
        return rc

def download_url(url, repeat=2):
    def do():
        return urllib2.urlopen(url)

    if repeat <= 0:
        do()
    else:
        try:
            return do()
        except:
            return download_url(url, repeat=repeat-1)

def empty_none(s):
    if s:
        return s
    else:
        return None

def join_ats(atts):
    """ Joins attribute attributes together. If all values are the same,
    set the parameter to the common value, else return a list of the
    values in the same order as the attributes are imputed. """
    keys = reduce(lambda x,y: x | y, (set(at.keys()) for at in atts))
    od = {}

    def man(x):
        if issequencens(x):
            return tuple(x)
        else:
            return x

    for k in keys:
        s = set(man(at[k]) for at in atts)
        if len(s) == 1:
            od[k] = list(s)[0]
        else:
            od[k] = [ at[k] for at in atts ]
    return od

def join_replicates(data, ignorenames=["id", "replicate", "name", "map_stop1"], namefn=None, avg=median):
    """ Join replicates by median. 
    Default parameters work for PIPA data."""
    d = defaultdict(list)

    if namefn == None:
        namefn = lambda att: ",".join(att["id"]) if issequencens(att["id"]) else att["id"]

    #key function
    def key_g(att):
        dk = att.copy()
        for iname in ignorenames:
            dk.pop(iname, None)
        
        def man(x):
            if issequencens(x):
                return tuple(x)
            else:
                return x

        return tuple(nth(sorted(((a, man(b)) for a,b in dk.items())), 1))

    #prepare groups
    for i,a in enumerate(data.domain.attributes):
        att = a.attributes
        k = key_g(att)
        d[k].append(i)

    d = dict(d) #want errors with wrong keys

    natts = []

    def nativeOrNone(val):
        if val.isSpecial(): 
            return None
        else: 
            return val.native()

    def avgnone(l):
        """ Removes None and run avg function"""
        l = filter(lambda x: x != None, l)
        if len(l):
            return avg(l)
        else:
            return None

    for group, elements in d.items():
        a = orange.FloatVariable()
        a.attributes.update(join_ats([data.domain.attributes[i].attributes for i in elements]))
        a.name = namefn(a.attributes)

        def avgel(ex, el):
            return orange.Value(avgnone([ nativeOrNone(ex[ind]) for ind in el ]))

        a.getValueFrom = lambda ex,rw,el=elements: avgel(ex,el)
        natts.append(a)

    ndom = orange.Domain(natts, data.domain.classVar)
    ndom.addmetas(data.domain.getmetas())
    return orange.ExampleTable(ndom, data)


class DictyBase(object):

    domain = "dictybase"
    filename = "information_mappings.pck"
    tags = [ "Dictyostelium discoideum", "gene", "essential", "dictyBase" ] 
 
    @classmethod
    def version(cls):
        orngServerFiles.localpath_download(cls.domain, cls.filename)
        return orngServerFiles.info(cls.domain, cls.filename)["datetime"]
    
    @classmethod
    def download_information(cls):
        """ 
        Downloads gene information and parses it. 
        Returns a dictionary {ID: (name, synonyms, products)}
        """
        s = download_url("http://www.dictybase.org/db/cgi-bin/dictyBase/download/download.pl?area=general&ID=gene_information.txt").read()
        out = []
        for l in txt2ll(s, separ='\t', lineSepar='\n')[1:]:
            if len(l) == 4:
                id = l[0]
                name = l[1]
                synonyms = filter(None, l[2].split(", "))
                products = l[3]
                out.append((id, name, synonyms, products))
        return dict((a,(b,c,d)) for a,b,c,d in out)

    @classmethod
    def download_mappings(cls):
        """ 
        Downloads DDB-GeneID-UniProt mappings and parses them. 
        Returns a list of (ddb, ddb_g, uniprot) triplets.
        
        2009/04/07: ddb's appear unique
        """
        s = download_url("http://www.dictybase.org/db/cgi-bin/dictyBase/download/download.pl?area=general&ID=DDB-GeneID-UniProt.txt").read()
        out = []
        for l in txt2ll(s, separ='\t', lineSepar='\n')[1:]:
            if len(l) == 3:
                ddb = empty_none(l[0])
                ddb_g = empty_none(l[1])
                uniprot = empty_none(l[2])
                out.append((ddb, ddb_g, uniprot))
        return out

    @classmethod
    def pickle_data(cls):
        info = cls.download_information()
        mappings = cls.download_mappings()
        return pickle.dumps((info,mappings), -1)

    def __init__(self):
        fn = orngServerFiles.localpath_download(self.domain, self.filename)
        self.info, self.mappings = pickle.load(open(fn, 'rb'))

if __name__=="__main__":
    verbose = 1

    def printet(et):
        et.save("ett.tab")
        print open("ett.tab").read()

    """
    a = DictyBase()
    print len(a.info)

    dbc = DictyExpress(buffer=BufferSQLite("../tmpbufnew"))

    print dbc.annotationOptions()

    count = 0
    def cb():
        global count
        count += 1
        #print "CBBB", count

    et = dbc.get_single_data(sample=[ "tagA-", "pkaC-"], callback=cb, exclude_constant_labels=True, allowed_labels=["sample"])
    print et.domain
    print et.domain[0].attributes
    printet(et)
    """

    d = PIPA(buffer=BufferSQLite("../tmpbufnewpipa"))

    print d.gene_expression_types()

    allids = d.annotations().keys()
    print ("list", allids)
    print d.annotations().items()[0]
    print ("annots", d.annotations().items()[:2])

    allids = map(str, [ 151, 150 ])

    import math

    #data = d.get_data(ids=allids)
    data = d.get_data(ids=allids, transform=lambda x: math.log(x+1, 2), allowed_labels=["strain"], ctype="3")

    printet(data)
