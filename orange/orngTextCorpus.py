import sys
from xml.sax import handler, make_parser
from modulTMT import tokenize
import modulTMT as lemmatizer
import orange


################
## utility functions
################
def loadWordSet(f):
    f = open(f, 'r')
    setW = []
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.lower()
        if line.find(';') != -1:
            line = line[:line.find(';')]
        line = line.strip(' \t\r\n')
        if len(line):
            setW.append(line)
    return setW

def flatten(l):
    ret = []
    if isinstance(l, list):
        for elem in l:
            ret.append(flatten(elem))
    else:
        return l
    return ret

def removeDuplicates(l):
    if not l: return []
    ret = []
    for elem in l:
        if elem not in ret:
            ret.append(elem)
    return elem

def intersection(l1, l2):
    return [e for e in l1 if e in l2]

################


class TextCorpusLoader:
    def __init__(self, fileName, tags = {}, additionalTags = [], lem = None, doNotParse = []):
        if lem:
            self.lem = lem
        else:
            self.lem = lemmatizer.NOPLemmatization()
        
        cat = orange.StringVariable("category")
        meta = orange.StringVariable("meta")
        addCat = [cat, meta]
        if additionalTags:
            addCat.extend([orange.StringVariable(s) for s in additionalTags])
        dom = orange.Domain(addCat, 0)
        self.data = orange.ExampleTable(dom)
    
        f = open(fileName, "r")
        t = DocumentSetRetriever(f, tags = tags, doNotParse = doNotParse, additionalTags = additionalTags)       
        
        while 1:
            # load document
            ex = orange.Example(dom)
            
            doc = t.getNextDocument()
            if not len(doc): break
            ex['meta'] = " ".join([("%s=\"%s\"" % meta).encode('iso-8859-2') for meta in doc['meta']])
            ex['category'] = ".".join([d.encode('iso-8859-2') for d in doc['categories']])
            for tag in additionalTags:
                ex[tag.encode('iso-8859-2')] = doc[tag].encode('iso-8859-2')
        
            # extract words from document
            tokens = tokenize(doc['content'].lower().encode('iso-8859-2'))
            words = []
            for token in tokens:
                if not self.lem.isStopword(token):
                    lemmas = self.lem.getLemmas(token)                    
                    if lemmas.empty():
##                        self.__incFreqWord(ex, token)
                        pass
                    else:                       
                        for lemma in lemmas:
                            self.__incFreqWord(ex, lemma)
            
            self.data.append(ex)
            
        self.data.setattr("meta_names", "fromText")
                            
    def __incFreqWord(self, ex, w):         
        domain = ex.domain
        if domain.hasmeta(w):
            id = domain.metaid(w)
            if ex.hasmeta(id):
                ex[id] += 1.0
            else:
                ex[id] = 1.0
        else:
            id = orange.newmetaid()
            domain.addmeta(id, orange.FloatVariable(w), True)
            ex[id] = 1.0          



    


###############

class DocumentSetHandler(handler.ContentHandler):            
    def __init__(self, tags = None, doNotParse = [], additionalTags = []):
        self.tagsToHandle = ["content", "category", "document", "categories"]
        # set default XML tags
        if not tags:
            self.tags = {}
            for tag in self.tagsToHandle:
                self.tags[tag] = tag
        else:
            self.tags = tags
            
        for k in additionalTags:
            if k not in self.tagsToHandle:
                self.tags[k] = k
            
        # for current document being parsed
        self.curDoc = {}
        self.documents = []
        self.level = {}
        for tag in self.tags:
            self.level[tag] = 0
            
        # other settings
        self.doNotParse = doNotParse[:]
        self.doNotParseFlag = 0
    def startElement(self, name, attrs):       
        if name in self.doNotParse:
            self.doNotParseFlag += 1
        else:
            try:
                func = getattr(self, 'do' + name.capitalize())
                self.level[name] += 1
                func(attrs)
            except:
                if name in self.tags:
                    self.level[name] += 1
                    self.curDoc[name] = []
    def endElement(self, name):                                
        if name in self.tags:
            self.level[name] -= 1
            if name == self.tags["document"]:
                self.curDoc["category"] = []
                self.curDoc["content"] = "".join(self.curDoc["content"])
                self.documents.append(self.curDoc)
                self.curDoc = {}
            elif name == self.tags["category"]:
                self.curDoc["categories"].append("".join(self.curDoc["category"]))
            elif name != self.tags["categories"]:
                self.curDoc[name] = "".join(self.curDoc[name])
        elif name in self.doNotParse:
            self.doNotParseFlag -= 1
    def characters(self, chrs):
        if not self.doNotParseFlag:
            try:
                # check in which tag are the characters
                # only tagsToHandle are parsed, others will raise exception that is ignored
                name = [k  for k, v in self.level.items() if k not in ["document", "categories"] and v][0]
                self.curDoc[name].append(chrs)
            except:
                pass
    def doDocument(self, attrs):
        self.curDoc["meta"] = attrs.items()[:]
        self.curDoc["content"] = []
        self.curDoc["category"] = []
        self.curDoc["categories"] = [] 
    def doContent(self, attrs):
        self.curDoc["content"] = []
    def doCategory(self, attrs):
        self.curDoc["category"] = []
    def doCategories(self, attrs):
        self.curDoc["categories"] = []        
class DocumentSetRetriever:
    def __init__(self, source, tags = None, doNotParse = [], additionalTags = []):
        self.source = source
        self.handler = DocumentSetHandler(tags, doNotParse, additionalTags)
        self.parser = make_parser()
        self.parser.reset()
        self.parser.setContentHandler(self.handler)    
    def getNextDocument(self):
        while 1:
            if len(self.handler.documents):
                curDoc = self.handler.documents.pop(0)
                return curDoc
            if self.source.closed:
                return {}
            chunk = self.source.read(10000)
            if not chunk:
                self.parser.close()               
                return {}
            else:
                self.parser.feed(chunk)     
                
if __name__ == "__main__":
    lem = lemmatizer.FSALemmatization('OrangeWidgets/TextData/engleski_rjecnik.fsa')
    for word in loadWordSet('OrangeWidgets/TextData/engleski_stoprijeci.txt'):
        lem.stopwords.append(word)       
    a = TextCorpusLoader('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/test.xml', lem = lem, additionalTags = ['date'], doNotParse = ['dontparse'])
##    a = TextCorpusLoader('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/test1.xml', lem = lem)
