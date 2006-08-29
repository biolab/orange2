import sys
from xml.sax import handler, make_parser
#from tokenizer import *
#import lemmatizer
from loadWordSet import *

from modulTMT import tokenize
import modulTMT as lemmatizer


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

class orngXMLData:
    def __init__(self, fileName, tags = None, lem = None ):
        self.docIDs = []        # meta information for each document
        self.categories = []  # categories for each document
        self.docWords = []   # document words matrix
        self.allWords = {}   # contains words and their indices
        
        # lemmatizer should have
        #   method isStopwords(string) -> bool
        #   method getLemmas(string) -> vectorStr // std::vector<std::string>
        
        ## remove this
        if not lem:
                self.lem = lemmatizer.FSALemmatization('vjesnik-10m.molex-3.3-1-ex-2way.fsa')
                for word in loadWordSet('hrvatski_stoprijeci.txt'):
                    self.lem.stopwords.append(word)    
                    
        ## user should provide lemmatizer
        ##############################
        
        self.fileName = fileName
        f = open(fileName, "r")
        t = DocumentSetRetriever(f)       
        
        while 1:
            # load document
            doc = t.getNextDocument()
            if not len(doc): break
            self.docIDs.append(doc['meta'])
            self.categories.append(doc['categories'])
        
            # extract words from document
            tokens = tokenize(doc['content'].lower().encode('iso-8859-2'))
            words = []
            for token in tokens:
                if not self.lem.isStopword(token):
                    lemmas = self.lem.getLemmas(token)
                    if lemmas.empty():
                        words.append(token)
                    else:                        
                        for lemma in lemmas:
                            words.append(lemma)
            
            # convert words to thier integer representation -- index from allWords
            intWords = []
            for word in words:
                if not self.allWords.has_key(word):
                    tmp = self.allWords[word] = len(self.allWords)
                    intWords.append(tmp)
                else:
                    intWords.append(self.allWords[word])
                
            self.docWords.append(intWords)

        # compact doc - word representation
        tmp = []
        for (i, words) in zip(range(len(self.docIDs)), self.docWords):
            singleDoc = [0] * len(self.allWords)
            for j in words:
                singleDoc[j] = singleDoc[j] + 1
            tmp.append(singleDoc)
                
        self.docWords = tmp
        
    def getCategories(self):
        categories = flatten(self.categories)                    
        return removeDuplicates(categories)
        
    def getDocumentInCategories(self, categories):
        if isinstance(categories, str):
            categories = [categories]        
        doc = []
        for i, category in zip(range(len(self.categories)), self.categories):
            if len(intersection(category, categories)):
                doc.append(self.docIDs[i]) 
 
        doc = [" ".join(["%s %s" % meta for meta in metas])
                            for metas in self.docIDs]
        return doc
    


###############

class DocumentSetHandler(handler.ContentHandler):            
    def __init__(self, tags = None, doNotParse = []):
        self.tagsToHandle = ["document", "content", "categories", "category"]
        # set default XML tags
        if not tags:
            self.tags = {}
            for tag in self.tagsToHandle:
                self.tags[tag] = tag
        else:
            self.tags = tags
            
        # for current document being parsed
        self.curDoc = {}
        self.documents = []
        self.level = {}
        for tag in self.tagsToHandle:
            self.level[tag] = 0
            
        # other settings
        self.doNotParse = doNotParse[:]
        self.doNotParseFlag = 0
    def startElement(self, name, attrs):                       
        try:
            func = getattr(self, 'do' + name.capitalize())
            self.level[name] += 1
            func(attrs)
        except:
            if name in self.doNotParse:
                self.doNotParseFlag += 1
    def endElement(self, name):                                
        if name in self.tagsToHandle:
            self.level[name] -= 1
            if name == self.tags["document"]:
                self.curDoc["category"] = []
                self.curDoc["content"] = "".join(self.curDoc["content"])
                self.documents.append(self.curDoc)
                self.curDoc = {}
            elif name == self.tags["category"]:
                self.curDoc["categories"].append("".join(self.curDoc["category"]))
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
    def doContent(self, attrs):
        self.curDoc["content"] = []
    def doCategories(self, attrs):
        self.curDoc["categories"] = []
    def doCategory(self, attrs):
        self.curDoc["category"] = []
        
class DocumentSetRetriever:
    def __init__(self, source):
        self.source = source
        self.handler = DocumentSetHandler()
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
    a = orngXMLData('reuters-exchanges.xml')
    print a.getCategories()
    print a.getDocumentInCategories("nyse")
##    print len(a.docWords)
