import sys
from xml.sax import handler, make_parser
#from tokenizer import *
#import lemmatizer
from loadWordSet import *

from modulTMT import tokenize
import modulTMT as lemmatizer

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

def main():
    docIDs = []        # meta information for each document
    categories = []  # categories for each document
    docWords = []   # document words matrix
    allWords = {}   # contains words and their indices
    
    # initialize lemmatizer
    lem = lemmatizer.FSALemmatization('vjesnik-10m.molex-3.3-1-ex-2way.fsa')
    for word in loadWordSet('hrvatski_stoprijeci.txt'):
        lem.stopwords.append(word)    
    
    f = open(sys.argv[1], "r")
    t = DocumentSetRetriever(f)
    count  = 0
    while 1:
        # load document
        doc = t.getNextDocument()
        if not len(doc): break
        docIDs.append(doc['meta'])
        categories.append(doc['categories'])
        
        # extract words from document
        tokens = tokenize(doc['content'].lower().encode('iso-8859-2'))
        words = []
        for token in tokens:
            if not lem.isStopword(token):
                lemmas = lem.getLemmas(token)
                if lemmas.empty():
                    words.append(token)
                else:                        
                    for lemma in lemmas:
                        words.append(lemma)
            
        # convert words to thier integer representation -- index from allWords
        intWords = []
        for word in words:
            if not allWords.has_key(word):
                tmp = allWords[word] = len(allWords)
                intWords.append(tmp)
            else:
                intWords.append(allWords[word])
                
        docWords.append(intWords)
        count += 1
##        print count
    
    # compact doc - word representation
    tmp = []
    for (i, words) in zip(range(len(docIDs)), docWords):
        singleDoc = [0] * len(allWords)
        for j in words:
            singleDoc[j] = singleDoc[j] + 1
        tmp.append(singleDoc)
            
    docWords = tmp
    
    #### print for debuging

    print "+++++++++++++++++++ meta +++++++++++++++++++"
    print "\n".join(
        ["%s %s" % inZip 
            for inZip in zip(
                    range(len(docIDs)), 
                    [" ".join(["%s %s" % meta for meta in metas])
                        for metas in docIDs]
                    )
        ])
    print "++++++++++++++++++++++++++++++++++++++++++++"
    print 
    print "+++++++++++++++++++ categories +++++++++++++++++++"
    print "\n".join(
        ["%s %s" % inZip 
            for inZip in zip(
                    range(len(docIDs)), 
                    [" ".join(categoriesDoc)
                        for categoriesDoc in categories]
                    )
        ])    
    print "+++++++++++++++++++++++++++++++++++++++++++++++++"
    print 
    print "+++++++++++++++++++ words +++++++++++++++++++"
    print "\n".join(["%s %s" % word for word in allWords.items()])
    print "+++++++++++++++++++++++++++++++++++++++++++++"
    print 
    print "+++++++++++++++++++ doc-word freq +++++++++++++++++++"
    print "\n".join([" ".join(["%s" % freq for freq in freqs]) 
            for freqs in docWords
            ])
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print 
    
    
if __name__ == '__main__':
    main()
