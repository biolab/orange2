import sys
from xml.sax import handler, make_parser
from modulTMT import tokenize
import modulTMT as lemmatizer
import orange
import operator

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

##def flatten(l):
##    ret = []
##    if isinstance(l, list):
##        for elem in l:
##            ret.append(flatten(elem))
##    else:
##        return l
##    return ret

##def removeDuplicates(l):
##    if not l: return []
##    ret = []
##    for elem in l:
##        if elem not in ret:
##            ret.append(elem)
##    return elem

##def intersection(l1, l2):
##    return [e for e in l1 if e in l2]

################
class FeatureSelection:
    measures = {
                            'Document Frequency': 'DF'
                        }
    def __init__(self, dataInput, userMeasures = []):
        self.dataInput = dataInput
        self.data = None
        
        if not hasattr(self.dataInput, 'meta_names'):
            self.dataInput = None
            self.data = None
            return
            
        if not userMeasures:
            userMeasures = self.measures.keys()
        self.data = orange.ExampleTable(orange.Domain([]))
        lstMeasures  = [m for m in userMeasures if FeatureSelection.measures.has_key(m)]
        for measure in lstMeasures:
            getattr(self, "_" + FeatureSelection.measures[measure])(measure)
        for id, name in zip(range(len(self.data)), self.dataInput.domain.getmetas().values()):
            self.data[id].name = name.name     
        
    def _DF(self, nameOfAttribute):
        meta2index = dict(zip(self.dataInput.domain.getmetas().keys(), range(len(self.dataInput.domain.getmetas().keys()))))
        df  = [0] * len(self.dataInput.domain.getmetas().keys())
        
        for ex in self.dataInput:
            toinc = [meta2index[meta] for meta in ex.getmetas().keys()]
            for i in toinc:
                df[i] = df[i] + 1
        
        df = [[i] for i in df]
##        df = [[len([ex for ex in self.dataInput if ex.hasmeta(meta.name)])] 
##            for meta in self.dataInput.domain.getmetas().values()]
            
        dom = orange.Domain([orange.FloatVariable(nameOfAttribute)], 0)
        exTable = orange.ExampleTable(dom, df)
        if len(self.data):
            self.data = orange.ExampleTable(self.data, exTable)
        else:
            self.data = orange.ExampleTable(exTable)
            
    def getFeatureMeasures(self):
        return self.data
        
    def selectFeatures(self, filter = None, list = None):
        if not self.dataInput:
            return None
        newDomain = orange.Domain(self.dataInput.domain)            
        if filter:            
            fdata = filter(self.data, negate = 1)
            removeMeta = [ex.name for ex in fdata if self.dataInput.domain.hasmeta(ex.name)]            
        elif list:
            removeMeta = [el for el in list if self.dataInput.domain.hasmeta(el)]
        else:
            return None
        newDomain.removemeta(removeMeta)            
        return orange.ExampleTable(newDomain, self.dataInput)
        
class TextCorpusLoader:
    def __init__(self, fileName, tags = {}, additionalTags = [], lem = None, doNotParse = [] , wordsPerDocRange = (-1, -1), charsPerDocRange = (-1, -1)):
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
                
            if not len(charsPerDocRange) == 2:
                raise Exception('length of charsPerDocRange != 2')                
            if not charsPerDocRange[0] == -1:
                if len(doc['content']) <= charsPerDocRange[0]: continue
            if not charsPerDocRange[1] == -1:
                if len(doc['content']) >= charsPerDocRange[1]: continue
            
            ex['meta'] = " ".join([("%s=\"%s\"" % meta).encode('iso-8859-2') for meta in doc['meta']])
            ex['category'] = ".".join([d.encode('iso-8859-2') for d in doc['categories']])
            for tag in additionalTags:
                ex[tag.encode('iso-8859-2')] = (doc.has_key(tag) and [doc[tag].encode('iso-8859-2')] or [''])[0]
        
            # extract words from document
            tokens = tokenize(doc['content'].lower().encode('iso-8859-2'))
            
            if not len(wordsPerDocRange) == 2:
                raise Exception('length of wordsPerDocRange != 2')                
            if not wordsPerDocRange[0] == -1:
                if len(tokens) <= wordsPerDocRange[0]: continue
            if not wordsPerDocRange[1] == -1:
                if len(tokens) >= wordsPerDocRange[1]: continue

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

class CategoryDocument:
    def __init__(self, data):
        self.data = data
        newDomain = orange.Domain([])
        newDomain.addmetas(self.data.domain.getmetas(), True)
        self.dataCD = orange.ExampleTable(newDomain)
        categories = set()
        for ex in self.data:
            if ex['category']:
                for cat in ex['category'].native().split('.'):
                    categories.add(cat)
        categories = list(categories)
        if not len(categories): return None
        for cat in categories:
            ex =  orange.Example(newDomain)
            ex.name = cat
            self.dataCD.append(ex)
        for ex in self.data:
            cat = (ex['category'] and [ex['category'].native().split('.')] or [[]])[0]
            if not cat: continue
            indices = [categories.index(c) for c in cat]
            for id, val in ex.getmetas().items():
                for i in indices:
                    try:
                        self.dataCD[i][id] = self.dataCD[i][id].native() + val.native()
                    except:
                        self.dataCD[i][id] = val.native()
        self.dataCD.setattr("meta_names", "fromText")
            
        
###############

class DocumentSetHandler(handler.ContentHandler):            
    def __init__(self, tags = None, doNotParse = [], additionalTags = []):
        self.tagsToHandle = ["content", "category", "document", "categories"]
        # set default XML tags
        self.name2tag = {}
        if not tags:
            self.tags = {}
            for tag in self.tagsToHandle:
                self.tags[tag] = tag
                self.name2tag[tag] = tag
        else:
            self.tags = {}
            for tag in self.tagsToHandle:
                if not tags.has_key(tag):
                    self.tags[tag] = tag
                    self.name2tag[tag] = tag
                else:
                    self.tags[tag] = tags[tag]
                    self.name2tag[tags[tag]] = tag
            
        for k in additionalTags:
            if k not in self.tagsToHandle:
                self.tags[k] = k
                
            
        # for current document being parsed
        self.curDoc = {}
        self.documents = []
        self.level = []
            
        # other settings
        self.doNotParse = doNotParse[:]
        self.doNotParseFlag = 0
    def startElement(self, name, attrs):
        if self.name2tag.has_key(name):
            name = self.name2tag[name]
##        if name == "document": 
##            globals()['countdoc'] =globals()['countdoc'] + 1
        if name in self.doNotParse:
            self.doNotParseFlag += 1
        else:
            try:
                func = getattr(self, 'do' + name.capitalize())
                self.level.append(name)
                func(attrs)
            except:
                if name in self.tags:
                    self.level.append(name)
                    self.curDoc[name] = []
    def endElement(self, name):
        if self.name2tag.has_key(name):
            name = self.name2tag[name]        
        if name in self.tags:
            self.level.pop()
            if name == "document":
                self.curDoc["category"] = []
                self.curDoc["content"] = "".join(self.curDoc["content"])
                self.documents.append(self.curDoc)
                self.curDoc = {}
            elif name == "category":
                self.curDoc["categories"].append("".join(self.curDoc["category"]))
            elif name != "categories":
                self.curDoc[name] = "".join(self.curDoc[name])
        elif name in self.doNotParse:
            self.doNotParseFlag -= 1
    def characters(self, chrs):
        if not self.doNotParseFlag:
            try:
                # check in which tag are the characters
                # only tagsToHandle are parsed, others will raise exception that is ignored
                name = self.level[-1]
                if name not in ["document", "categories"]:
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
    hrdict = 'OrangeWidgets/TextData/hrvatski_rjecnik.fsa'
    engdict = 'OrangeWidgets/TextData/engleski_rjecnik.fsa'
    hrstop = 'OrangeWidgets/TextData/hrvatski_stoprijeci.txt'
    engstop = 'OrangeWidgets/TextData/engleski_stoprijeci.txt'    
    
    lem = lemmatizer.FSALemmatization(engdict)
    for word in loadWordSet(engstop):
        lem.stopwords.append(word)       

    fName = '/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/reuters-exchanges-small1.xml'
    #fName = '/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/Other/test.xml'
    #fName = '/home/mkolar/Docs/Diplomski/repository/orange/HR-learn-norm.xml'

##    a = TextCorpusLoader(fName
##            , lem = lem
####            , wordsPerDocRange = (50, -1)
####            , doNotParse = ['small', 'a']
##            , tags = {"content":"cont"}
##            )
##    df = CategoryDocument(a.data).dataCD
            
    import cPickle
    f = open('allDataCW', 'r')
    data=cPickle.load(f)
    f.close()
    data.setattr("meta_names", "fromText")    
    fs = FeatureSelection(data)
    rem = []
    for ex in fs.data:
        if ex[0] <= 7:
            rem.append(ex.name)
    newData = fs.selectFeatures(list = rem)
    
    
