import os, os.path, re, cPickle

allWords = {}

def searchWords(path, category, recurse):
    for fn in os.listdir(path):
        nfn = path + "/" + fn
        if os.path.isdir(nfn):
            if recurse:
                searchWords(nfn, category, recurse)
        elif fn[-5:] == ".html" or fn[-4:] == ".htm":
            addWords(nfn, category)


re_word = re.compile(r"\W(?P<word>\w\w\w+)\W")
re_dottedWord = re.compile(r"\W(?P<word>\w+(\.\w+)+)")

def addWord(word, category):
    if allWords.has_key(word):
        if category not in allWords[word]:
            allWords[word].append(category)
    else:
        allWords[word] = [category]
    
def addWords(nfn, category):
    content = file(nfn).read()
    for wm in re_word.finditer(content):
        addWord(wm.group("word"), category)
    for wm in re_dottedWord.finditer(content):
        addWord(wm.group("word"), category)

categories = [("widgets/catalog", 1),
              ("ofb", 1),
              ("modules", 1),
              ("reference", 1),
              ("widgets", 0)]

for category, (path, recurse) in enumerate(categories):
    searchWords(path, category, recurse)

cPickle.dump(allWords, file("wordDict.pickle", "wb"))