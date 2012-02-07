import orange, orngPade, orngTree
reload(orngPade)

#data = orange.ExampleTable("c:\\d\\x2.txt")
data = orange.ExampleTable("c:\\D\\ai\\Orange\\test\\squin\\xyz-t")
##pa, cid, did = orngPade.pade(data, [0, 2], orngPade.tubedRegression)
##print pa[0]
##
##pa, cid, did = orngPade.pade(data, [0, 2], orngPade.tubedRegression, derivativeAsMeta=1, outputAttr=1)
##print pa[0]
##
##pa, cid, did = orngPade.pade(data, [0, 2], orngPade.tubedRegression, derivativeAsMeta=1, differencesAsMeta=1, originalAsMeta=1, outputAttr=1)
##print pa[0]
##
##pa, cid, did = orngPade.pade(data, [0, 2], orngPade.tubedRegression, derivativeAsMeta=1, originalAsMeta=1)
##print pa[0]
##


## Testiranje na ucnih podatkih

pa, qid, did, cid = orngPade.pade(data, [2], orngPade.firstTriangle, originalAsMeta=True)
tree = orngTree.TreeLearner(pa, maxDepth=4)
orngTree.printTxt(tree)

print orngPade.computeAmbiguityAccuracy(tree, pa, cid)



## Precno preverjanje

cv = orange.MakeRandomIndicesCV(data, 10)
amb = acc = 0.0
for fold in range(10):
    train = data.select(cv, fold, negate=1)
    test = data.select(cv, fold)
    pa, qid, did, cid = orngPade.pade(train, [0, 2], originalAsMeta=True)
    tree = orngTree.TreeLearner(pa, maxDepth=4)

    mb, cc = orngPade.computeAmbiguityAccuracy(tree, test, -1)
    amb += mb
    acc += cc
    print (mb, cc)
print amb/10, acc/10

