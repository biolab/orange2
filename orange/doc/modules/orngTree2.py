import orange, orngTree, re
reload(orngTree)

data = orange.ExampleTable("iris")
tree = orngTree.TreeLearner(data, maxDepth=3)

def getMargin(dist):
    if dist.abs < 1e-30:
        return 0
    l = list(dist)
    l.sort()
    return (l[-1] - l[-2]) / dist.abs

def replaceB(strg, mo, node, parent, tree):
    margin = getMargin(node.distribution)

    by = mo.group("by")
    if margin and by:
        whom = orngTree.byWhom(by, parent, tree)
        if whom and whom.distribution:
            divMargin = getMargin(whom.distribution)
            if divMargin > 1e-30:
                margin /= divMargin
            else:
                orngTree.insertDot(strg, mo)
        else:
            return orngTree.insertDot(strg, mo)
    return orngTree.insertNum(strg, mo, margin)


    
myFormat = [(re.compile("%"+orngTree.fs+"B"+orngTree.by), replaceB)]
            
orngTree.printTree(tree, leafStr="%V %^B% (%^3.2BbP%)", userFormats = myFormat)
