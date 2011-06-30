import Orange
import re

data = Orange.data.Table("iris")
tree = Orange.classification.tree.TreeLearner(data, maxDepth=3)

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
        whom = Orange.classification.tree.byWhom(by, parent, tree)
        if whom and whom.distribution:
            divMargin = getMargin(whom.distribution)
            if divMargin > 1e-30:
                margin /= divMargin
            else:
                Orange.classification.tree.insertDot(strg, mo)
        else:
            return Orange.classification.tree.insertDot(strg, mo)
    return Orange.classification.tree.insertNum(strg, mo, margin)
    
myFormat = [(re.compile("%"+Orange.classification.tree.fs
    +"B"+Orange.classification.tree.by), replaceB)]
            
print tree.dump(leafStr="%V %^B% (%^3.2BbP%)", userFormats=myFormat)
