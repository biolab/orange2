def showBranch(node, classvar, lev, i):
    var = node.tested
    if node.nodeType == 1:
        print ("\n"+"|   "*lev + "%s = %s:") % (var.name, var.values[i]),
        printTree0(node.branch[i], classvar, lev+1)
    elif node.nodeType == 2:
        print ("\n"+"|   "*lev + "%s %s %.1f:") % (var.name, ["<=", ">"][i], node.cut),
        printTree0(node.branch[i], classvar, lev+1)
    else:
        inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
        inset = [var.values[j[0]] for j in inset]
        if len(inset)==1:
            print ("\n"+"|   "*lev + "%s = %s:") % (var.name, inset[0]),
        else:
            print ("\n"+"|   "*lev + "%s in {%s}:") % (var.name, ", ".join(inset)),
        printTree0(node.branch[i], classvar, lev+1)
        
        
def printTree0(node, classvar, lev):
    var = node.tested
    if node.nodeType == 0:
        print "%s (%.1f)" % (classvar.values[int(node.leaf)], node.items),
    else:
        for i, branch in enumerate(node.branch):
            if not branch.nodeType:
                showBranch(node, classvar, lev, i)
        for i, branch in enumerate(node.branch):
            if branch.nodeType:
                showBranch(node, classvar, lev, i)

def printTree(tree):
    printTree0(tree.tree, tree.classVar, 0)
    print