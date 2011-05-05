def printTree0(node, classvar, lev):
    var = node.tested

    if node.nodeType == 0:
        print "%s (%.1f)" % (classvar.values[int(node.leaf)], node.items),

    elif node.nodeType == 1:
        for i, val in enumerate(var.values):
            print ("\n"+"|    "*lev + "%s = %s:") % (var.name, val),
            printTree0(node.branch[i], classvar, lev+1)

    elif node.nodeType == 2:
        print ("\n"+"|    "*lev + "%s &lt;= %.1f:") % (var.name, node.cut),
        printTree0(node.branch[0], classvar, lev+1)
        print ("\n"+"|    "*lev + "%s > %.1f:") % (var.name, node.cut),
        printTree0(node.branch[1], classvar, lev+1)

    elif node.nodeType == 3:
        for i, branch in enumerate(node.branch):
            inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
            inset = [var.values[j[0]] for j in inset]
            if len(inset)==1:
                print ("\n"+"|    "*lev + "%s = %s:") % (var.name, inset[0]),
            else:
                print ("\n"+"|    "*lev + "%s in {%s}:") % (var.name, 
                    reduce(lambda x,y:x+", "+y, inset)),
            printTree0(branch, classvar, lev+1)

def printTree(tree):
    printTree0(tree.tree, tree.classVar, 0)
    print
