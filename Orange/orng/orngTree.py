from Orange.classification.tree import *

from Orange.classification.tree import _countNodes, _countLeaves,\
    _TreeClassifier, _TreeDumper

def countNodes(self):
    """
    DEPRECATED. Return the number of nodes of tree.
    """
    return _countNodes(self.tree if isinstance(self, _TreeClassifier) or \
        isinstance(self, TreeClassifier) else self)

def countLeaves(self):
    """
    DEPRECATED. Return the number of leaves in the tree.
    """
    return _countLeaves(self.tree if isinstance(self, _TreeClassifier) or \
        isinstance(self, TreeClassifier) else self)

def dump(self, leafStr = "", nodeStr = "", **argkw):  
    """
    DEPRECATED. Replaced by :obj:`TreeClassifier.dump`.
    """
    return _TreeDumper(leafStr, nodeStr, argkw.get("userFormats", []) + 
        _TreeDumper.defaultStringFormats, argkw.get("minExamples", 0), 
        argkw.get("maxDepth", 1e10), argkw.get("simpleFirst", True),
        self).dumpTree()

def dot(self, fileName, leafStr = "", nodeStr = "", leafShape="plaintext", nodeShape="plaintext", **argkw):
    """
    DEPRECATED. Replaced by :obj:`TreeClassifier.dump`.
    """
    fle = type(fileName) == str and file(fileName, "wt") or fileName

    _TreeDumper(leafStr, nodeStr, argkw.get("userFormats", []) + 
        _TreeDumper.defaultStringFormats, argkw.get("minExamples", 0), 
        argkw.get("maxDepth", 1e10), argkw.get("simpleFirst", True), self,
        leafShape=leafShape, nodeShape=nodeShape, fle=fle).dotTree()

dumpTree = dump
""" DEPRECATED: Replaced by :obj:`TreeClassifier.dump`. """

def printTree(*a, **aa):
    """
    DEPRECATED Print out the tree (call :func:`dumpTree` with the same
    arguments and print out the result).
    """
    print dumpTree(*a, **aa)

printTxt = printTree
""" DEPRECATED. Replaced by :obj:`TreeClassifier.dump` """

printDot = dot
""" DEPRECATED. Replaced by :obj:`TreeClassifier.dot` """

dotTree = printDot
""" DEPRECATED. Replaced by :obj:`TreeClassifier.dot` """

byWhom = by_whom
insertStr = insert_str
insertDot = insert_dot
insertNum = insert_num
