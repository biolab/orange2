# Description: Shows how to derive a tree descender from orange.TreeDescender
# Category:    classification, decision trees, callbacks to Python
# Classes:     TreeDescender
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree

import random
random.seed(0)

data = orange.ExampleTable("lenses")

treeLearner = orange.TreeLearner()
tree = treeLearner(data)
orngTree.printTxt(tree)

class Descender_RandomBranch(orange.TreeDescender):
    def __call__(self, node, example):
        while node.branchSelector:
            branch = node.branchSelector(example)
            if branch.isSpecial()  or int(branch)>=len(node.branches):
                branch = random.randint(0, len(node.branches)-1)
                print "Descender decides for ", branch
            nextNode = node.branches[int(branch)]
            if not nextNode:
                break
            node = nextNode
        return node

ex = orange.Example(data.domain, data[3])
ex[tree.tree.branchSelector.classVar] = "?"
print ex

print "\n\nDecisions by random branch choice"
tree.descender = Descender_RandomBranch()
for i in range(3):
    print tree(ex)



class Descender_RandomVote(orange.TreeDescender):
  def __call__(self, node, example):
    while node.branchSelector:
      branch = node.branchSelector(example)
      if branch.isSpecial() or int(branch)>=len(node.branches):
        votes = orange.DiscDistribution([random.randint(0, 100) for i in node.branches])
        votes.normalize()
        print "Weights:", votes
        return node, votes
      nextNode = node.branches[int(branch)]
      if not nextNode:
        break
      node = nextNode
    return node

print "\n\nDecisions by random voting"
tree.descender = Descender_RandomVote()
print tree(ex, orange.GetProbabilities)


class Descender_Report(orange.TreeDescender):
    def __call__(self, node, example):
        print "Descent: root ",
        while node.branchSelector:
            branch = node.branchSelector(example)
            if branch.isSpecial() or int(branch)>=len(node.branches):
                break
            nextNode = node.branches[int(branch)]
            if not nextNode:
                break
            print ">> %s = %s" % (node.branchSelector.classVar.name, node.branchDescriptions[int(branch)]),
            node = nextNode
        return node

print "\n\nReporting descender"
tree.descender = Descender_Report()
print "Classifying example", data[0]
print "----> %s" % tree(data[1])

tree.descender = Descender_RandomBranch()
ex = orange.Example(data.domain, list(data[2]))
ex[tree.tree.branchSelector.classVar] = "?"
for i in range(5):
    print tree(ex, orange.Classifier.GetProbabilities)


tree.descender = Descender_RandomVote()
ex = orange.Example(data.domain, list(data[2]))
ex[tree.tree.branchSelector.classVar] = "?"
print tree(ex, orange.GetProbabilities)

#orngTree.printModel(tree)

