# Description: Demonstrates the use of BestOnTheFly class from Orange.misc.selection
# Category:    misc, selection
# Classes:     BestOnTheFly
# Uses:        lymphography.tab
# Referenced:  

import Orange

table = Orange.data.Table("lymphography")

findBest = Orange.misc.selection.BestOnTheFly(callCompareOn1st = True)

for attr in table.domain.attributes:
    findBest.candidate((Orange.feature.scoring.GainRatio(attr, table), attr))

print "%5.3f: %s" % findBest.winner()

findBest = Orange.misc.selection.BestOnTheFly(Orange.misc.selection.compareFirstBigger)

for attr in table.domain.attributes:
    findBest.candidate((Orange.feature.scoring.GainRatio(attr, table), attr))

print "%5.3f: %s" % findBest.winner()

findBest = Orange.misc.selection.BestOnTheFly()

for attr in table.domain.attributes:
    findBest.candidate(Orange.feature.scoring.GainRatio(attr, table))

bestIndex = findBest.winnerIndex()
print "%5.3f: %s" % (findBest.winner(), table.domain[bestIndex])