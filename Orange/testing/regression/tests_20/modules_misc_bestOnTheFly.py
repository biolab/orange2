import orange, orngMisc

data = orange.ExampleTable("zoo")

findBest = orngMisc.BestOnTheFly(orngMisc.compare2_firstBigger)

for attr in data.domain.attributes:
    findBest.candidate((orange.MeasureAttribute_gainRatio(attr, data), attr))

print "%5.3f: %s" % findBest.winner()


findBest = orngMisc.BestOnTheFly(callCompareOn1st=True)
for attr in data.domain.attributes:
    findBest.candidate((orange.MeasureAttribute_gainRatio(attr, data), attr))

print "%5.3f: %s" % findBest.winner()

findBest = orngMisc.BestOnTheFly()

for attr in data.domain.attributes:
    findBest.candidate(orange.MeasureAttribute_gainRatio(attr, data))

bestIndex = findBest.winnerIndex()
print "%5.3f: %s" % (findBest.winner(), data.domain[bestIndex])
