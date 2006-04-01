import orange, orngMisc

data = orange.ExampleTable("lymphography")

findBest = orngMisc.BestOnTheFly(orngMisc.compare2_firstBigger)

for attr in data.domain.attributes:
    findBest.candidate((orange.MeasureAttribute_gainRatio(attr, data), attr))

print findBest.winner()


findBest = orngMisc.BestOnTheFly(callCompareOn1st = True)
for attr in data.domain.attributes:
    findBest.candidate((orange.MeasureAttribute_gainRatio(attr, data), attr))

print findBest.winner()

findBest = orngMisc.BestOnTheFly()

for attr in data.domain.attributes:
    findBest.candidate(orange.MeasureAttribute_gainRatio(attr, data))

bestIndex = findBest.winnerIndex()
print data.domain[bestIndex],", ", findBest.winner()
