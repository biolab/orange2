# Description: Shows a classifier that makes random decisions
# Category:    classification
# Classes:     RandomClassifier
# Uses:        lenses
# Referenced:  RandomClassifier.htm

import orange, orngTest, orngStat

data = orange.ExampleTable("lenses")

rc = orange.RandomClassifier()
rc.classVar = data.domain.classVar
rc.probabilities = [0.5, 0.3, 0.2]

for i in range(3):
    for ex in data[:5]:
        print ex, rc(ex)
    print