# Description: Constructs indices for cross-validation
# Category:    sampling
# Classes:     MakeRandomIndices, MakeRandomIndicesCV
# Uses:        lenses
# Referenced:  RandomIndices.htm

import orange

data = orange.ExampleTable("lenses")

print orange.MakeRandomIndicesCV(data)

print orange.MakeRandomIndicesCV(10, folds=5)