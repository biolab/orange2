# Description: Constructs indices for sampling procedure in which examples are divided in more than two groups
# Category:    sampling
# Classes:     MakeRandomIndices, MakeRandomIndicesN
# Uses:        lenses
# Referenced:  RandomIndices.htm

import Orange

data = Orange.data.Table("lenses")

indicesn = Orange.data.sample.SubsetIndicesN(p=[0.5, 0.25])

ind = indicesn(data)
print ind

indicesn = Orange.data.sample.SubsetIndicesN(p=[12, 6])

ind = indicesn(data)
print ind
