# Description: Entropy based discretization compared to discretization with equal-frequency
#              of instances in intervals
# Category:    preprocessing
# Uses:        wdbc.tab
# Classes:     Preprocessor_discretize, EntropyDiscretization
# Referenced:  o_categorization.htm

import orange

def show_values(data, heading):
  for a in data.domain.attributes:
    print "%s/%d: %s" % (a.name, len(a.values), reduce(lambda x,y: x+', '+y, [i for i in a.values]))
        
data = orange.ExampleTable("../datasets/wdbc")
print '%d features in original data set, discretized:' % len(data.domain.attributes)
data_ent = orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())
show_values(data_ent, "Entropy based discretization")

print '\nFeatures with sole value after discretization:'
for a in data_ent.domain.attributes:
  if len(a.values)==1:
    print a.name

import orngDisc
reload(orngDisc)
data_ent2 = orngDisc.entropyDiscretization(data)
print '%d features after removing features discretized to a constant value' % len(data_ent2.domain.attributes)
