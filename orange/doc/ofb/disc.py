# Author:      B Zupan
# Version:     1.0
# Description: Entropy based discretization compared to discretization with equal-frequency
#              of instances in intervals
# Category:    preprocessing
# Uses:        iris.tab

import orange

def show_values(data, heading):
    print heading
    for a in data.domain.attributes:
        print a.name, a
        print "%s: %s" % (a.name, reduce(lambda x,y: x+', '+y, [i for i in a.values]))
        
data = orange.ExampleTable("iris")

data_ent = orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())
show_values(data_ent, "Entropy based discretization")
print

data_n = orange.Preprocessor_discretize(data, method=orange.EquiNDiscretization(numberOfIntervals=3))
show_values(data_n, "Equal-frequency intervals")
