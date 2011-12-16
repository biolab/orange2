# Description: Show frequences for values of discrete attributes, count number of instances where attribute is not defined
# Category:    description
# Uses:        adult_sample.tab
# Referenced:  basic_exploration.htm

import orange
data = orange.ExampleTable("../../datasets/adult_sample")
dist = orange.DomainDistributions(data)

print "Average values and mean square errors:"
for i in range(len(data.domain.attributes)):
    if data.domain.attributes[i].varType == orange.VarTypes.Continuous:
        print "%s, mean=%5.2f +- %5.2f" % \
          (data.domain.attributes[i].name, dist[i].average(), dist[i].error())

print "\nFrequencies for values of discrete attributes:"
for i in range(len(data.domain.attributes)):
    a = data.domain.attributes[i]
    if a.varType == orange.VarTypes.Discrete:
        print "%s:" % a.name
        for j in range(len(a.values)):
            print "  %s: %d" % (a.values[j], int(dist[i][j]))

print "\nNumber of instances where attribute is not defined:"
for i in range(len(data.domain.attributes)):
    a = data.domain.attributes[i]
    print "  %2d %s" % (dist[i].unknowns, a.name)
