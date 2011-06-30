# Description: Read data, show mean for continuous attributes and contingency matrix for nominal attributes
# Category:    description
# Uses:        adult_sample.tab
# Classes:     DomainContingency
# Referenced:  basic_exploration.htm

import orange
data = orange.ExampleTable("../../datasets/adult_sample")

print "Continuous attributes:"
for a in range(len(data.domain.attributes)):
    if data.domain.attributes[a].varType == orange.VarTypes.Continuous:
        d = 0.; n = 0
        for e in data:
            if not e[a].isSpecial():
                d += e[a]
                n += 1
        print "  %s, mean=%3.2f" % (data.domain.attributes[a].name, d/n)

print "\nNominal attributes (contingency matrix for classes:", data.domain.classVar.values, ")"
cont = orange.DomainContingency(data)
for a in data.domain.attributes:
    if a.varType == orange.VarTypes.Discrete:
        print "  %s:" % a.name
        for v in range(len(a.values)):
            sum = 0
            for cv in cont[a][v]:
                sum += cv
            print "    %s, total %d, %s" % (a.values[v], sum, cont[a][v])
        print
