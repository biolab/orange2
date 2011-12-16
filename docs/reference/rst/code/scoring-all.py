# Description: Scoring and selection of best N features
# Category:    feature scoring
# Uses:        voting
# Referenced:  Orange.feature.scoring
# Classes:     Orange.feature.scoring.score_all, Orange.feature.scoring.Relief

import Orange
table = Orange.data.Table("voting")

def print_best_3(ma):
    for m in ma[:3]:
        print "%5.3f %s" % (m[1], m[0])

print 'Feature scores for best three features (with score_all):'
ma = Orange.feature.scoring.score_all(table)
print_best_3(ma)

print

print 'Feature scores for best three features (scored individually):'
meas = Orange.feature.scoring.Relief(k=20, m=50)
mr = [ (a.name, meas(a, table)) for a in table.domain.attributes ]
mr.sort(key=lambda x: -x[1]) #sort decreasingly by the score
print_best_3(mr)




