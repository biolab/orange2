# Description: Scoring and selection of best N features
# Category:    feature scoring
# Uses:        voting
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.att_measure, Orange.features.scoring.GainRatio

import Orange
table = Orange.data.Table("voting")

print 'Feature scores for best three features:'
ma = Orange.feature.scoring.att_measure(table)
for m in ma[:3]:
    print "%5.3f %s" % (m[1], m[0])
