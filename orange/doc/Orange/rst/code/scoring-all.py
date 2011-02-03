# Description: Scoring and selection of best N features
# Category:    feature scoring
# Uses:        voting
# Referenced:  Orange.feature.html#scoring
# Classes:     Orange.feature.scoring.attMeasure, Orange.features.scoring.gainRatio

import Orange
table = Orange.data.Table("voting")

print 'Feature scores for best three features:'
ma = Orange.feature.scoring.attMeasure(table)
for m in ma[:3]:
    print "%5.3f %s" % (m[1], m[0])
