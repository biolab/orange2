# Description: Demonstrates the use of BestOnTheFly class from Orange.misc.selection
# Category:    misc, selection
# Classes:     BestOnTheFly
# Uses:        lymphography.tab
# Referenced:  

import Orange

lymphography = Orange.data.Table("zoo")

find_best = Orange.misc.selection.BestOnTheFly(call_compare_on_1st=True)

for attr in lymphography.domain.attributes:
    find_best.candidate((Orange.feature.scoring.GainRatio(attr, lymphography), attr))

print "%5.3f: %s" % find_best.winner()

find_best = Orange.misc.selection.BestOnTheFly(Orange.misc.selection.compare_first_bigger)

for attr in lymphography.domain.attributes:
    find_best.candidate((Orange.feature.scoring.GainRatio(attr, lymphography), attr))

print "%5.3f: %s" % find_best.winner()

find_best = Orange.misc.selection.BestOnTheFly()

for attr in lymphography.domain.attributes:
    find_best.candidate(Orange.feature.scoring.GainRatio(attr, lymphography))

best_index = find_best.winner_index()
print "%5.3f: %s" % (find_best.winner(), lymphography.domain[best_index])
