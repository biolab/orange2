# Description: Shows how to derive a class from orange.Filter
# Category:    filters, preprocessing, callbacks to Python
# Classes:     Filter
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree, orngMisc
tab = orange.ExampleTable(r"lenses.tab")

filt = orange.Filter(lambda ex:ex["age"]=="young")
for e in tab.select(filt):
    print e

class FilterYoung(orange.Filter):
    def __call__(self, ex):
        return ex["age"]=="young"

print "\n"
for e in tab.select(FilterYoung()):
    print e


