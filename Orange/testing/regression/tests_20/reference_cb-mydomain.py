# Description: Shows an example of an Orange class that cannot be subtyped in Python
# Category:    callbacks to Python
# Classes:     Domain
# Uses:        lenses
# Referenced:  callbacks.htm

import orange, orngTree, orngMisc
tab = orange.ExampleTable(r"lenses.tab")


class MyDomain(orange.Domain):
	def __call__(self, ex):
		ex2 = orange.Domain.__call__(self, ex)
		ex2.setclass("?")
		return ex2

md = MyDomain(tab.domain)
ce1 = orange.Example(md, tab[0])
ce2 = md(tab[0])
print ce1
print ce2