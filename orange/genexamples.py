def domainByLimits(limits, add=0):
  import orange
  attrlist=[orange.EnumVariable(chr(97+li), values=map(repr, range(add, add+limits[li]))) for li in range(len(limits))]
  attrlist[-1].name="y"
  return orange.Domain(attrlist)

def examplesByFunction(dom, function):
  import orange, orngMisc

  tab=orange.ExampleTable(dom)
  for attributes in orngMisc.LimitedCounter([len(attr.values) for attr in dom.attributes]):
    tab.append(orange.Example(dom, attributes + [function(attributes)]))
  return tab

def randomExamplesByFunction(domain, function, N):
  import orange

  tab=orange.ExampleTable(domain)
  for i in range(N):
    example=[attr.randomvalue() for attr in domain.attributes]
    tab.append(orange.Example(domain, example+ [function(example)]))
  return tab
