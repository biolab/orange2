# Description: Tests different ways for building a domain
# Category:    basic classes
# Classes:     Domain
# Uses:        
# Referenced:  Domain.htm

import orange

a, b, c = [orange.EnumVariable(x) for x in ["a", "b", "c"]]
BOD = "bug in orange.Domain"

d = orange.Domain([a, b, c])
if d.attributes != [a, b] or d.variables != [a, b, c] or d.classVar != c:
    raise BOD

d = orange.Domain([a, b], c)
if d.attributes != [a, b] or d.variables != [a, b, c] or d.classVar != c:
    raise BOD

d = orange.Domain([a, b, c], 0)
if d.attributes != [a, b, c] or d.variables != d.attributes or d.classVar != None:
    raise BOD

d = orange.Domain([a, b, c], 1)
if d.attributes != [a, b] or d.variables != [a, b, c] or d.classVar != c:
    raise BOD

d1 = orange.Domain([a, b])
d2 = orange.Domain(["a", b, c], d1)
if d2.attributes != [a, b] or d2.variables != [a, b, c] or d2.classVar != c:
    raise BOD

d1 = orange.Domain([a, b])
d2 = orange.Domain(["a", b, c], 0, [a, b, c])
if d2.attributes != [a, b, c] or d2.variables != d2.attributes or d2.classVar != None:
    raise BOD

d1 = orange.Domain([a, b])
d2 = orange.Domain(["a", b, c], 1, [a, b, c])
if d2.attributes != [a, b] or d2.variables != [a, b, c] or d2.classVar != c:
    raise BOD

d2 = orange.Domain(d1)
if d1==d2 or d1.attributes != d2.attributes or d1.variables != d2.variables or d1.classVar != d2.classVar:
    raise BOD

d2 = orange.Domain(d1, 0)
if d1.variables!=d2.variables or d2.classVar:
    raise BOD

d2 = orange.Domain(d1, 1)
if d1==d2 or d1.attributes != d2.attributes or d1.variables != d2.variables or d1.classVar != d2.classVar:
    raise BOD

d2 = orange.Domain(d1, a)
if d2.attributes != [b] or d2.variables != [b, a] or d2.classVar != a:
    raise BOD

d2 = orange.Domain(d1, c)
if d2.attributes != [a, b] or d2.variables != [a, b, c] or d2.classVar != c:
    raise BOD

d1 = orange.Domain([a, b], 0)
d2 = orange.Domain(d1, c)
if d2.attributes != [a, b] or d2.variables != [a, b, c] or d2.classVar != c:
    raise BOD

d1 = orange.Domain([a, b], 1)
d2 = orange.Domain(d1, 0)
if d2.attributes != [a, b] or d2.variables != [a, b] or d2.classVar:
    raise BOD
