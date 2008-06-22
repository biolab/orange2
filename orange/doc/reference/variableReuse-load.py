import orange
import warnings

warnings.simplefilter('ignore', orange.KernelWarning)

MakeStatus = orange.Variable.MakeStatus

base = orange.ExampleTable("reuse-load-1.tab")
b2 = orange.ExampleTable("reuse-load-1.tab")
print b2.attributeLoadStatus
print " "+"  ".join(["N "[x==y] for x,y in zip(base.domain, b2.domain)])

for status in range(5):
    del base
    del b2
    del x
    del y
    base = orange.ExampleTable("reuse-load-1.tab")
    b2 = orange.ExampleTable("reuse-load-2.tab", createNewOn=status)
    print status
    print base.attributeLoadStatus
    print b2.attributeLoadStatus
    print " "+"  ".join(["N "[x==y] for x,y in zip(base.domain, b2.domain)])
    print
