# Description: Shows how to construct and use PythonVariables
# Category:    basic classes
# Classes:     PythonVariable
# Uses:        
# Referenced:  pythonvariable.htm

# xtest: RANDOM

import orange

data = orange.ExampleTable("lenses")

newattr = orange.PythonVariable("foo")
data.domain.addmeta(orange.newmetaid(), newattr)

data[0]["foo"] = ("a", "tuple")
data[1]["foo"] = "a string"

# Could, but won't do this; the script is also used
# in regression tests and printing the pointer would
# cause a mismatch btw Win, Linux and Mac version;
# printing the module would cause a mismatch in the path
# data[2]["foo"] = orange
# data[3]["foo"] = data

for i in range(4):
    print data[i]
print

def extolist(ex, wh=0):
#    return orange.PythonValue(map(int, ex))
    return map(int, ex)

listvar = orange.PythonVariable("a_list")
listvar.getValueFrom = extolist
listvar.getValueFrom.classVar = listvar

newdomain = orange.Domain(data.domain.attributes + [listvar], data.domain.classVar)
newdata = orange.ExampleTable(newdomain, data)
for i in newdata:
    print i

