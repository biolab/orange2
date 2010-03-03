# Description: Shows how to construct and use orange.Value
# Category:    basic classes
# Classes:     Value
# Uses:        
# Referenced:  Value.htm

import orange

def err():
    raise "Error"

fruit = orange.EnumVariable("fruit", values = ["plum", "apple", "lemon"])
iq = orange.FloatVariable("iq")
lm = orange.Value(fruit, "lemon")
ap = orange.Value(fruit, 1)
un = orange.Value(fruit)

Mary = orange.Value(iq, "105")
Harry = orange.Value(iq, 80)
Dick = orange.Value(iq)

sf = orange.Value(2)
Sally = orange.Value(118.0)

sf.variable = fruit


city = orange.Value(orange.StringValue("Cicely"))



if (lm!="lemon"): raise error
if (lm<"apple"): raise error
if (orange.Value(1)>lm): raise error


deg3 = orange.EnumVariable(values=["little", "medium", "big"])
deg4 = orange.EnumVariable(values=["tiny", "little", "big", "huge"])

val3 = orange.Value(deg3)
val4 = orange.Value(deg4)

val3.value = "medium"
val4.value = "little"
print val3<val4, val3<=val4, val3==val4, val3>=val4, val3>val4, val3!=val4
print val4<val3, val4<=val3, val4==val3, val4>=val3, val4>val3, val4!=val3

val3.value = "medium"
val4.value = "huge"
#print val3<val4

degb = orange.EnumVariable(values=["black", "gray", "white"])
degd = orange.EnumVariable(values=["white", "gray", "black"])
print orange.Value(degb, "black") == orange.Value(degd, "black")
try:
    print orange.Value(degb, "black") < orange.Value(degd, "white")
except:
    print """'orange.Value(degb, "black") < orange.Value(degd, "white")' failed (as it should)"""

print 
